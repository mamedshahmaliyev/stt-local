#!/usr/bin/env python3
"""
voice_stt.py — local English speech-to-text popup for Apple Silicon Macs.

  * Parakeet TDT-CTC 110M (MLX) for transcription
  * Silero VAD cuts speech into sentences at your pauses
  * Terminal UI: live transcript + clickable buttons / keys
  * Optional (--window) floating, half-transparent, always-on-top popup

Global hotkeys (need Accessibility permission):
  Ctrl Ctrl Ctrl     tap Control three times       → start / stop listening
  Ctrl Ctrl Shift    tap Control twice, then Shift → copy text to clipboard
  Shift Shift Ctrl   tap Shift twice, then Control → clear text

Usage:
  python voice_stt.py                 # terminal UI (press space to listen)
  python voice_stt.py --window        # also show the floating popup
  python voice_stt.py --autostart     # start listening immediately
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

SR = 16_000            # models expect 16 kHz mono
BLOCK = 512            # Silero VAD frame size at 16 kHz (32 ms)
HOME = Path(os.environ.get("VOICE_STT_HOME", Path.home() / ".voice_stt"))
TRANSCRIPTS = HOME / "transcripts"
SETTINGS = HOME / "settings.json"
DEFAULT_MODEL = "mlx-community/parakeet-tdt_ctc-110m"


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except Exception:
        return {}


def save_settings(**changes):
    data = load_settings()
    data.update(changes)
    try:
        HOME.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"(note: couldn't save settings: {e})", file=sys.stderr)


# ─────────────────────────────── engine ──────────────────────────────────

class Engine(threading.Thread):
    """Mic → VAD sentences → Parakeet → callbacks. Mic is open only while listening."""

    def __init__(self, args, on_text, on_status, on_ready):
        super().__init__(daemon=True)
        self.args = args
        self.on_text = on_text
        self.on_status = on_status
        self.on_ready = on_ready
        self.listening = threading.Event()
        self.stopped = threading.Event()
        self.ready = False
        TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
        self.transcript_path = TRANSCRIPTS / f"{datetime.now():%Y-%m-%d_%H%M%S}.txt"

    def _load(self):
        self.on_status("Loading speech model…")
        from parakeet_mlx import from_pretrained
        self.asr = from_pretrained(self.args.model)

        self.on_status("Loading VAD…")
        import torch
        from silero_vad import load_silero_vad, VADIterator
        self.torch = torch
        self.vad = VADIterator(
            load_silero_vad(),
            threshold=0.5,
            sampling_rate=SR,
            min_silence_duration_ms=self.args.silence_ms,
            speech_pad_ms=150,
        )

    def _transcribe(self, audio: np.ndarray):
        if len(audio) / SR < 0.4:
            return
        self.on_status("Transcribing…")
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, audio, SR)
            text = self.asr.transcribe(tmp).text.strip()
        finally:
            os.unlink(tmp)
        if text:
            self.on_text(text)
            with open(self.transcript_path, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        self.on_status("Listening" if self.listening.is_set() else "Stopped")

    def _listen_session(self):
        """One Start→Stop session. Opens the mic, closes it on Stop."""
        import sounddevice as sd

        audio_q: queue.Queue[np.ndarray] = queue.Queue()
        max_blocks = int(self.args.max_segment * SR / BLOCK)
        preroll: deque[np.ndarray] = deque(maxlen=10)   # ~0.3 s before speech start
        seg: list[np.ndarray] | None = None
        self.vad.reset_states()

        def cb(indata, frames, t, status):
            audio_q.put(indata[:, 0].copy())

        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            blocksize=BLOCK, callback=cb):
            self.on_status("Listening")
            while self.listening.is_set() and not self.stopped.is_set():
                try:
                    chunk = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if len(chunk) != BLOCK:
                    continue
                ev = self.vad(self.torch.from_numpy(chunk))
                if seg is None:
                    preroll.append(chunk)
                    if ev and "start" in ev:
                        seg = list(preroll)
                        preroll.clear()
                else:
                    seg.append(chunk)
                    if ev and "end" in ev:
                        audio, seg = np.concatenate(seg), None
                        self._transcribe(audio)
                    elif len(seg) >= max_blocks:      # very long sentence: cut, keep going
                        audio, seg = np.concatenate(seg), []
                        self._transcribe(audio)

        # Stop pressed mid-sentence: don't lose the last words.
        if seg:
            self._transcribe(np.concatenate(seg))
        self.on_status("Stopped")

    def run(self):
        try:
            self._load()
        except Exception as e:
            self.on_status(f"Error: {e}")
            return
        self.ready = True
        self.on_ready()
        self.on_status("Ready — press Start")
        while not self.stopped.is_set():
            if self.listening.wait(timeout=0.2):
                try:
                    self._listen_session()
                except Exception as e:
                    self.listening.clear()
                    self.on_status(f"Error: {e}")


# ─────────────────────────────── global hotkeys ──────────────────────────

class TapHotkeys:
    """Detects modifier-key *taps* system-wide (macOS NSEvent monitors).

    Ctrl Ctrl Ctrl     → toggle (start / stop listening)
    Ctrl Ctrl Shift    → copy
    Shift Shift Ctrl   → clear

    A tap = press and release of Control/Shift alone (no other key in between).
    Taps must follow each other within TAP_GAP seconds.
    """
    TAP_MAX = 0.35   # longest press that still counts as a tap
    TAP_GAP = 0.5    # longest pause between taps in a sequence

    CTRL_CODES = (59, 62)    # left / right Control
    SHIFT_CODES = (56, 60)   # left / right Shift

    def __init__(self, on_toggle, on_copy, on_clear):
        self.patterns = {
            ("ctrl", "ctrl", "ctrl"): on_toggle,
            ("ctrl", "ctrl", "shift"): on_copy,
            ("shift", "shift", "ctrl"): on_clear,
        }
        self.taps: list[tuple[str, float]] = []
        self.down: tuple[str, float] | None = None
        self.monitors = []

    def start(self) -> bool:
        try:
            from AppKit import (NSEvent, NSEventMaskFlagsChanged, NSEventMaskKeyDown,
                                NSEventModifierFlagCommand, NSEventModifierFlagControl,
                                NSEventModifierFlagOption, NSEventModifierFlagShift)
        except Exception as e:
            print(f"(note: global hotkeys unavailable: {e})", file=sys.stderr)
            return False
        self.CTRL = NSEventModifierFlagControl
        self.SHIFT = NSEventModifierFlagShift
        self.ANY = (NSEventModifierFlagControl | NSEventModifierFlagShift
                    | NSEventModifierFlagOption | NSEventModifierFlagCommand)
        mask = NSEventMaskFlagsChanged | NSEventMaskKeyDown

        def local(ev):          # events while our popup is focused
            self._handle(ev)
            return ev

        self.monitors = [
            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, self._handle),
            NSEvent.addLocalMonitorForEventsMatchingMask_handler_(mask, local),
        ]
        return True

    def _reset(self):
        self.taps.clear()
        self.down = None

    def _handle(self, ev):
        try:
            from AppKit import NSEventTypeFlagsChanged
            if ev.type() != NSEventTypeFlagsChanged:   # a normal key: not a tap sequence
                self._reset()
                return
            code, flags, now = ev.keyCode(), ev.modifierFlags(), time.monotonic()
            if code in self.CTRL_CODES:
                key, bit = "ctrl", self.CTRL
            elif code in self.SHIFT_CODES:
                key, bit = "shift", self.SHIFT
            else:                                       # Cmd / Option / Fn …
                self._reset()
                return

            if flags & bit:                             # pressed
                if flags & self.ANY & ~bit:             # chorded with another modifier
                    self._reset()
                else:
                    self.down = (key, now)
                return

            # released
            if not self.down or self.down[0] != key or now - self.down[1] > self.TAP_MAX:
                self._reset()
                return
            self.down = None
            if self.taps and now - self.taps[-1][1] > self.TAP_GAP:
                self.taps.clear()
            self.taps.append((key, now))
            seq = tuple(k for k, _ in self.taps[-3:])
            if (action := self.patterns.get(seq)):
                self.taps.clear()
                action()
            del self.taps[:-3]
        except Exception as e:
            print(f"(hotkey error: {e})", file=sys.stderr)


def ensure_accessibility() -> bool:
    """Global key monitors only receive events if this process is trusted for
    Accessibility. Shows the macOS permission prompt if it isn't."""
    try:
        import ctypes
        import objc
        from Foundation import NSDictionary
        ax = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        ax.AXIsProcessTrustedWithOptions.restype = ctypes.c_bool
        ax.AXIsProcessTrustedWithOptions.argtypes = [ctypes.c_void_p]
        opts = NSDictionary.dictionaryWithObject_forKey_(True, "AXTrustedCheckOptionPrompt")
        return bool(ax.AXIsProcessTrustedWithOptions(objc.pyobjc_id(opts)))
    except Exception:
        return True   # can't tell; assume OK


# ─────────────────────────────── terminal UI ─────────────────────────────

class TerminalUI:
    """The main UI: status, clickable buttons and live transcript in the terminal.
    Keys: space/s start-stop · c copy · x clear · m minimize window · q quit.

    A reader thread puts actions on a queue; the GUI thread calls tick() on a
    timer, which runs the actions and redraws only when something changed."""

    KEYS = {" ": "toggle", "s": "toggle", "c": "copy", "x": "clear",
            "m": "minimize", "q": "quit", "\x03": "quit"}
    MOUSE = __import__("re").compile(r"\x1b\[<(\d+);(\d+);(\d+)([Mm])")

    def __init__(self, get_state, do_action, transcript_path):
        self.get_state = get_state
        self.do_action = do_action
        self.transcript_path = transcript_path
        self.actions: queue.Queue[str] = queue.Queue()
        self.spans: list[tuple[int, int, str]] = []
        self.last = None
        self.active = False
        self.fd = sys.stdin.fileno()

    @staticmethod
    def available() -> bool:
        return sys.stdin.isatty() and sys.stdout.isatty()

    # ── lifecycle
    def start(self):
        import atexit
        import termios
        self.old_attrs = termios.tcgetattr(self.fd)
        new = termios.tcgetattr(self.fd)
        new[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)   # keys one by one; Ctrl+C → q
        new[6][termios.VMIN], new[6][termios.VTIME] = 1, 0
        termios.tcsetattr(self.fd, termios.TCSANOW, new)
        # alt screen, hide cursor, SGR mouse clicks
        self._write("\033[?1049h\033[?25l\033[?1000h\033[?1006h")
        self.active = True
        atexit.register(self.stop)
        threading.Thread(target=self._read_loop, daemon=True).start()

    def stop(self):
        if not self.active:
            return
        self.active = False
        import termios
        self._write("\033[?1000l\033[?1006l\033[?25h\033[?1049l")
        termios.tcsetattr(self.fd, termios.TCSANOW, self.old_attrs)
        text = self.get_state()["text"].strip()
        if text:
            print(text + "\n")
        print(f"Transcript saved to: {self.transcript_path}")

    @staticmethod
    def _write(s: str):
        sys.stdout.write(s)
        sys.stdout.flush()

    # ── input (reader thread)
    def _read_loop(self):
        import select
        while self.active:
            r, _, _ = select.select([self.fd], [], [], 0.2)
            if not r:
                continue
            data = os.read(self.fd, 256).decode("utf-8", "ignore")
            for m in self.MOUSE.finditer(data):
                btn, x, y, kind = int(m[1]), int(m[2]), int(m[3]), m[4]
                if kind == "M" and btn == 0 and y == 2:
                    for x0, x1, action in self.spans:
                        if x0 <= x <= x1:
                            self.actions.put(action)
            data = self.MOUSE.sub("", data)
            data = __import__("re").sub(r"\x1b\[[0-9;?]*[A-Za-z~]|\x1b.", "", data)  # arrows etc.
            for ch in data:
                if (action := self.KEYS.get(ch.lower() if ch.isalpha() else ch)):
                    self.actions.put(action)

    # ── GUI thread
    def tick(self):
        while not self.actions.empty():
            self.do_action(self.actions.get_nowait())
        if not self.active:
            return
        import shutil
        st = self.get_state()
        size = shutil.get_terminal_size()
        key = (tuple(sorted(st.items())), size)
        if key != self.last:
            self.last = key
            self._render(st, size.columns, size.lines)

    def _render(self, st, cols, rows):
        import textwrap
        R, DIM = "\033[0m", "\033[2m"
        status = st["status"]
        color = "32" if status == "Listening" else "33" if status.startswith("Transcrib") \
            else "31" if status.startswith("Error") else "90"
        if st["copied"]:
            status, color = "Copied to clipboard ✓", "32"
        out = ["\033[H"]
        title = "Voice STT"
        left = f" \033[{color}m●{R} {status}"
        pad = max(1, cols - len(status) - 3 - len(title) - 1)
        out.append(left + " " * pad + DIM + title + R + "\033[K\r\n")

        # buttons (row 2) — remember column spans for mouse clicks
        listening, ready = st["listening"], st["ready"]
        buttons = [
            ("■ Stop  ␣" if listening else "▶ Start ␣", "toggle",
             "41" if listening else ("42" if ready else "100")),
            ("Copy c", "copy", "100"),
            ("Clear x", "clear", "100"),
            ("Quit q", "quit", "100"),
        ]
        if st["window"]:
            buttons.insert(3, ("Restore m" if st["minimized"] else "Minimize m", "minimize", "100"))
        self.spans, segs, x = [], [], 2
        for label, action, bg in buttons:
            t = f" {label} "
            segs.append(f"\033[{bg};97m{t}{R}")
            self.spans.append((x, x + len(t) - 1, action))
            x += len(t) + 1
        out.append(" " + " ".join(segs) + "\033[K\r\n")
        out.append(DIM + "─" * cols + R + "\033[K\r\n")

        # transcript (last lines that fit)
        body_rows = max(1, rows - 4)
        width = max(10, cols - 2)
        text = st["text"]
        if text:
            lines = []
            for para in text.split("\n"):
                lines += textwrap.wrap(para, width) or [""]
            lines = [" " + ln for ln in lines[-body_rows:]]
        else:
            msg = "Press Start (space) and talk…" if ready else "Loading models…"
            lines = [f" {DIM}{msg}{R}"]
        lines += [""] * (body_rows - len(lines))
        out.extend(ln + "\033[K\r\n" for ln in lines)

        foot = f" ⌃⌃⌃ start/stop · ⌃⌃⇧ copy · ⇧⇧⌃ clear · saved to {self.transcript_path}"
        out.append(DIM + foot[:cols] + R + "\033[K")
        self._write("".join(out))


# ─────────────────────────────── popup UI ────────────────────────────────

ICONS = {
    "play":     '<polygon points="7 4 20 12 7 20 7 4" fill="{c}" stroke="none"/>',
    "stop":     '<rect x="6" y="6" width="12" height="12" rx="2" fill="{c}" stroke="none"/>',
    "copy":     '<rect x="9" y="9" width="12" height="12" rx="2"/>'
                '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
    "check":    '<polyline points="20 6 9 17 4 12"/>',
    "clear":    '<path d="M3 6h18"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>'
                '<path d="M9 6V4h6v2"/>',
    "minimize": '<line x1="6" y1="12" x2="18" y2="12"/>',
    "close":    '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    "opacity":  '<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16z" fill="{c}"/>',
}


def run_app(args):
    from PySide6.QtCore import QByteArray, QObject, QPoint, QSize, Qt, QTimer, Signal
    from PySide6.QtGui import QFont, QIcon, QPainter, QPixmap, QTextCursor
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                                   QPushButton, QSizeGrip, QSlider, QTextEdit,
                                   QVBoxLayout, QWidget)

    def icon(name: str, color: str = "white", size: int = 16) -> QIcon:
        body = ICONS[name].replace("{c}", color)
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
               f'stroke="{color}" stroke-width="2" stroke-linecap="round" '
               f'stroke-linejoin="round">{body}</svg>')
        dpr = 3
        pm = QPixmap(size * dpr, size * dpr)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        QSvgRenderer(QByteArray(svg.encode())).render(p)
        p.end()
        pm.setDevicePixelRatio(dpr)
        return QIcon(pm)

    class Controller(QObject):
        """App state + actions, shared by the terminal UI and the optional window."""
        changed = Signal()
        _text = Signal(str)        # engine thread → GUI thread
        _status = Signal(str)
        _ready = Signal()

        def __init__(self):
            super().__init__()
            self.text = ""
            self.status = "Starting…"
            self.copied_until = 0.0
            self.engine = Engine(args, on_text=self._text.emit, on_status=self._status.emit,
                                 on_ready=self._ready.emit)
            self._text.connect(self.add_text)
            self._status.connect(self.set_status)
            self._ready.connect(self.on_ready)

        @property
        def listening(self) -> bool:
            return self.engine.listening.is_set()

        @property
        def ready(self) -> bool:
            return self.engine.ready

        @property
        def copied(self) -> bool:
            return time.monotonic() < self.copied_until

        def add_text(self, t: str):
            self.text += (" " if self.text else "") + t
            self.changed.emit()

        def set_status(self, s: str):
            self.status = s
            self.changed.emit()

        def on_ready(self):
            self.changed.emit()
            if args.autostart:
                self.toggle()

        def toggle(self):
            if not self.ready:
                return
            if self.listening:
                self.engine.listening.clear()
                self.set_status("Stopping…")
            else:
                self.engine.listening.set()
                self.set_status("Starting mic…")

        def copy(self):
            QApplication.clipboard().setText(self.text)
            self.copied_until = time.monotonic() + 1.2
            self.changed.emit()
            QTimer.singleShot(1300, self.changed.emit)   # end the "copied" flash

        def clear(self):
            self.text = ""
            self.changed.emit()

        def quit(self):
            self.engine.listening.clear()
            self.engine.stopped.set()
            QApplication.quit()

    class Popup(QWidget):
        def __init__(self, ctrl: Controller):
            super().__init__()
            self.ctrl = ctrl
            self._drag: QPoint | None = None

            settings = load_settings()
            opacity = args.opacity if args.opacity is not None else settings.get("opacity", 0.72)
            self.opacity = min(1.0, max(0.1, float(opacity)))

            self.setWindowTitle("Voice STT")
            # A normal window (not Qt.Tool/NSPanel) so it can be minimized into the Dock.
            self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.resize(480, 260)

            self.panel = QFrame(self)
            self.panel.setObjectName("panel")
            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.addWidget(self.panel)

            # ── header: status · [start|clear|copy] · [minimize|close]
            self.dot = QLabel("●")
            self.status = QLabel()

            def btn(name, tip, obj=None, pos="mid"):
                b = QPushButton()
                b.setIcon(icon(name))
                b.setIconSize(QSize(14, 14))
                b.setToolTip(tip)
                b.setCursor(Qt.PointingHandCursor)
                b.setProperty("pos", pos)
                if obj:
                    b.setObjectName(obj)
                return b

            self.listen_btn = btn("play", "Start listening  (⌃ ⌃ ⌃)", "listen", "first")
            self.listen_btn.setProperty("on", "false")
            self.clear_btn = btn("clear", "Clear text  (⇧ ⇧ ⌃)")
            self.copy_btn = btn("copy", "Copy to clipboard  (⌃ ⌃ ⇧)", pos="last")
            self.min_btn = btn("minimize", "Minimize", pos="first")
            close_btn = btn("close", "Quit", "close", "last")

            def group(*buttons):
                g = QFrame()
                g.setObjectName("group")
                lay = QHBoxLayout(g)
                lay.setContentsMargins(0, 0, 0, 0)
                lay.setSpacing(0)
                for b in buttons:
                    lay.addWidget(b)
                return g

            top = QHBoxLayout()
            top.setSpacing(8)
            top.addWidget(self.dot)
            top.addWidget(self.status, 1)
            top.addWidget(group(self.listen_btn, self.clear_btn, self.copy_btn))
            top.addWidget(group(self.min_btn, close_btn))

            # ── transcript
            self.box = QTextEdit(readOnly=True)
            # macOS font smoothing thickens light-on-dark text, so "Regular" reads
            # as semi-bold here; Light weight renders as normal-looking text.
            font = QFont(".AppleSystemUIFont", args.font_size)
            font.setWeight(QFont.Weight.Light)
            font.setBold(False)
            self.box.setFont(font)
            self.box.setPlaceholderText("Press Start and talk…")

            # ── footer: transparency slider · resize grip
            self.footer = QWidget()
            bottom = QHBoxLayout(self.footer)
            bottom.setContentsMargins(0, 0, 0, 0)
            op_icon = QLabel()
            op_icon.setPixmap(icon("opacity", "#9a9aa0", 12).pixmap(12, 12))
            op_icon.setToolTip("Transparency")
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(10, 100)
            self.slider.setValue(int(round(self.opacity * 100)))
            self.slider.setFixedWidth(90)
            self.slider.setToolTip("Transparency")
            bottom.addWidget(op_icon)
            bottom.addWidget(self.slider)
            bottom.addStretch(1)
            bottom.addWidget(QSizeGrip(self.panel))

            lay = QVBoxLayout(self.panel)
            lay.setContentsMargins(14, 10, 10, 6)
            lay.addLayout(top)
            lay.addWidget(self.box, 1)
            lay.addWidget(self.footer)

            self.listen_btn.clicked.connect(ctrl.toggle)
            self.clear_btn.clicked.connect(ctrl.clear)
            self.copy_btn.clicked.connect(ctrl.copy)
            self.min_btn.clicked.connect(self.minimize_to_dock)
            close_btn.clicked.connect(ctrl.quit)
            self.slider.valueChanged.connect(self.set_opacity)
            self.slider.sliderReleased.connect(self.save_opacity)
            ctrl.changed.connect(self.refresh)

            self._save_timer = QTimer(self, singleShot=True, interval=400)
            self._save_timer.timeout.connect(self.save_opacity)

            self.apply_style()
            self.refresh()
            screen = QApplication.primaryScreen().availableGeometry()
            self.move(screen.right() - self.width() - 24, screen.top() + 40)

        # ── styling
        def apply_style(self):
            a = int(255 * self.opacity)
            self.panel.setStyleSheet(f"""
                #panel {{ background: rgba(18,18,22,{a});
                          border: 1px solid rgba(255,255,255,40); border-radius: 14px; }}
                QLabel {{ color: rgba(255,255,255,170); font-size: 12px; }}
                QTextEdit {{ background: transparent; border: none; color: rgba(255,255,255,235);
                             selection-background-color: rgba(120,160,255,120); }}
                #group {{ background: rgba(255,255,255,26); border-radius: 7px; }}
                QPushButton {{ background: transparent; border: none; border-radius: 0;
                               padding: 5px 9px; }}
                QPushButton[pos="mid"], QPushButton[pos="last"] {{
                               border-left: 1px solid rgba(255,255,255,30); }}
                QPushButton[pos="first"] {{ border-top-left-radius: 7px; border-bottom-left-radius: 7px; }}
                QPushButton[pos="last"]  {{ border-top-right-radius: 7px; border-bottom-right-radius: 7px; }}
                QPushButton:hover {{ background: rgba(255,255,255,45); }}
                QPushButton:pressed {{ background: rgba(255,255,255,70); }}
                QPushButton:disabled {{ background: transparent; }}
                #listen[on="false"] {{ background: rgba(34,197,94,160); }}
                #listen[on="false"]:hover {{ background: rgba(34,197,94,220); }}
                #listen[on="true"]  {{ background: rgba(239,68,68,170); }}
                #listen[on="true"]:hover  {{ background: rgba(239,68,68,225); }}
                #listen:disabled {{ background: rgba(255,255,255,20); }}
                #close:hover {{ background: rgba(239,68,68,180); }}
                QSlider::groove:horizontal {{ height: 3px; background: rgba(255,255,255,45);
                                              border-radius: 1px; }}
                QSlider::sub-page:horizontal {{ background: rgba(255,255,255,140); border-radius: 1px; }}
                QSlider::handle:horizontal {{ width: 10px; height: 10px; margin: -4px 0;
                                              border-radius: 5px; background: white; }}
                QToolTip {{ color: white; background: rgb(30,30,36);
                            border: 1px solid rgba(255,255,255,50); padding: 3px 6px; }}
            """)

        def set_opacity(self, value: int):
            self.opacity = value / 100
            self.apply_style()
            self._save_timer.start()          # debounce writes while dragging

        def save_opacity(self):
            save_settings(opacity=round(self.opacity, 2))

        def refresh(self):
            c = self.ctrl
            self.status.setText(c.status)
            color = {"Listening": "#4ade80", "Transcribing…": "#facc15"}.get(c.status, "#94a3b8")
            if c.status.startswith("Error"):
                color = "#f87171"
            self.dot.setStyleSheet(f"color: {color}; font-size: 14px;")

            on = c.listening
            self.listen_btn.setEnabled(c.ready)
            self.listen_btn.setIcon(icon("stop" if on else "play"))
            self.listen_btn.setToolTip(("Stop" if on else "Start") + " listening  (⌃ ⌃ ⌃)")
            if self.listen_btn.property("on") != ("true" if on else "false"):
                self.listen_btn.setProperty("on", "true" if on else "false")
                self.listen_btn.style().unpolish(self.listen_btn)
                self.listen_btn.style().polish(self.listen_btn)

            self.copy_btn.setIcon(icon("check", "#4ade80") if c.copied else icon("copy"))

            if self.box.toPlainText() != c.text:
                self.box.setPlainText(c.text)
                self.box.moveCursor(QTextCursor.End)

        def minimize_to_dock(self):
            """Send the window into the Dock. Borderless NSWindows can't
            miniaturize, so add the Miniaturizable style bit first."""
            try:
                import objc
                from AppKit import NSWindowStyleMaskMiniaturizable
                win = objc.objc_object(c_void_p=int(self.winId())).window()
                win.setStyleMask_(win.styleMask() | NSWindowStyleMaskMiniaturizable)
                win.miniaturize_(None)
            except Exception as e:
                print(f"(note: native minimize failed, using Qt: {e})", file=sys.stderr)
                self.showMinimized()

        # drag anywhere on the window
        def mousePressEvent(self, e):
            if e.button() == Qt.LeftButton:
                self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

        def mouseMoveEvent(self, e):
            if self._drag is not None and e.buttons() & Qt.LeftButton:
                self.move(e.globalPosition().toPoint() - self._drag)

        def mouseReleaseEvent(self, e):
            self._drag = None

    def float_on_all_spaces(widget):
        """macOS: keep the popup visible over full-screen apps and on every Space."""
        try:
            import objc
            from AppKit import (NSStatusWindowLevel,
                                NSWindowCollectionBehaviorCanJoinAllSpaces,
                                NSWindowCollectionBehaviorFullScreenAuxiliary)
            win = objc.objc_object(c_void_p=int(widget.winId())).window()
            win.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                                       | NSWindowCollectionBehaviorFullScreenAuxiliary)
            win.setLevel_(NSStatusWindowLevel)
            win.setHidesOnDeactivate_(False)
        except Exception as e:
            print(f"(note: couldn't pin to all Spaces: {e})", file=sys.stderr)

    def hide_dock_icon():
        """Terminal-only mode: no Dock icon / Cmd-Tab entry for the Python process."""
        try:
            from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
            NSApplication.sharedApplication().setActivationPolicy_(
                NSApplicationActivationPolicyAccessory)
        except Exception:
            pass

    signal.signal(signal.SIGINT, signal.SIG_DFL)   # Ctrl+C quits (when no terminal UI)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    ctrl = Controller()
    popup = None
    if args.window:
        popup = Popup(ctrl)
        popup.show()
        float_on_all_spaces(popup)

        def restore_if_minimized(state):
            # Clicking the app's Dock icon activates it → bring the popup back.
            if state == Qt.ApplicationActive and popup.isMinimized():
                popup.showNormal()
                popup.raise_()
        app.applicationStateChanged.connect(restore_if_minimized)
    else:
        hide_dock_icon()

    hotkeys = None
    if not args.no_hotkeys:
        if not ensure_accessibility():
            print("Global hotkeys need Accessibility permission: System Settings → "
                  "Privacy & Security → Accessibility → enable your terminal app, "
                  "then restart this script.", file=sys.stderr)
        hotkeys = TapHotkeys(on_toggle=ctrl.toggle, on_copy=ctrl.copy, on_clear=ctrl.clear)
        hotkeys.start()

    ctrl.engine.start()

    if not args.no_tui and TerminalUI.available():
        def state():
            return {
                "status": ctrl.status,
                "text": ctrl.text,
                "listening": ctrl.listening,
                "ready": ctrl.ready,
                "copied": ctrl.copied,
                "window": popup is not None,
                "minimized": popup is not None and popup.isMinimized(),
            }

        def do(action):
            if action == "toggle":
                ctrl.toggle()
            elif action == "copy":
                ctrl.copy()
            elif action == "clear":
                ctrl.clear()
            elif action == "minimize" and popup is not None:
                if popup.isMinimized():
                    popup.showNormal()
                    popup.raise_()
                else:
                    popup.minimize_to_dock()
            elif action == "quit":
                ctrl.quit()

        tui = TerminalUI(state, do, ctrl.engine.transcript_path)
        tui.start()
        tui_timer = QTimer(interval=80)
        tui_timer.timeout.connect(tui.tick)
        tui_timer.start()
        app.aboutToQuit.connect(tui.stop)
    else:
        ctrl._text.connect(lambda t: print(t, end=" ", flush=True))
        print(f"Transcript is also saved to: {ctrl.engine.transcript_path}")
        if not popup:
            print("Hotkeys: ⌃⌃⌃ start/stop · ⌃⌃⇧ copy · ⇧⇧⌃ clear · Ctrl+C quit")

    sys.exit(app.exec())


# ─────────────────────────────── CLI ─────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Parakeet MLX model (default {DEFAULT_MODEL}; "
                        "more accurate: mlx-community/parakeet-tdt-0.6b-v2)")
    p.add_argument("--autostart", action="store_true", help="start listening as soon as models load")
    p.add_argument("--silence-ms", type=int, default=600,
                   help="pause length that ends a sentence (default 600 ms)")
    p.add_argument("--max-segment", type=float, default=20.0,
                   help="force-cut sentences longer than this many seconds")
    p.add_argument("--opacity", type=float, default=None,
                   help="panel opacity 0–1 (default: last slider value, else 0.72)")
    p.add_argument("--window", action="store_true",
                   help="also show the floating always-on-top popup window")
    p.add_argument("--no-tui", action="store_true",
                   help="no terminal controls; just print transcribed text to stdout")
    p.add_argument("--no-hotkeys", action="store_true",
                   help="disable the global Ctrl-tap hotkeys")
    p.add_argument("--font-size", type=int, default=15)
    run_app(p.parse_args())


if __name__ == "__main__":
    main()
