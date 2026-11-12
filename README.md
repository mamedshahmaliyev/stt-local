# Voice STT — local English speech-to-text in your terminal

Run it and the terminal becomes the app: a status line, a row of clickable buttons, and the live transcript. Press **space** (or click Start) and talk: each sentence appears about a second after you pause. Press it again and the microphone turns off completely. Text flows as one continuous block. Everything runs locally on your Mac.

To also get a floating, half-transparent popup that stays on top of other windows, add `--window`.

- **Speech model:** Parakeet TDT-CTC 110M, via MLX
- **Sentence splitting:** Silero VAD (~2 MB)
- **Saved transcripts:** every session is also written to `~/.voice_stt/transcripts/`

## Requirements: Mac with Apple Silicon (MLX)

This project is built for **macOS on Apple Silicon (M1 or newer)**. Speech recognition runs on [MLX](https://github.com/ml-explore/mlx), Apple's machine-learning framework, through `parakeet-mlx`. MLX already uses the Mac's built-in GPU (via Metal) by default, so there is nothing to switch on.

It will **not** run as-is on Intel Macs, Linux or Windows, because MLX only supports Apple Silicon. To run it elsewhere with an NVIDIA GPU, see [Running on an NVIDIA GPU (Linux/Windows)](#running-on-an-nvidia-gpu-linuxwindows).

## Install (once)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run

```bash
./run.sh            # or: .venv/bin/python voice_stt.py
```

The first run downloads the models (~150 MB). While they load, Start is greyed out; it becomes active when they're ready. The first time you start listening, macOS will ask to give **Terminal** (or iTerm) microphone access. Allow it.

### Terminal controls

Click a button with the mouse, or press its key:

| Key | Action |
|---|---|
| `space` / `s` | Start / stop listening |
| `c` | Copy the text to the clipboard |
| `x` | Clear |
| `m` | Minimize the popup to the Dock / restore (only with `--window`) |
| `q` / Ctrl+C | Quit (the full transcript is printed in the terminal) |

- **Pressing Stop mid-sentence:** the last words still get transcribed.
- **Selecting text in the terminal:** clickable buttons mean the terminal is capturing the mouse. Hold **Option** (iTerm) or **Fn** (Terminal.app) while dragging, or just press `c`.
- With `--no-tui`, the text is just printed as it arrives, with no controls.

### Popup window (`--window`)

- **Toolbar:** icon buttons (hover for labels): Start/Stop, Clear, Copy, Minimize, Quit.
- **Move it:** drag it anywhere. **Resize it:** use the bottom-right corner.
- **Transparency:** the small slider at the bottom left. It's remembered in `~/.voice_stt/settings.json`.
- **Minimize:** press – to send it into the Dock. Click its Dock thumbnail or the app's Dock icon to bring it back.

### Global hotkeys

| Keys | Action |
|---|---|
| Tap **Ctrl Ctrl Ctrl** | Start / stop listening |
| Tap **Ctrl Ctrl**, then **Shift** | Copy the text to the clipboard |
| Tap **Shift Shift**, then **Ctrl** | Clear the text |

Taps must be quick, with less than half a second between them. These work in any app, but macOS only allows them once the app running the script has **Accessibility** permission. The first launch opens the prompt: enable your terminal app (or Cursor/VS Code, if you launch from there) under System Settings → Privacy & Security → Accessibility, then restart the script. Use `--no-hotkeys` to turn them off.

### Options

| Option | What it does |
|---|---|
| `--window` | Also show the floating popup window |
| `--autostart` | Start listening as soon as the models load |
| `--model mlx-community/parakeet-tdt-0.6b-v2` | Use the bigger, more accurate model (~730 MB) |
| `--silence-ms 900` | Wait longer before ending a sentence, if you pause mid-thought |
| `--opacity 0.5` | Popup: set the starting transparency (otherwise the last slider value is used) |
| `--no-tui` | No terminal controls; just print the text |
| `--no-hotkeys` | Turn off the global tap hotkeys |
| `--font-size 18` | Popup: use bigger text |

## Running on an NVIDIA GPU (Linux/Windows)

The same Parakeet models are published by NVIDIA for its **NeMo** toolkit, which runs on CUDA GPUs. To switch from MLX to NeMo:

**1. Install PyTorch with CUDA, then NeMo**, in place of the Mac-only packages:

```bash
uv venv --python 3.12 .venv
# PyTorch build for your CUDA version (see https://pytorch.org/get-started/locally/)
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python "nemo_toolkit[asr]" silero-vad sounddevice soundfile numpy PySide6
```

Leave out `parakeet-mlx` and `pyobjc-framework-Cocoa`; they only install on macOS.

**2. Change the model loader** in `voice_stt.py` (`_load`):

```python
# before (MLX)
from parakeet_mlx import from_pretrained
self.asr = from_pretrained(self.args.model)

# after (NeMo, uses the GPU automatically when CUDA is available)
import nemo.collections.asr as nemo_asr
self.asr = nemo_asr.models.ASRModel.from_pretrained(self.args.model).eval()
```

**3. Change the transcribe call** in `_transcribe`:

```python
# before (MLX)
text = self.asr.transcribe(tmp).text.strip()

# after (NeMo returns a list; newer versions return objects with .text)
out = self.asr.transcribe([tmp])[0]
text = (out.text if hasattr(out, "text") else out).strip()
```

**4. Use the NVIDIA model names**, either by changing `DEFAULT_MODEL` or by passing `--model`:

| MLX (Mac) | NeMo (NVIDIA GPU) |
|---|---|
| `mlx-community/parakeet-tdt_ctc-110m` | `nvidia/parakeet-tdt_ctc-110m` |
| `mlx-community/parakeet-tdt-0.6b-v2` | `nvidia/parakeet-tdt-0.6b-v2` |

**5. Run with `--no-hotkeys`.** The global tap hotkeys, pinning the popup to every Space and minimizing to the Dock all use macOS APIs. Without macOS they turn themselves off and print a note, and everything else works.

To check that the GPU is being used, run `python -c "import torch; print(torch.cuda.is_available())"`. It should print `True`. If it prints `False`, NeMo falls back to the CPU, which is much slower.

## Troubleshooting

- **The popup (`--window`) hides when you switch to a full-screen app:** check that `pyobjc-framework-Cocoa` is installed. It's what pins the window to every Space.
- **The status shows an error after pressing Start:** usually the microphone permission. Check System Settings → Privacy & Security → Microphone and allow your terminal app.
