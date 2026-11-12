#!/bin/bash
# Launch the Voice STT popup using the project venv.
cd "$(dirname "$0")"
exec .venv/bin/python voice_stt.py "$@"
