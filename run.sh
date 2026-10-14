#!/usr/bin/env bash
# One-command launcher (macOS / Linux): creates .venv on first run, installs deps, starts the app.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python -m streamlit run app.py
