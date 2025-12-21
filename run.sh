#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Create venv if missing
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

# Upgrade pip tools
python -m pip install --upgrade pip setuptools wheel

# Install your deps
pip install -r requirements.txt

# Gradio needs numpy<2 right now on many Ubuntu setups
pip install "numpy<2" gradio

# Launch UI
python -m app.gui
