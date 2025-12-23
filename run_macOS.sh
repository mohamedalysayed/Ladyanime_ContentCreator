#!/usr/bin/env bash

set -e
cd "$(dirname "$0")"

echo "🍎 macOS detected"

PYTHON_BIN="/usr/local/bin/python3.11"
VENV_DIR=".venv"
DEPS_MARKER="$VENV_DIR/.deps_installed"

# ---- Python check ----
if [ ! -x "$PYTHON_BIN" ]; then
  echo "❌ Python 3.11 not found."
  echo "Install it with:"
  echo "   brew install python@3.11"
  exit 1
fi

# ---- Homebrew check ----
command -v brew >/dev/null 2>&1 || {
  echo "❌ Homebrew not found. Install from https://brew.sh"
  exit 1
}

# ---- FFmpeg check ----
command -v ffmpeg >/dev/null 2>&1 || {
  echo "❌ FFmpeg not found. Run:"
  echo "   brew install ffmpeg pkg-config"
  exit 1
}

# ---- Virtual environment ----
if [ ! -d "$VENV_DIR" ]; then
  echo "📦 Creating virtual environment (Python 3.11)..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "🐍 Python in venv: $(python --version)"

PY_VER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

if [ "$PY_VER" != "3.11" ]; then
  echo "❌ Wrong Python version in venv: $PY_VER"
  echo "Delete .venv and rerun the script."
  exit 1
fi

# ---- Ensure pkg-config can see FFmpeg ----
export PKG_CONFIG_PATH="$(brew --prefix)/lib/pkgconfig:$PKG_CONFIG_PATH"

# ---- Install dependencies only once ----
if [ ! -f "$DEPS_MARKER" ]; then
  echo "📥 Installing Python dependencies (one-time)..."

  python -m pip install --upgrade pip setuptools wheel

  echo "🔧 Installing core ML runtime deps..."
  pip install \
    "onnxruntime>=1.14,<2" \
    "av>=11" \
    "ctranslate2>=4,<5" \
    "tokenizers>=0.13,<1"

  echo "🔧 Installing faster-whisper..."
  pip install faster-whisper --no-deps

  echo "🔧 Installing remaining app deps..."
  pip install -r requirements.txt --no-deps
  pip install "numpy<2" gradio

  touch "$DEPS_MARKER"
  echo "✅ Dependencies installed"
else
  echo "⚡ Dependencies already installed — skipping"
fi


# ---- macOS fork safety ----
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# ---- Launch app ----
echo "🚀 Launching Lady Anime..."
python -m app.gui