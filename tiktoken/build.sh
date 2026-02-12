#!/bin/bash
# tiktoken build script for Python 3.14
# Builds from upstream OpenAI tiktoken repo
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_VENV="${AI_VENV:-/var/mnt/build/.venv}"

echo "=== tiktoken Build for Python 3.14 ==="
echo "AI_VENV: $AI_VENV"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo ""

# Source Rust environment
source ~/.cargo/env 2>/dev/null || true

# Activate venv
source "$AI_VENV/bin/activate"

# Verify prerequisites
echo "=== Prerequisites ==="
echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"
echo "Python: $(python --version)"
echo ""

# Ensure setuptools-rust is installed
pip install setuptools-rust -q

# Clone tiktoken if not present
if [ ! -d "$SCRIPT_DIR/tiktoken-src" ]; then
    echo "=== Cloning tiktoken ==="
    git clone https://github.com/openai/tiktoken.git "$SCRIPT_DIR/tiktoken-src"
fi

# Build with pip wheel
cd "$SCRIPT_DIR/tiktoken-src"
echo "=== Building tiktoken with setuptools-rust ==="
pip wheel . --no-deps -w dist/

# Find wheel
WHEEL=$(ls dist/tiktoken-*-cp314-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
    echo "ERROR: No wheel found!"
    exit 1
fi

# Copy to wheels dir
cp "$WHEEL" "$SCRIPT_DIR/wheels/"

echo ""
echo "=== Installing $WHEEL ==="
pip install --force-reinstall "$WHEEL"

# Install custom encodings
echo ""
echo "=== Installing custom encodings ==="
pip install -e "$SCRIPT_DIR/my_encodings"

# Verify from clean dir
echo ""
echo "=== Verification ==="
cd /tmp
python -c "import tiktoken; print(f'tiktoken {tiktoken.__version__}')"
python -c "import tiktoken; print('Encodings:', tiktoken.list_encoding_names())"

echo ""
echo "=== Done ==="
echo "Wheel: $SCRIPT_DIR/wheels/$(basename $WHEEL)"
