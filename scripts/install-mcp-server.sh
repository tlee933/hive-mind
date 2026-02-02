#!/bin/bash
# Install and configure Hive-Mind MCP server

set -euo pipefail

PROJECT_DIR="/mnt/build/MCP/hive-mind"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.14}"

echo "🐝 Hive-Mind: MCP Server Installation"
echo "======================================"
echo ""

# Check if we're in the right directory
cd "$PROJECT_DIR"

# Create virtual environment
echo "📦 Creating Python virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "   Virtual environment already exists, skipping..."
else
    $PYTHON_BIN -m venv "$VENV_DIR"
    echo "   ✅ Virtual environment created"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create config if it doesn't exist
echo ""
echo "⚙️  Setting up configuration..."
if [ ! -f "$PROJECT_DIR/config.yaml" ]; then
    cp "$PROJECT_DIR/config.example.yaml" "$PROJECT_DIR/config.yaml"
    echo "   ⚠️  Created config.yaml from example"
    echo "   ⚠️  IMPORTANT: Edit config.yaml and add your Redis password!"
else
    echo "   ✅ config.yaml already exists"
fi

# Test MCP server
echo ""
echo "🧪 Testing MCP server..."
if python "$PROJECT_DIR/mcp-server/server.py" --help >/dev/null 2>&1; then
    echo "   ✅ MCP server imports successfully"
else
    echo "   ❌ MCP server failed to import"
    exit 1
fi

# Create systemd service (optional)
echo ""
read -p "📋 Create systemd service? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    SERVICE_FILE="/etc/systemd/system/hive-mind-mcp.service"

    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Hive-Mind MCP Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="CONFIG_PATH=$PROJECT_DIR/config.yaml"
ExecStart=$VENV_DIR/bin/python $PROJECT_DIR/mcp-server/server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    echo "   ✅ Systemd service created: $SERVICE_FILE"
    echo "   To enable: sudo systemctl enable hive-mind-mcp"
    echo "   To start: sudo systemctl start hive-mind-mcp"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "📝 Next steps:"
echo "  1. Edit $PROJECT_DIR/config.yaml"
echo "  2. Add your Redis password (from NAS setup)"
echo "  3. Test: $VENV_DIR/bin/python $PROJECT_DIR/mcp-server/server.py --debug"
echo "  4. Configure Claude Code MCP settings"
echo ""
