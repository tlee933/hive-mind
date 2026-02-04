#!/bin/bash
# Add Hive-Mind to Claude Code MCP config

CONFIG_FILE="$HOME/.config/claude-code/mcp_config.json"

echo "🔌 Connecting Hive-Mind to Claude Code..."

# Create config dir if needed
mkdir -p "$(dirname "$CONFIG_FILE")"

# Create or update config
if [ -f "$CONFIG_FILE" ]; then
    echo "   Found existing MCP config"
    # Backup
    cp "$CONFIG_FILE" "$CONFIG_FILE.backup"
else
    echo "   Creating new MCP config"
    echo '{"mcpServers":{}}' > "$CONFIG_FILE"
fi

# Add hive-mind server
python3 << 'PYSCRIPT'
import json

config_file = "/home/hashcat/.config/claude-code/mcp_config.json"

with open(config_file, 'r') as f:
    config = json.load(f)

if 'mcpServers' not in config:
    config['mcpServers'] = {}

config['mcpServers']['hive-mind'] = {
    "command": "/mnt/build/MCP/hive-mind/.venv/bin/python",
    "args": ["/mnt/build/MCP/hive-mind/mcp-server/server.py"],
    "env": {
        "CONFIG_PATH": "/mnt/build/MCP/hive-mind/config.yaml"
    }
}

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Hive-Mind MCP server added!")
print()
print("Config:")
print(json.dumps(config['mcpServers']['hive-mind'], indent=2))
PYSCRIPT

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ MCP Integration Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next: Restart Claude Code to activate"
echo ""
