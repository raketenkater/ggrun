#!/bin/bash

# Configuration
# Find the config file in a portable way
CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/opencode.json"
[[ ! -f "$CONFIG_FILE" ]] && CONFIG_FILE="$HOME/.opencode.json"

MODELS_URL="http://localhost:8081/v1/models"
SLOTS_URL="http://localhost:8081/slots"

echo "Syncing OpenCode model with llm-server..."

# 1. Fetch the current model ID from the server
MODEL_ID=$(curl -s "$MODELS_URL" | jq -r '.data[0].id')

if [ -z "$MODEL_ID" ] || [ "$MODEL_ID" == "null" ]; then
    echo "Error: Could not fetch model ID from server. Is the server running?"
    exit 1
fi

echo "Detected model: $MODEL_ID"

# 2. Fetch the actual context size from the server slots
# We use the first slot's n_ctx as the baseline
SERVER_CTX=$(curl -s "$SLOTS_URL" | jq -r '.[0].n_ctx')

if [ -z "$SERVER_CTX" ] || [ "$SERVER_CTX" == "null" ]; then
    echo "Warning: Could not fetch context size from server. Defaulting to 65536."
    SERVER_CTX=65536
fi

# 3. Calculate the safe limit (5% less than server context)
SAFE_CTX=$(( SERVER_CTX * 95 / 100 ))

echo "Server context: $SERVER_CTX | OpenCode safe limit: $SAFE_CTX"

# 4. Update opencode.json using jq
TMP_FILE=$(mktemp)
jq --arg mid "$MODEL_ID" --argjson ctx "$SAFE_CTX" \
   '.provider["llama-cpp"].models = {($mid): {"name": $mid, "limit": {"context": $ctx, "output": 64000}}} | .model = "llama-cpp/" + $mid' \
   "$CONFIG_FILE" > "$TMP_FILE" && mv "$TMP_FILE" "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo "Successfully updated $CONFIG_FILE"
    echo "  - Model: $MODEL_ID"
    echo "  - Context Limit: $SAFE_CTX"
else
    echo "Error: Failed to update configuration file."
    exit 1
fi

# 5. Sync DCP config to match server context
# Scale maxContextLimit to ~75% of server context, minContextLimit to ~10% (floor 15000)
DCP_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/dcp.jsonc"
DCP_MAX=$(( SERVER_CTX * 75 / 100 ))
DCP_MIN=$(( SERVER_CTX * 10 / 100 ))
[[ $DCP_MIN -lt 15000 ]] && DCP_MIN=15000

if [ -f "$DCP_FILE" ]; then
    TMP_DCP=$(mktemp)
    sed -e "s/\"maxContextLimit\": [0-9]*/\"maxContextLimit\": $DCP_MAX/" \
        -e "s/\"minContextLimit\": [0-9]*/\"minContextLimit\": $DCP_MIN/" \
        "$DCP_FILE" > "$TMP_DCP" && mv "$TMP_DCP" "$DCP_FILE"
    echo "Updated $DCP_FILE"
    echo "  - DCP minContextLimit: $DCP_MIN"
    echo "  - DCP maxContextLimit: $DCP_MAX"
else
    echo "Warning: $DCP_FILE not found. Skipping DCP sync."
fi

# 6. Sync agent context files (AGENTS.md, CLAUDE.md) so system prompts stay accurate
AGENT_FILES=("$HOME/AGENTS.md" "$HOME/.claude/CLAUDE.md")
DCP_TRIGGER=$DCP_MAX
DCP_TRIGGER_FMT=$(printf "%'d" "$DCP_TRIGGER" 2>/dev/null || echo "$DCP_TRIGGER")
DCP_BUFFER=$(( SERVER_CTX / 4 ))
DCP_BUFFER_K=$(( DCP_BUFFER / 1000 ))

for af in "${AGENT_FILES[@]}"; do
    if [ -f "$af" ]; then
        if grep -q "exceeds" "$af" 2>/dev/null; then
            sed -i "s/exceeds \*\*[0-9,]* tokens\*\*/exceeds **${DCP_TRIGGER_FMT} tokens**/" "$af"
            echo "Updated $af — DCP trigger: ${DCP_TRIGGER_FMT}"
        fi
        if grep -q "[0-9]*k token buffer" "$af" 2>/dev/null; then
            sed -i "s/[0-9]*k token buffer/${DCP_BUFFER_K}k token buffer/" "$af"
            echo "Updated $af — DCP buffer: ${DCP_BUFFER_K}k"
        fi
    fi
done
