#!/bin/bash

TARGET_FILE="$HOME/.bash_aliases"
ALIAS_CMD="alias run_drfb='/home/lu/streamlit/DRFB/start_app.sh'"

# Create .bash_aliases if it doesn't exist
touch "$TARGET_FILE"

# Check if alias already exists to avoid duplicates
if grep -q "run_drfb" "$TARGET_FILE"; then
    echo "Alias 'run_drfb' already exists in $TARGET_FILE"
else
    echo "$ALIAS_CMD" >> "$TARGET_FILE"
    echo "✅ Alias 'run_drfb' added to $TARGET_FILE"
fi

echo ""
echo "To apply changes now, run:"
echo "  source ~/.bashrc"
echo ""
echo "Then you can start the app from anywhere by typing:"
echo "  run_drfb"
