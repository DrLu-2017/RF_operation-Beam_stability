#!/bin/bash

TARGET_FILE="$HOME/.bash_aliases"
ALIAS_CMD="alias run_albums='/home/lu/streamlit/albums-main/start_app.sh'"

# Create .bash_aliases if it doesn't exist
touch "$TARGET_FILE"

# Check if alias already exists to avoid duplicates
if grep -q "run_albums" "$TARGET_FILE"; then
    echo "Alias 'run_albums' already exists in $TARGET_FILE"
else
    echo "$ALIAS_CMD" >> "$TARGET_FILE"
    echo "✅ Alias 'run_albums' added to $TARGET_FILE"
fi

echo ""
echo "To apply changes now, run:"
echo "  source ~/.bashrc"
echo ""
echo "Then you can start the app from anywhere by typing:"
echo "  run_albums"
