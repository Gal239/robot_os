#!/bin/bash
# Quick push script - commits and pushes all changes to robot_os repo

echo "🚀 Auto-commit and push to robot_os"
echo "=================================="

# Add all changes
echo "📦 Adding all changes..."
git add -A

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "✅ No changes to commit"
else
    # Commit with timestamp
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "💾 Committing changes..."
    git commit -m "Auto-update: $TIMESTAMP

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
fi

# Verify we're on robot_os remote
REMOTE_URL=$(git remote get-url origin)
if [[ "$REMOTE_URL" != *"robot_os"* ]]; then
    echo "⚠️  WARNING: Remote is not robot_os!"
    echo "   Current: $REMOTE_URL"
    echo "   Setting to robot_os..."
    git remote set-url origin https://github.com/Gal239/robot_os.git
fi

# Push to robot_os
echo "⬆️  Pushing to robot_os..."
git push origin main

echo ""
echo "✅ Done! Check: https://github.com/Gal239/robot_os"
