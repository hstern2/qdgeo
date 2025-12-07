#!/bin/bash
# Setup git repository for qdgeo

set -e

echo "=========================================="
echo "Setting up git repository for qdgeo"
echo "=========================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "✗ git is not installed"
    exit 1
fi

# Check if already a git repo
if [ -d .git ]; then
    echo "✓ Git repository already initialized"
    echo ""
    echo "Current git status:"
    git status --short
else
    echo "Initializing git repository..."
    git init
    echo "✓ Git repository initialized"
fi

# Add remote if it doesn't exist
if ! git remote get-url origin &> /dev/null; then
    echo ""
    echo "Adding remote origin..."
    git remote add origin https://github.com/hstern2/qdgeo.git
    echo "✓ Remote 'origin' added: https://github.com/hstern2/qdgeo.git"
else
    echo ""
    echo "Remote 'origin' already exists:"
    git remote get-url origin
fi

# Show what will be committed
echo ""
echo "=========================================="
echo "Files to be added:"
echo "=========================================="
git status --short

echo ""
echo "=========================================="
echo "Next steps:"
echo "=========================================="
echo "1. Review the files above"
echo "2. Add files: git add ."
echo "3. Commit: git commit -m 'Initial commit'"
echo "4. Push: git push -u origin main"
echo ""
echo "Or run: git add . && git commit -m 'Initial commit' && git push -u origin main"

