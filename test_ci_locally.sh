#!/bin/bash
# Test script to verify CI workflow steps locally

set -e  # Exit on error

echo "=========================================="
echo "Testing CI workflow steps locally"
echo "=========================================="

# Step 1: Check Python version
echo ""
echo "Step 1: Checking Python version..."
python3 --version

# Step 2: Check if uv is available
echo ""
echo "Step 2: Checking for uv..."
if command -v uv &> /dev/null; then
    echo "✓ uv is installed"
    uv --version
else
    echo "✗ uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Step 3: Install project with dev dependencies
echo ""
echo "Step 3: Installing project with dev dependencies..."
uv pip install -e ".[dev]"

# Step 4: Run tests
echo ""
echo "Step 4: Running tests..."
# Use --active to use the active environment, or let uv create its own
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Note: Active venv detected, using --active flag"
    uv run --active pytest tests -v
else
    uv run pytest tests -v
fi

echo ""
echo "=========================================="
echo "✓ All CI steps completed successfully!"
echo "=========================================="

