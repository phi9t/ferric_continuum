#!/bin/bash
# setup-colima.sh - Set up Colima for Docker on Apple Silicon with x86_64 emulation

set -eu -o pipefail

echo "Setting up Colima for Docker on Apple Silicon..."
echo ""

# Check if running on Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "❌ This script is for Apple Silicon Macs only"
    echo "Your architecture: $(uname -m)"
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew is required but not installed"
    echo "Install from: https://brew.sh"
    exit 1
fi

echo "📦 Installing Colima, Docker CLI, QEMU, and Lima guest agents..."
brew install colima docker qemu lima-additional-guestagents

echo ""
echo "✅ Installation complete!"
echo ""

# Check if Colima instance exists (even if stopped)
if colima list 2>/dev/null | grep -q "colima"; then
    echo "⚠️  Existing Colima instance found"
    echo "Stopping and removing it to reconfigure..."
    colima stop 2>/dev/null || true
    colima delete -f
    echo ""
fi

echo "Starting Colima with x86_64 emulation (matches Ubuntu CI)..."
echo ""

# Start Colima with x86_64 emulation using QEMU
colima start \
    --cpu 2 \
    --memory 8 \
    --disk 20 \
    --arch x86_64 \
    --vm-type=qemu \
    --mount-type=virtiofs

echo "✅ Colima started successfully!"
