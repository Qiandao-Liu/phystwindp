#!/bin/bash
set -e

echo "🧹 Uninstalling existing Warp from pip..."
pip uninstall warp -y || true

echo "📥 Cloning latest Warp from GitHub..."
cd ~/workspace
rm -rf warp
git clone https://github.com/NVIDIA/warp.git
cd warp

echo "🔧 Installing Warp with autodiff support..."
pip install . --upgrade --force-reinstall --config-settings=--build-extension

echo "✅ Done. Verifying ScopedTape..."
python -c "import warp as wp; assert hasattr(wp, 'ScopedTape'); print('✅ Warp autodiff is working!')"
