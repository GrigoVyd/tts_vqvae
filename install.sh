#!/bin/bash
# Installation script for Time Cell Inspired Temporal Codebook in Spiking Neural Networks
# This script sets up the environment and installs all necessary dependencies

echo "=============================================="
echo "Setting up Spiking Neural Network Environment"
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $python_version detected"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

# Update pip
echo "🔄 Updating pip..."
pip3 install --upgrade pip

# Install PyTorch (with CUDA support if available)
echo "🔄 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected, installing PyTorch with CUDA support"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "ℹ️  No NVIDIA GPU detected, installing CPU-only PyTorch"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo "🔄 Installing other dependencies..."
pip3 install -r requirements.txt

# Verify installation
echo "🔄 Verifying installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test the probabilistic neuron integration
echo "🔄 Testing probabilistic neuron integration..."
if python3 test_probabilistic_neuron_integration.py; then
    echo "✅ Probabilistic neuron integration test passed!"
else
    echo "❌ Probabilistic neuron integration test failed!"
fi

echo "=============================================="
echo "✅ Installation completed successfully!"
echo "=============================================="
echo ""
echo "📋 Next steps:"
echo "1. Prepare your dataset (see README.md for data preparation)"
echo "2. Run training scripts in train/ directory"
echo "3. Use sample scripts in sample/ directory for generation"
echo ""
echo "🔧 For troubleshooting:"
echo "- Check PROBABILISTIC_NEURON_INTEGRATION.md for detailed documentation"
echo "- Ensure your GPU drivers are properly installed for CUDA support"
echo "- Verify dataset paths in configuration files"
echo ""
echo "🚀 Happy training with probabilistic spiking neurons!"