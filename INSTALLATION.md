# Installation Guide

This guide provides multiple methods to install the required dependencies for the Time Cell Inspired Temporal Codebook in Spiking Neural Networks project.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended for faster training)
- Git

## Installation Methods

### Method 1: Automated Installation Script (Recommended)

The easiest way to get started is using the provided installation script:

```bash
# Clone the repository
git clone https://github.com/Brain-Cog-Lab/tts_vqvae.git
cd tts_vqvae

# Make the script executable and run it
chmod +x install.sh
./install.sh
```

This script will:
- Check your Python version
- Install PyTorch with appropriate CUDA support
- Install all required dependencies
- Test the probabilistic neuron integration

### Method 2: Manual Installation with pip

If you prefer manual installation or need more control:

```bash
# Clone the repository
git clone https://github.com/Brain-Cog-Lab/tts_vqvae.git
cd tts_vqvae

# Update pip
pip install --upgrade pip

# Install PyTorch (choose one based on your system)
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
pip install -r requirements.txt
```

### Method 3: Conda Environment

For users who prefer conda environments:

```bash
# Clone the repository
git clone https://github.com/Brain-Cog-Lab/tts_vqvae.git
cd tts_vqvae

# Create conda environment from file
conda env create -f environment.yml

# Activate the environment
conda activate snn-temporal-codebook
```

### Method 4: Docker (Advanced)

For containerized deployment:

```bash
# Build the Docker image
docker build -t snn-temporal-codebook .

# Run the container
docker run -it --gpus all snn-temporal-codebook
```

## Verification

After installation, verify that everything is working correctly:

```bash
# Test the installation
python test_probabilistic_neuron_integration.py

# Check PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check probabilistic neuron
python -c "from src.modules.sblock.probabilistic_neuron import ProbabilisticLIFActivation; print('✅ ProbabilisticLIFActivation imported successfully')"
```

## Key Dependencies

The main dependencies include:

- **PyTorch** (>=1.12.0): Deep learning framework
- **PyTorch Lightning** (>=1.7.0): High-level training framework
- **NumPy** (>=1.21.0): Numerical computing
- **Matplotlib** (>=3.5.0): Plotting and visualization
- **OmegaConf** (>=2.2.0): Configuration management
- **Einops** (>=0.6.0): Tensor operations
- **BrainCog** (>=0.2.0): Spiking neural network framework
- **Pillow** (>=9.0.0): Image processing
- **ImageIO** (>=2.19.0): Image I/O operations

## Hardware Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB free space

### Recommended Requirements
- CPU: 8+ cores
- RAM: 32GB+
- GPU: NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4090, A100, etc.)
- Storage: 100GB+ SSD

## Common Issues and Solutions

### Issue 1: CUDA Installation Problems
```bash
# Check CUDA version
nvidia-smi

# Install appropriate PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu[YOUR_CUDA_VERSION]
```

### Issue 2: BrainCog Installation Issues
```bash
# Install from source if package fails
pip install git+https://github.com/BrainCog-X/Brain-Cog.git
```

### Issue 3: Memory Issues
- Reduce batch size in configuration files
- Use gradient accumulation instead of large batches
- Consider using model parallelism for large models

### Issue 4: Import Errors
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/tts_vqvae"
```

## Development Setup

For development and contributing:

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/
flake8 src/
```

## Performance Optimization

### For CPU-only Systems
- Use smaller batch sizes
- Enable CPU optimizations: `torch.set_num_threads(4)`
- Consider using mixed precision training

### For GPU Systems
- Use appropriate batch sizes based on GPU memory
- Enable mixed precision training: `trainer.precision=16`
- Use multiple GPUs if available

## Next Steps

After successful installation:

1. **Prepare your dataset** (see README.md for data preparation)
2. **Configure training parameters** in the config files
3. **Run training scripts** in the `train/` directory
4. **Generate samples** using scripts in the `sample/` directory

## Getting Help

If you encounter issues:

1. Check this installation guide
2. Review the main README.md
3. Check the PROBABILISTIC_NEURON_INTEGRATION.md for neuron-specific issues
4. Open an issue on the GitHub repository

## Updates

To update the installation:

```bash
# Update the repository
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Re-run tests
python test_probabilistic_neuron_integration.py
```

---

Happy training with probabilistic spiking neurons! 🚀