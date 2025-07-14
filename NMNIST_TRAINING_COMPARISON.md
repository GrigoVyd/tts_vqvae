# NMNIST Training with LIF Comparison

This guide shows how to train NMNIST with both the original brain LIF and your developed stochastic LIF for direct comparison.

## Quick Start

### Train with Original Brain LIF

```bash
# Prerequisites: Install braincog
pip install braincog

# Train stage 1 (autoencoder)
./train_nmnist_original.sh --stage 1

# Train stage 2 (transformer) 
./train_nmnist_original.sh --stage 2
```

### Train with Stochastic LIF (Your Implementation)

```bash
# Train stage 1 (autoencoder)
./train/stage_1/run_snn_te_nmnist.sh

# Train stage 2 (transformer)
./train/stage_2/run_snn_transformer_nmnist.sh
```

## Detailed Instructions

### 1. Prerequisites

**For Original LIF:**
- Install braincog: `pip install braincog`
- NMNIST dataset in `../dataset/DVS/NMNIST/`

**For Stochastic LIF:**
- Only PyTorch required (braincog optional)
- NMNIST dataset in `../dataset/DVS/NMNIST/`

### 2. Training Process

Both implementations follow a **2-stage training process**:

1. **Stage 1**: Train SNN autoencoder (VQ-VAE style)
2. **Stage 2**: Train SNN transformer on the learned latent space

### 3. Training Commands

#### Option A: Using Shell Scripts (Recommended)

**Original LIF:**
```bash
# Stage 1
./train_nmnist_original.sh --stage 1

# Stage 2 (after stage 1 completes)
./train_nmnist_original.sh --stage 2
```

**Stochastic LIF:**
```bash
# Stage 1
./train/stage_1/run_snn_te_nmnist.sh

# Stage 2 (after stage 1 completes)  
./train/stage_2/run_snn_transformer_nmnist.sh
```

#### Option B: Using Python Scripts Directly

**Original LIF:**
```bash
python3 train_nmnist_original.py --stage 1
python3 train_nmnist_original.py --stage 2
```

**Stochastic LIF:**
```bash
python3 train/stage_1/snn_te_nmnist.py --base train/stage_1/snn_te_nmnist.yaml
python3 train/stage_2/snn_transformer_nmnist.py --base train/stage_2/snn_transformer_nmnist.yaml
```

### 4. Results and Comparison

#### Result Directories

**Original LIF Results:**
- Stage 1: `res/snn_te_dvs/nmnist_original_lif/`
- Stage 2: `res/snn_transformer/nmnist_original_lif/`

**Stochastic LIF Results:**
- Stage 1: `res/snn_te_dvs/nmnist/`
- Stage 2: `res/snn_transformer/nmnist/`

#### What to Compare

1. **Training Loss Curves**: Check TensorBoard logs in `tb/` directories
2. **Final Model Performance**: Compare validation metrics
3. **Generated Samples**: Visual comparison of reconstructed events
4. **Training Speed**: Compare training time and convergence
5. **Model Checkpoints**: Compare final model sizes and architectures

#### Quick Comparison Script

```bash
# Compare results visually
python3 compare_lif_implementations.py

# Or analyze specific metrics
tensorboard --logdir res/snn_te_dvs/nmnist/tb/ --port 6006 &
tensorboard --logdir res/snn_te_dvs/nmnist_original_lif/tb/ --port 6007 &
```

### 5. Configuration Details

The training automatically handles the differences:

**Original LIF Script Changes:**
- Sets global neuron type to `'original'` before model creation
- Uses `nmnist_original_lif` as experiment suffix
- Automatically updates checkpoint paths for stage 2

**Key Differences:**
- **Neuron Behavior**: Deterministic vs. probabilistic spiking
- **Training Dynamics**: Different gradient flows and convergence patterns
- **Memory Usage**: Potentially different memory requirements
- **Computational Cost**: Different forward/backward pass times

### 6. Troubleshooting

#### Common Issues

**"braincog not found" Error:**
```bash
pip install braincog
# or use stochastic LIF instead
```

**"Dataset not found" Error:**
- Download NMNIST dataset to `../dataset/DVS/NMNIST/`
- Check dataset path in config files

**Stage 2 "Checkpoint not found" Error:**
- Complete stage 1 training first
- Check that stage 1 results exist in expected directory

#### Performance Tips

1. **GPU Memory**: Monitor GPU memory usage, adjust batch size if needed
2. **Training Time**: Stage 1 typically takes 4-6 hours, stage 2 takes 2-4 hours
3. **Checkpointing**: Models save every 25 epochs, you can resume from any checkpoint

### 7. Experiment Workflow

Here's a complete comparison workflow:

```bash
# 1. Train both implementations
./train_nmnist_original.sh --stage 1           # Original LIF stage 1
./train/stage_1/run_snn_te_nmnist.sh           # Stochastic LIF stage 1

# 2. Compare stage 1 results
tensorboard --logdir res/snn_te_dvs/ --port 6006

# 3. Train stage 2 for both
./train_nmnist_original.sh --stage 2           # Original LIF stage 2  
./train/stage_2/run_snn_transformer_nmnist.sh  # Stochastic LIF stage 2

# 4. Final comparison
tensorboard --logdir res/snn_transformer/ --port 6007
python3 compare_lif_implementations.py
```

### 8. Expected Outcomes

You should observe:

- **Different spike patterns** between deterministic and probabilistic neurons
- **Varying convergence rates** during training
- **Different final performance metrics** 
- **Distinct generated sample characteristics**

This comparison will help you understand the impact of your stochastic LIF innovation on the NMNIST neuromorphic vision task!

## Key Files Created

- `train_nmnist_original.py` - Python training script for original LIF
- `train_nmnist_original.sh` - Shell script wrapper
- Original training scripts continue to work for stochastic LIF
- All comparison tools from the main LIF comparison system