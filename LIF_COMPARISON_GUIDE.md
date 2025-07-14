# Brain LIF Implementation Comparison Guide

This guide explains how to switch between and compare different LIF (Leaky Integrate-and-Fire) neuron implementations in your project.

## Available Implementations

1. **Original Brain LIF** (`original`): 
   - Uses `LIFNode` from the braincog library
   - Traditional deterministic LIF neuron implementation

2. **Stochastic LIF** (`stochastic`): 
   - Uses your developed `ProbabilisticLIFActivation`
   - Probabilistic spiking based on membrane potential and tanh activation

## Quick Start

### Method 1: Using the Configuration Script

```bash
# Switch to stochastic LIF (your implementation)
python configure_neuron_type.py --type stochastic

# Switch to original brain LIF
python configure_neuron_type.py --type original

# Check current configuration
python configure_neuron_type.py --info
```

### Method 2: Programmatic Configuration

```python
from src.modules.sblock import set_global_neuron_type, get_global_neuron_type

# Set neuron type globally
set_global_neuron_type('stochastic')  # Use your stochastic LIF
set_global_neuron_type('original')    # Use original brain LIF

# Check current setting
current_type = get_global_neuron_type()
print(f"Current neuron type: {current_type}")
```

### Method 3: Per-Model Configuration

```python
from src.modules.sblock import NeuronConfig, create_lif_neuron

# Create specific neuron types
stochastic_config = NeuronConfig("stochastic")
original_config = NeuronConfig("original")

# Create neurons with specific configs
stoch_neuron = create_lif_neuron(stochastic_config)
orig_neuron = create_lif_neuron(original_config)
```

## Using in Your Models

Once you've set the neuron type, all models will automatically use the selected implementation:

```python
from src.modules.sblock import ResnetBlockSnn, AttnBlockSnn, set_global_neuron_type

# Choose your LIF implementation
set_global_neuron_type('stochastic')  # or 'original'

# Create models as usual - they'll use the selected neuron type
resnet_block = ResnetBlockSnn(
    in_channels=64,
    out_channels=128,
    dropout=0.1,
    temb_channels=512
)

attn_block = AttnBlockSnn(in_channels=64)

# Models will now use the selected neuron type internally
```

## Running Comparisons

### Complete Comparison Script

Run the comprehensive comparison script to see side-by-side results:

```bash
python compare_lif_implementations.py
```

This script will:
- Test individual neurons with both implementations
- Test network blocks (ResNet and Attention) with both types
- Compare outputs and show statistical differences
- Provide usage instructions

### Custom Comparisons

```python
import torch
from src.modules.sblock import set_global_neuron_type, ResnetBlockSnn

# Test input
x = torch.randn(2, 64, 16, 16)
temb = torch.randn(2, 512)

results = {}

for neuron_type in ['stochastic', 'original']:
    set_global_neuron_type(neuron_type)
    
    # Create model with current neuron type
    model = ResnetBlockSnn(in_channels=64, out_channels=128, dropout=0.1)
    
    # Forward pass
    output = model(x, temb)
    results[neuron_type] = output.detach()
    
    print(f"{neuron_type} output mean: {output.mean().item():.4f}")

# Compare results
diff = torch.abs(results['stochastic'] - results['original'])
print(f"Mean absolute difference: {diff.mean().item():.6f}")
```

## Implementation Details

### Neuron Factory System

The system uses a factory pattern with these key components:

- **`NeuronConfig`**: Configuration class for specifying neuron type
- **`create_lif_neuron()`**: Factory function for creating neurons
- **`set_global_neuron_type()`**: Sets global default neuron type
- **`LIFNodeWrapper`**: Wrapper for braincog's LIFNode to ensure compatibility

### Modified Files

The following files were updated to support the selection system:

- `src/modules/sblock/neuron_factory.py` - New factory system
- `src/modules/sblock/base_block.py` - Updated to use factory
- `src/modules/sblock/__init__.py` - Added factory exports

### Compatibility

- **Stochastic LIF**: Always available (uses your implementation)
- **Original LIF**: Requires braincog library to be installed

If braincog is not available and you try to use 'original', you'll get a helpful error message.

## Parameter Customization

### Stochastic LIF Parameters

```python
from src.modules.sblock import NeuronConfig, create_lif_neuron

config = NeuronConfig("stochastic")
neuron = create_lif_neuron(config, 
    dt=1e-3,           # Simulation timestep
    C=1.0,             # Membrane capacitance  
    g_L=0.1,           # Leak conductance
    E_L=-70.0,         # Leak reversal potential
    V_reset=-80.0,     # Reset voltage after spike
    V_rest=-70.0,      # Initial/resting voltage
    beta=0.1,          # Steepness parameter for tanh
    V_offset=0.0,      # Voltage offset for tanh
    tau_ref=2.0        # Refractory period
)
```

### Original LIF Parameters

```python
config = NeuronConfig("original")
neuron = create_lif_neuron(config,
    threshold=1.0,          # Spike threshold
    v_reset=0.0,           # Reset potential
    tau=2.0,               # Membrane time constant
    surrogate_function=None # Surrogate gradient function
)
```

## Tips for Comparison

1. **Use the same random seed** for reproducible comparisons
2. **Test with different input magnitudes** to see scaling behavior
3. **Monitor spike rates** to understand activity differences
4. **Compare training dynamics** if using in learning scenarios
5. **Measure computational performance** for efficiency analysis

## Troubleshooting

### ImportError: braincog not available
- Install braincog: `pip install braincog`
- Or use only stochastic LIF: `set_global_neuron_type('stochastic')`

### Models not using new neuron type
- Make sure to set neuron type **before** creating models
- Global setting only affects newly created models

### Inconsistent results
- Check that both models use the same random seed
- Verify input preprocessing is identical
- Consider the stochastic nature of the probabilistic LIF

## Example Workflow

Here's a typical workflow for comparing the implementations:

```python
import torch
torch.manual_seed(42)  # For reproducibility

from src.modules.sblock import set_global_neuron_type

# Test your stochastic implementation
set_global_neuron_type('stochastic')
# ... create models and run experiments ...
stochastic_results = run_experiment()

# Test original implementation  
set_global_neuron_type('original')
# ... create models and run experiments ...
original_results = run_experiment()

# Compare results
analyze_differences(stochastic_results, original_results)
```

This system makes it easy to switch between implementations and compare their behavior in your specific use case!