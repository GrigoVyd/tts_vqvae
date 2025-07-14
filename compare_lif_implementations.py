#!/usr/bin/env python3
"""
Comparison script for brain LIF implementations.

This script demonstrates how to use and compare:
- Original brain LIF (LIFNode from braincog) 
- Developed stochastic LIF (ProbabilisticLIFActivation)

Run this script to see side-by-side comparisons of both implementations.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.modules.sblock import (
    NeuronConfig, 
    create_lif_neuron, 
    set_global_neuron_type,
    get_global_neuron_type,
    ResnetBlockSnn,
    AttnBlockSnn
)


def test_individual_neurons():
    """Test individual neuron responses."""
    print("=" * 60)
    print("INDIVIDUAL NEURON COMPARISON")
    print("=" * 60)
    
    # Create test input
    batch_size = 1
    channels = 1
    height = 4
    width = 4
    x = torch.randn(batch_size, channels, height, width)
    
    print(f"Input tensor shape: {x.shape}")
    print(f"Input tensor mean: {x.mean().item():.4f}")
    print(f"Input tensor std: {x.std().item():.4f}")
    print()
    
    # Test both neuron types
    results = {}
    
    for neuron_type in ["stochastic", "original"]:
        print(f"Testing {neuron_type} LIF neuron:")
        try:
            config = NeuronConfig(neuron_type)
            neuron = create_lif_neuron(config)
            
            # Forward pass
            output = neuron(x)
            
            print(f"  Output shape: {output.shape}")
            print(f"  Output mean: {output.mean().item():.4f}")
            print(f"  Output std: {output.std().item():.4f}")
            print(f"  Output min: {output.min().item():.4f}")
            print(f"  Output max: {output.max().item():.4f}")
            print(f"  Spikes (>0.5): {(output > 0.5).sum().item()}")
            
            results[neuron_type] = output.detach().clone()
            
        except ImportError as e:
            print(f"  ✗ Failed: {e}")
            results[neuron_type] = None
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[neuron_type] = None
        
        print()
    
    return results


def test_network_blocks():
    """Test network blocks with different neuron types."""
    print("=" * 60)
    print("NETWORK BLOCK COMPARISON")
    print("=" * 60)
    
    # Create test inputs
    batch_size = 2
    in_channels = 64
    out_channels = 128
    height = 16
    width = 16
    
    x = torch.randn(batch_size, in_channels, height, width)
    temb = torch.randn(batch_size, 512)  # time embedding
    
    print(f"Input shape: {x.shape}")
    print(f"Time embedding shape: {temb.shape}")
    print()
    
    results = {}
    
    for neuron_type in ["stochastic", "original"]:
        print(f"Testing ResnetBlockSnn with {neuron_type} neurons:")
        try:
            # Set global neuron type
            set_global_neuron_type(neuron_type)
            print(f"  Current global neuron type: {get_global_neuron_type()}")
            
            # Create ResNet block
            block = ResnetBlockSnn(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=0.1,
                temb_channels=512
            )
            
            # Forward pass
            output = block(x, temb)
            
            print(f"  Output shape: {output.shape}")
            print(f"  Output mean: {output.mean().item():.4f}")
            print(f"  Output std: {output.std().item():.4f}")
            print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            results[f"resnet_{neuron_type}"] = output.detach().clone()
            
        except ImportError as e:
            print(f"  ✗ Failed: {e}")
            results[f"resnet_{neuron_type}"] = None
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[f"resnet_{neuron_type}"] = None
        
        print()
    
    # Test attention blocks
    attn_in_channels = 64
    x_attn = torch.randn(batch_size, attn_in_channels, 8, 8)
    
    for neuron_type in ["stochastic", "original"]:
        print(f"Testing AttnBlockSnn with {neuron_type} neurons:")
        try:
            # Set global neuron type
            set_global_neuron_type(neuron_type)
            
            # Create attention block
            attn_block = AttnBlockSnn(attn_in_channels)
            
            # Forward pass
            output = attn_block(x_attn)
            
            print(f"  Output shape: {output.shape}")
            print(f"  Output mean: {output.mean().item():.4f}")
            print(f"  Output std: {output.std().item():.4f}")
            print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            results[f"attn_{neuron_type}"] = output.detach().clone()
            
        except ImportError as e:
            print(f"  ✗ Failed: {e}")
            results[f"attn_{neuron_type}"] = None
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[f"attn_{neuron_type}"] = None
        
        print()
    
    return results


def compare_outputs(results):
    """Compare outputs between different implementations."""
    print("=" * 60)
    print("OUTPUT COMPARISON")
    print("=" * 60)
    
    # Compare individual neurons
    if "stochastic" in results and "original" in results:
        stoch_out = results["stochastic"]
        orig_out = results["original"]
        
        if stoch_out is not None and orig_out is not None:
            print("Individual neuron comparison:")
            diff = torch.abs(stoch_out - orig_out)
            print(f"  Mean absolute difference: {diff.mean().item():.6f}")
            print(f"  Max absolute difference: {diff.max().item():.6f}")
            print(f"  Correlation coefficient: {torch.corrcoef(torch.stack([stoch_out.flatten(), orig_out.flatten()]))[0,1].item():.6f}")
            print()
    
    # Compare network blocks if available
    for block_type in ["resnet", "attn"]:
        stoch_key = f"{block_type}_stochastic"
        orig_key = f"{block_type}_original"
        
        if stoch_key in results and orig_key in results:
            stoch_out = results[stoch_key]
            orig_out = results[orig_key]
            
            if stoch_out is not None and orig_out is not None:
                print(f"{block_type.capitalize()} block comparison:")
                diff = torch.abs(stoch_out - orig_out)
                print(f"  Mean absolute difference: {diff.mean().item():.6f}")
                print(f"  Max absolute difference: {diff.max().item():.6f}")
                
                # Calculate correlation for flattened tensors
                stoch_flat = stoch_out.flatten()
                orig_flat = orig_out.flatten()
                if len(stoch_flat) > 1:
                    corr = torch.corrcoef(torch.stack([stoch_flat, orig_flat]))[0,1]
                    print(f"  Correlation coefficient: {corr.item():.6f}")
                print()


def print_usage_instructions():
    """Print instructions for using the system."""
    print("=" * 60)
    print("USAGE INSTRUCTIONS")
    print("=" * 60)
    
    print("To switch between LIF implementations in your code:")
    print()
    print("1. Import the neuron factory:")
    print("   from src.modules.sblock import set_global_neuron_type")
    print()
    print("2. Set the neuron type before creating models:")
    print("   set_global_neuron_type('original')    # Use original brain LIF")
    print("   set_global_neuron_type('stochastic')  # Use your stochastic LIF")
    print()
    print("3. Or use the configuration script:")
    print("   python configure_neuron_type.py --type original")
    print("   python configure_neuron_type.py --type stochastic")
    print()
    print("4. Create models as usual - they will use the selected neuron type:")
    print("   from src.modules.sblock import ResnetBlockSnn")
    print("   block = ResnetBlockSnn(in_channels=64, out_channels=128, dropout=0.1)")
    print()
    print("5. The neuron type affects all models created after setting it.")


def main():
    """Main comparison function."""
    print("Brain LIF Implementation Comparison")
    print("=" * 60)
    print()
    
    # Test individual neurons
    neuron_results = test_individual_neurons()
    
    # Test network blocks
    network_results = test_network_blocks()
    
    # Combine results
    all_results = {**neuron_results, **network_results}
    
    # Compare outputs
    compare_outputs(all_results)
    
    # Print usage instructions
    print_usage_instructions()
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()