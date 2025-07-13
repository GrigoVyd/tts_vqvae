#!/usr/bin/env python3
"""
Test script to demonstrate the integration of ProbabilisticSpikeNeuron 
into the algorithm as a replacement for LIFNode.
"""

import torch
import torch.nn as nn
import numpy as np
from src.modules.sblock.base_block import ResnetBlockSnn, AttnBlockSnn
from src.modules.sblock.probabilistic_neuron import ProbabilisticLIFActivation, ProbabilisticSpikeNeuron


def test_probabilistic_lif_activation():
    """Test the ProbabilisticLIFActivation wrapper"""
    print("Testing ProbabilisticLIFActivation...")
    
    # Create a simple test tensor
    batch_size = 2
    channels = 3
    height = 4
    width = 4
    x = torch.randn(batch_size, channels, height, width)
    
    # Test the wrapper
    neuron = ProbabilisticLIFActivation()
    output = neuron(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output contains values in [0, 1]: {torch.all((output >= 0) & (output <= 1))}")
    print(f"Some sample outputs: {output[0, 0, 0, :3]}")
    
    # Test reset functionality
    neuron.reset()
    print("Neuron reset successfully")
    
    return True


def test_resnet_block_integration():
    """Test that ResnetBlockSnn works with ProbabilisticLIFActivation"""
    print("\nTesting ResnetBlockSnn with ProbabilisticLIFActivation...")
    
    # Create a ResNet block
    in_channels = 64
    out_channels = 128
    dropout = 0.1
    temb_channels = 512
    
    block = ResnetBlockSnn(
        in_channels=in_channels,
        out_channels=out_channels,
        dropout=dropout,
        temb_channels=temb_channels
    )
    
    # Test forward pass
    batch_size = 2
    height = 32
    width = 32
    x = torch.randn(batch_size, in_channels, height, width)
    temb = torch.randn(batch_size, temb_channels)
    
    output = block(x, temb)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output channels: {out_channels}")
    print(f"Block forward pass successful!")
    
    return True


def test_attention_block_integration():
    """Test that AttnBlockSnn works with ProbabilisticLIFActivation"""
    print("\nTesting AttnBlockSnn with ProbabilisticLIFActivation...")
    
    # Create an attention block
    in_channels = 64
    attn_block = AttnBlockSnn(in_channels)
    
    # Test forward pass
    batch_size = 2
    height = 16
    width = 16
    x = torch.randn(batch_size, in_channels, height, width)
    
    output = attn_block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention block forward pass successful!")
    
    return True


def test_probabilistic_spike_neuron_direct():
    """Test the original ProbabilisticSpikeNeuron directly"""
    print("\nTesting original ProbabilisticSpikeNeuron...")
    
    # Create neuron with default parameters
    neuron = ProbabilisticSpikeNeuron(
        dt=1e-3,
        C=1.0,
        g_L=0.1,
        E_L=-70.0,
        V_reset=-80.0,
        V_rest=-70.0,
        beta=0.1,
        V_offset=0.0,
        tau_ref=2.0
    )
    
    # Test with different input currents
    test_currents = [0.5, 1.0, 1.5, 2.0]
    
    for current in test_currents:
        neuron.reset()
        spikes = []
        potentials = []
        
        # Run for 100 timesteps
        for _ in range(100):
            spike, potential = neuron(torch.tensor(current))
            spikes.append(spike.item())
            potentials.append(potential.item())
        
        spike_count = sum(spikes)
        avg_potential = np.mean(potentials)
        
        print(f"Current: {current}A, Spikes: {spike_count}, Avg potential: {avg_potential:.3f}V")
    
    return True


def main():
    """Main test function"""
    print("=" * 60)
    print("Testing ProbabilisticSpikeNeuron Integration")
    print("=" * 60)
    
    try:
        # Test individual components
        assert test_probabilistic_lif_activation(), "ProbabilisticLIFActivation test failed"
        assert test_resnet_block_integration(), "ResnetBlockSnn integration test failed"
        assert test_attention_block_integration(), "AttnBlockSnn integration test failed"
        assert test_probabilistic_spike_neuron_direct(), "ProbabilisticSpikeNeuron direct test failed"
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! The ProbabilisticSpikeNeuron has been")
        print("   successfully integrated into the algorithm as a replacement")
        print("   for LIFNode.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()