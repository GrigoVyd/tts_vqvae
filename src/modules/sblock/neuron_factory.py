"""
Neuron Factory for switching between different LIF implementations.

This module provides a unified interface to select between:
- Original brain LIF (LIFNode from braincog)
- Developed stochastic LIF (ProbabilisticLIFActivation)
"""

import torch
import torch.nn as nn
from typing import Literal, Optional, Dict, Any
from .probabilistic_neuron import ProbabilisticLIFActivation

try:
    from braincog.base.node import LIFNode
    BRAINCOG_AVAILABLE = True
except ImportError:
    BRAINCOG_AVAILABLE = False
    LIFNode = None


class NeuronConfig:
    """Configuration class for neuron selection."""
    
    def __init__(self, neuron_type: Literal["original", "stochastic"] = "stochastic"):
        """
        Initialize neuron configuration.
        
        Args:
            neuron_type: Type of neuron to use
                - "original": Use original brain LIF (LIFNode from braincog)
                - "stochastic": Use developed stochastic LIF (ProbabilisticLIFActivation)
        """
        self.neuron_type = neuron_type
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration."""
        if self.neuron_type == "original" and not BRAINCOG_AVAILABLE:
            raise ImportError(
                "braincog is not available. Cannot use 'original' neuron type. "
                "Please install braincog or use 'stochastic' neuron type."
            )
        
        if self.neuron_type not in ["original", "stochastic"]:
            raise ValueError(
                f"Invalid neuron_type: {self.neuron_type}. "
                "Must be 'original' or 'stochastic'."
            )


class LIFNodeWrapper(nn.Module):
    """
    Wrapper for the original LIFNode to provide consistent interface.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        if not BRAINCOG_AVAILABLE:
            raise ImportError("braincog is not available.")
        
        # Default parameters for LIFNode if not provided
        default_params = {
            'threshold': 1.0,
            'v_reset': 0.0,
            'tau': 2.0,
            'surrogate_function': None
        }
        
        # Merge with provided kwargs
        params = {**default_params, **kwargs}
        self.lif_node = LIFNode(**params)
    
    def forward(self, x):
        """Forward pass through LIFNode."""
        return self.lif_node(x)
    
    def reset(self):
        """Reset the neuron state."""
        if hasattr(self.lif_node, 'reset'):
            self.lif_node.reset()


def create_lif_neuron(config: Optional[NeuronConfig] = None, **kwargs) -> nn.Module:
    """
    Factory function to create LIF neurons based on configuration.
    
    Args:
        config: NeuronConfig instance. If None, defaults to stochastic neuron.
        **kwargs: Additional parameters passed to the neuron constructor.
    
    Returns:
        nn.Module: The created neuron module.
    
    Examples:
        # Use stochastic LIF (default)
        neuron = create_lif_neuron()
        
        # Use original brain LIF
        config = NeuronConfig("original")
        neuron = create_lif_neuron(config)
        
        # Use stochastic LIF with custom parameters
        config = NeuronConfig("stochastic")
        neuron = create_lif_neuron(config, beta=0.2, V_offset=5.0)
    """
    if config is None:
        config = NeuronConfig("stochastic")
    
    if config.neuron_type == "original":
        return LIFNodeWrapper(**kwargs)
    elif config.neuron_type == "stochastic":
        return ProbabilisticLIFActivation(**kwargs)
    else:
        raise ValueError(f"Unknown neuron type: {config.neuron_type}")


# Global configuration for easy switching
_global_neuron_config = NeuronConfig("stochastic")


def set_global_neuron_type(neuron_type: Literal["original", "stochastic"]):
    """
    Set the global neuron type for all subsequent neuron creations.
    
    Args:
        neuron_type: Type of neuron to use globally.
    """
    global _global_neuron_config
    _global_neuron_config = NeuronConfig(neuron_type)


def get_global_neuron_type() -> str:
    """Get the current global neuron type."""
    return _global_neuron_config.neuron_type


def create_lif_neuron_global(**kwargs) -> nn.Module:
    """
    Create a LIF neuron using the global configuration.
    
    Args:
        **kwargs: Additional parameters passed to the neuron constructor.
    
    Returns:
        nn.Module: The created neuron module.
    """
    return create_lif_neuron(_global_neuron_config, **kwargs)