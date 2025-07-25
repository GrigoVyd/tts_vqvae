import torch
import torch.nn as nn


class ProbabilisticSpikeNeuron(nn.Module):
    """
    A probabilistic spiking neuron model that generates spikes based on membrane potential
    and probabilistic activation using a tanh function.
    
    The neuron integrates input current, updates membrane potential according to 
    leak-integrate-and-fire dynamics, and generates spikes probabilistically based on
    the membrane potential.
    """
    
    def __init__(self, dt, C, g_L, E_L, V_reset, V_rest, beta, V_offset, tau_ref=0.0):
        """
        Initialize the ProbabilisticSpikeNeuron.
        
        Args:
            dt (float): Simulation timestep
            C (float): Membrane capacitance
            g_L (float): Leak conductance
            E_L (float): Leak reversal potential (resting potential)
            V_reset (float): Reset voltage after spike
            V_rest (float): Initial voltage or resting voltage for leak
            beta (float): Steepness parameter for tanh activation
            V_offset (float): Voltage offset for tanh activation
            tau_ref (float): Refractory period (default: 0.0)
        """
        super(ProbabilisticSpikeNeuron, self).__init__()
        self.dt = dt  # Simulation timestep
        self.C = C    # Membrane capacitance
        self.g_L = g_L  # Leak conductance
        self.E_L = E_L  # Leak reversal potential (resting potential)
        self.V_reset = V_reset # Reset voltage after spike
        self.V_rest = V_rest # Initial voltage or resting voltage for leak
        self.beta = beta  # Steepness parameter for tanh
        self.V_offset = V_offset # Voltage offset for tanh
        self.tau_ref = tau_ref # Refractory period
        self.ref_steps = int(tau_ref / dt) # Number of time steps for refractory period

        # Initialize membrane potential and refractory counter
        self.V = torch.tensor(V_rest)
        self.ref_counter = 0

    def forward(self, I_syn):
        """
        Forward pass of the probabilistic spiking neuron.
        
        Args:
            I_syn (torch.Tensor): Synaptic input current
            
        Returns:
            tuple: (spike, membrane_potential)
                - spike: Binary spike output (0 or 1)
                - membrane_potential: Current membrane potential
        """
        # Update membrane potential
        if self.ref_counter > 0:
            # During refractory period, V is clamped and no new spikes
            self.ref_counter -= 1
            spike = torch.tensor(0.0) # No spike during refractory
            return spike, self.V # Return clamped V
        else:
            # Integrate membrane potential
            dV = (-self.g_L * (self.V - self.E_L) + I_syn) / self.C
            self.V = self.V + dV * self.dt

            # Calculate spiking probability
            # Clamp V if it goes too high/low to avoid numerical issues with tanh for very large inputs
            V_clamped = torch.clamp(self.V, min=-100.0, max=100.0) # Example clamping
            prob_spike = (torch.tanh(self.beta * (V_clamped + self.V_offset)) + 1) / 2.0

            # Probabilistic spiking
            spike = torch.where(torch.rand_like(prob_spike) < prob_spike, torch.tensor(1.0), torch.tensor(0.0))

            # Apply reset and set refractory period if spike occurred
            if spike.item() == 1.0: # Check if a spike actually occurred
                self.V = torch.tensor(self.V_reset)
                self.ref_counter = self.ref_steps

            return spike, self.V
            
    def reset(self):
        """
        Reset the neuron to its initial state.
        """
        self.V = torch.tensor(self.V_rest)
        self.ref_counter = 0
        
    def set_membrane_potential(self, V):
        """
        Set the membrane potential to a specific value.
        
        Args:
            V (float or torch.Tensor): New membrane potential value
        """
        self.V = torch.tensor(V) if not isinstance(V, torch.Tensor) else V


class ProbabilisticLIFActivation(nn.Module):
    """
    A wrapper class that makes ProbabilisticSpikeNeuron compatible with LIFNode interface.
    This can be used as a drop-in replacement for LIFNode in existing code.
    """
    
    def __init__(self, dt=1e-3, C=1.0, g_L=0.1, E_L=-70.0, V_reset=-80.0, V_rest=-70.0, 
                 beta=0.1, V_offset=0.0, tau_ref=2.0):
        """
        Initialize the ProbabilisticLIFActivation.
        
        Args:
            dt (float): Simulation timestep (default: 1e-3)
            C (float): Membrane capacitance (default: 1.0)
            g_L (float): Leak conductance (default: 0.1)
            E_L (float): Leak reversal potential (default: -70.0)
            V_reset (float): Reset voltage after spike (default: -80.0)
            V_rest (float): Initial voltage or resting voltage (default: -70.0)
            beta (float): Steepness parameter for tanh activation (default: 0.1)
            V_offset (float): Voltage offset for tanh activation (default: 0.0)
            tau_ref (float): Refractory period (default: 2.0)
        """
        super(ProbabilisticLIFActivation, self).__init__()
        self.neuron = ProbabilisticSpikeNeuron(dt, C, g_L, E_L, V_reset, V_rest, beta, V_offset, tau_ref)
        
    def forward(self, x):
        """
        Forward pass compatible with LIFNode interface.
        
        Args:
            x (torch.Tensor): Input tensor (treated as current)
            
        Returns:
            torch.Tensor: Output spikes with same shape as input
        """
        # Handle batch dimensions
        original_shape = x.shape
        x_flat = x.flatten()
        
        # Process each element
        output = torch.zeros_like(x_flat)
        for i in range(x_flat.size(0)):
            spike, _ = self.neuron(x_flat[i])
            output[i] = spike
        
        # Restore original shape
        return output.reshape(original_shape)
    
    def reset(self):
        """Reset the neuron state."""
        self.neuron.reset()