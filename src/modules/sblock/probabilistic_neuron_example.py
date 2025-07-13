import torch
import matplotlib.pyplot as plt
import numpy as np
from .probabilistic_neuron import ProbabilisticSpikeNeuron


def example_usage():
    """
    Example usage of the ProbabilisticSpikeNeuron model.
    """
    # Neuron parameters
    dt = 0.1  # ms
    C = 1.0   # membrane capacitance
    g_L = 0.1 # leak conductance
    E_L = -70.0 # leak reversal potential (mV)
    V_reset = -80.0 # reset voltage (mV)
    V_rest = -70.0  # resting voltage (mV)
    beta = 0.1      # steepness parameter for tanh
    V_offset = 0.0  # voltage offset for tanh
    tau_ref = 2.0   # refractory period (ms)
    
    # Create neuron
    neuron = ProbabilisticSpikeNeuron(
        dt=dt, C=C, g_L=g_L, E_L=E_L, 
        V_reset=V_reset, V_rest=V_rest, 
        beta=beta, V_offset=V_offset, tau_ref=tau_ref
    )
    
    # Simulation parameters
    t_sim = 100.0  # simulation time (ms)
    n_steps = int(t_sim / dt)
    
    # Input current (step input)
    I_input = torch.zeros(n_steps)
    I_input[200:800] = 2.0  # 2 nA input from 20ms to 80ms
    
    # Storage for results
    spikes = []
    membrane_potentials = []
    
    # Run simulation
    for i in range(n_steps):
        spike, V = neuron.forward(I_input[i])
        spikes.append(spike.item())
        membrane_potentials.append(V.item())
    
    # Time array
    time = np.arange(n_steps) * dt
    
    return time, spikes, membrane_potentials, I_input.numpy()


def plot_results(time, spikes, membrane_potentials, input_current):
    """
    Plot the simulation results.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot input current
    ax1.plot(time, input_current, 'b-', linewidth=2)
    ax1.set_ylabel('Input Current (nA)')
    ax1.set_title('Probabilistic Spiking Neuron Simulation')
    ax1.grid(True)
    
    # Plot membrane potential
    ax2.plot(time, membrane_potentials, 'r-', linewidth=1)
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.grid(True)
    
    # Plot spikes
    spike_times = time[np.array(spikes) > 0.5]
    ax3.eventplot(spike_times, colors='black', lineoffsets=1, linelengths=0.8)
    ax3.set_ylim(0.5, 1.5)
    ax3.set_ylabel('Spikes')
    ax3.set_xlabel('Time (ms)')
    ax3.grid(True)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Run example
    time, spikes, membrane_potentials, input_current = example_usage()
    
    # Plot results
    fig = plot_results(time, spikes, membrane_potentials, input_current)
    plt.show()
    
    # Print some statistics
    spike_count = sum(spikes)
    spike_rate = spike_count / (time[-1] / 1000.0)  # Hz
    print(f"Total spikes: {spike_count}")
    print(f"Average firing rate: {spike_rate:.2f} Hz")
    print(f"Simulation completed successfully!")