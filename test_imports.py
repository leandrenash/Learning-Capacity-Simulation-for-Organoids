import streamlit as st
import numpy as np
import plotly.graph_objects as go
from models.organoid_model import OrganoidSimulation
from models.reinforcement import ReinforcementLearner
from utils.visualization import plot_neuron_spike_raster, create_network_graph

print("All imports successful!")

# Test OrganoidSimulation
sim = OrganoidSimulation(
    num_neurons=100,
    connectivity=0.2,
    noise_level=0.1,
    excitatory_ratio=0.8,
    learning_rate=0.01,
    neuron_model="LIF"
)

print("OrganoidSimulation created successfully!")

# Test ReinforcementLearner
rl = ReinforcementLearner(
    state_size=50,
    action_size=10,
    learning_rate=0.001
)

print("ReinforcementLearner created successfully!")

# Test visualization functions
print("Testing visualization functions...")
test_data = np.random.rand(100, 100)
plot_neuron_spike_raster(test_data)
create_network_graph(np.random.rand(10, 10))

print("All tests completed successfully!") 