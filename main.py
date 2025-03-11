import streamlit as st
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure asyncio to use the event loop policy appropriate for the platform
try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except AttributeError:
    # Not on Windows, use default policy
    pass

st.set_page_config(
    page_title="Organoid Learning Simulation",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Organoid Learning Simulation Platform")

st.markdown("""
This platform simulates and analyzes the learning capacity of organoids using advanced neural models and machine learning.

### Enhanced Features:
- **Biologically-Inspired Neural Models**: 
  - Leaky Integrate-and-Fire (LIF)
  - Izhikevich Model
- **Synaptic Plasticity & Learning**:
  - Spike-Timing-Dependent Plasticity (STDP)
  - Reinforcement Learning Integration
- **Comprehensive Analysis**:
  - Information Theory Metrics
  - Frequency Domain Analysis  
  - 3D Visualization Tools
- **Network Structure Visualization**:
  - Connection Graphs
  - Weight Distribution Analysis
""")

# Display quick overview metrics
col1, col2, col3 = st.columns(3)

# Demo metrics
with col1:
    st.metric(label="Neuron Models", value="2", delta="New")
with col2:
    st.metric(label="Analysis Tools", value="12+", delta="Enhanced")
with col3:
    st.metric(label="Visualization Types", value="8+", delta="Advanced")

# Display neural model comparison
st.header("Available Neural Models")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Leaky Integrate-and-Fire (LIF)")
    st.write("""
    A simplified spiking neuron model that captures key neuronal dynamics:
    - Membrane potential integration
    - Threshold-based spiking
    - Refractory periods
    - Efficient for large-scale simulations
    """)

with col2:
    st.subheader("Izhikevich Model")
    st.write("""
    A biologically realistic model that reproduces various neuronal behaviors:
    - Tonic spiking
    - Phasic spiking
    - Mixed-mode oscillations
    - Computationally efficient while maintaining biophysical realism
    """)

# Display learning capabilities
st.header("Learning & Plasticity")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Synaptic Plasticity (STDP)")
    st.write("""
    Implements biological learning through connection strength changes:
    - Connections strengthen when pre-synaptic neuron fires before post-synaptic
    - Connections weaken in the reverse case
    - Mimics actual brain learning mechanisms
    """)

with col2:
    st.subheader("Reinforcement Learning")
    st.write("""
    Integrated reinforcement learning for directed adaptation:
    - Stimulus pattern recognition
    - Activity optimization
    - Temporal pattern learning
    - Reward-based adaptation
    """)

# Display example visualizations
st.header("Advanced Visualization Features")

# Sample data for demonstration
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.sin(x*2)

col1, col2 = st.columns(2)

with col1:
    # Sample activity heatmap
    fig = go.Figure(data=go.Heatmap(
        z=np.outer(np.sin(x), np.cos(x)), 
        colorscale='Viridis'
    ))
    fig.update_layout(title="Neural Activity Heatmap")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Sample frequency analysis
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Signal 1'))
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Signal 2'))
    fig.update_layout(title="Multi-signal Analysis")
    st.plotly_chart(fig, use_container_width=True)

# Getting started section
st.header("Getting Started")
st.write("""
1. Navigate to the **Simulation** page to configure and run a neural simulation
2. Use the **Visualization** page to explore the results with advanced visualization tools
3. Analyze the performance with comprehensive metrics in the **Analysis** page
""")

# Footnote
st.markdown("---")
st.markdown("**Version 2.0 with enhanced neural models, reinforcement learning, and advanced analytics**")