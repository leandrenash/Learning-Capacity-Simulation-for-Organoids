import streamlit as st
import numpy as np
import plotly.graph_objects as go
from models.organoid_model import OrganoidSimulation
from models.reinforcement import ReinforcementLearner
from utils.visualization import plot_neuron_spike_raster, create_network_graph

st.set_page_config(page_title="Simulation Configuration", page_icon="⚙️", layout="wide")

st.title("⚙️ Simulation Configuration")

# Create tabs for different simulation modes
tab1, tab2 = st.tabs(["Standard Simulation", "Reinforcement Learning"])

with tab1:
    # Simulation parameters
    st.header("Simulation Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_neurons = st.slider("Number of Neurons", 100, 1000, 500)
        stimulus_duration = st.slider("Stimulus Duration (time steps)", 100, 1000, 500)

    with col2:
        connectivity = st.slider("Neural Connectivity (%)", 1, 100, 20)
        noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1)

    with col3:
        excitatory_ratio = st.slider("Excitatory/Inhibitory Ratio", 0.5, 0.9, 0.8)
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
        
    # Neural model selection
    neuron_model = st.selectbox(
        "Neuron Model",
        ["LIF", "Izhikevich"],
        help="LIF (Leaky Integrate-and-Fire) is simpler and faster. Izhikevich model is more biologically realistic."
    )

    # Stimulus pattern configuration
    st.header("Stimulus Pattern")
    pattern_type = st.selectbox(
        "Select Stimulus Pattern",
        ["Simple Pulse", "Complex Pattern", "Oscillatory", "Multi-frequency", "Random Sequence"]
    )
    
    # Learning configuration
    enable_learning = st.checkbox("Enable Learning", value=True, 
                                help="Enable synaptic plasticity (STDP) during simulation")

    # Initialize or run simulation button
    if st.button("Initialize/Run Simulation", key="standard_sim"):
        try:
            with st.spinner("Running simulation..."):
                # Create simulation instance
                sim = OrganoidSimulation(
                    num_neurons=num_neurons,
                    connectivity=connectivity/100,
                    noise_level=noise_level,
                    excitatory_ratio=excitatory_ratio,
                    learning_rate=learning_rate,
                    neuron_model=neuron_model
                )

                # Run simulation
                results = sim.run_simulation(
                    pattern_type=pattern_type,
                    duration=stimulus_duration,
                    learning=enable_learning
                )

                # Store results and model
                st.session_state['latest_results'] = results
                st.session_state['organoid_sim'] = sim
                st.success("Simulation completed!")

                # Display Results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Stimulus Pattern")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results['time_points'],
                        y=results['stimulus'],
                        mode='lines',
                        name='Stimulus'
                    ))
                    fig.update_layout(
                        xaxis_title="Time Step",
                        yaxis_title="Amplitude"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Network Response")
                    fig = go.Figure()
                    avg_response = np.mean(results['response'], axis=1)
                    fig.add_trace(go.Scatter(
                        x=results['time_points'],
                        y=avg_response,
                        mode='lines',
                        name='Average Response'
                    ))
                    fig.update_layout(
                        xaxis_title="Time Step",
                        yaxis_title="Average Activity"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Neural activity heatmap and spike raster
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Neural Activity Heatmap")
                    fig = go.Figure(data=go.Heatmap(
                        z=results['response'].T,
                        x=results['time_points'],
                        y=np.arange(num_neurons),
                        colorscale='Viridis'
                    ))
                    fig.update_layout(
                        xaxis_title="Time Step",
                        yaxis_title="Neuron Index",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Spike Raster Plot")
                    fig = plot_neuron_spike_raster(
                        results['spikes'],
                        time_points=results['time_points'],
                        title="Neuron Spike Activity"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Network connectivity visualization
                st.subheader("Neural Network Connectivity")
                fig = create_network_graph(
                    sim.weights, 
                    neuron_types=results['neuron_types'], 
                    threshold=0.05,
                    max_neurons=150
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Learning progress if enabled
                if enable_learning and len(results['learning_progress']) > 0:
                    st.subheader("Learning Progress")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results['time_points'],
                        y=results['learning_progress'],
                        mode='lines',
                        name='Learning Progress'
                    ))
                    fig.update_layout(
                        xaxis_title="Time Step",
                        yaxis_title="Learning Rate"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

with tab2:
    # Reinforcement Learning Configuration
    st.header("Reinforcement Learning with Organoids")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rl_num_neurons = st.slider("Number of Neurons", 100, 1000, 500, key="rl_neurons")
        rl_state_size = st.slider("State Size", 10, 100, 50, 
                               help="Number of neurons to use as state representation")
        rl_action_size = st.slider("Action Size", 2, 20, 10,
                                help="Number of possible stimulus strength levels")
    
    with col2:
        rl_episodes = st.slider("Training Episodes", 10, 500, 100)
        rl_learning_rate = st.slider("RL Learning Rate", 0.0001, 0.01, 0.001, 
                                  format="%.4f", key="rl_lr")
        rl_epsilon = st.slider("Initial Exploration Rate", 0.1, 1.0, 1.0)
    
    # Stimulus patterns for RL training
    st.subheader("Training Stimulus Patterns")
    rl_patterns = st.multiselect(
        "Select Stimulus Patterns for Training",
        ["Simple Pulse", "Complex Pattern", "Oscillatory", "Multi-frequency", "Random Sequence"],
        default=["Simple Pulse", "Complex Pattern"]
    )
    
    if len(rl_patterns) == 0:
        st.warning("Please select at least one stimulus pattern.")
    
    # Run RL simulation button
    if st.button("Run Reinforcement Learning", key="rl_sim") and len(rl_patterns) > 0:
        try:
            with st.spinner("Training reinforcement learning model..."):
                # Create organoid simulation
                sim = OrganoidSimulation(
                    num_neurons=rl_num_neurons,
                    connectivity=0.2,  # Default connectivity
                    noise_level=0.1,   # Default noise
                    excitatory_ratio=0.8,
                    learning_rate=0.01,
                    neuron_model="LIF"  # LIF is faster for RL
                )
                
                # Create RL model
                rl_model = ReinforcementLearner(
                    state_size=rl_state_size,
                    action_size=rl_action_size,
                    learning_rate=rl_learning_rate,
                    epsilon=rl_epsilon,
                    epsilon_decay=0.995,
                    epsilon_min=0.01,
                    batch_size=32
                )
                
                # Train the model
                rl_results = rl_model.integrate_with_organoid(
                    sim, 
                    stimulus_patterns=rl_patterns,
                    episodes=rl_episodes
                )
                
                # Store results
                st.session_state['rl_results'] = rl_results
                st.session_state['rl_model'] = rl_model
                st.success("Reinforcement learning training completed!")
                
                # Display results
                st.subheader("Training Progress")
                
                # Plot reward history
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=np.arange(len(rl_results['rewards'])),
                    y=rl_results['rewards'],
                    mode='lines',
                    name='Reward'
                ))
                fig.update_layout(
                    title="Reward History",
                    xaxis_title="Episode",
                    yaxis_title="Total Reward"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot loss history if available
                if 'loss' in rl_results and len(rl_results['loss']) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=np.arange(len(rl_results['loss'])),
                        y=rl_results['loss'],
                        mode='lines',
                        name='Loss'
                    ))
                    fig.update_layout(
                        title="Loss History",
                        xaxis_title="Training Step",
                        yaxis_title="Loss"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display final epsilon
                st.metric("Final Exploration Rate (Epsilon)", f"{rl_model.epsilon:.4f}")
                
                # Display cumulative reward
                st.metric("Cumulative Reward", f"{rl_model.cumulative_reward:.2f}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")