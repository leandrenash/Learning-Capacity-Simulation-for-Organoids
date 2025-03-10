import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.visualization import (
    create_activity_heatmap, 
    plot_learning_curve, 
    plot_neuron_spike_raster, 
    create_network_graph,
    plot_frequency_analysis,
    create_3d_activity_plot,
    plot_reinforcement_learning
)

st.set_page_config(page_title="Real-time Visualization", page_icon="📊", layout="wide")

st.title("📊 Advanced Visualization")

# Create tabs for different visualization types
tab1, tab2, tab3 = st.tabs(["Neural Activity", "Network Structure", "Learning Dynamics"])

with tab1:
    # Neural activity visualization
    st.header("Neural Activity Visualization")

    # Check if simulation results exist
    if 'latest_results' in st.session_state:
        results = st.session_state['latest_results']
        
        # Choose visualization type
        viz_type = st.radio(
            "Visualization Type",
            ["Activity Heatmap", "Spike Raster", "3D Activity Plot", "Frequency Analysis"],
            horizontal=True
        )
        
        if viz_type == "Activity Heatmap":
            # Display heatmap
            st.subheader("Neural Activity Heatmap")
            fig = create_activity_heatmap(
                results['response'].T, 
                title="Neural Activity Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Spike Raster":
            # Display spike raster plot
            st.subheader("Spike Raster Plot")
            fig = plot_neuron_spike_raster(
                results['spikes'],
                time_points=results['time_points'],
                title="Neuron Spike Patterns"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "3D Activity Plot":
            # Display 3D activity visualization
            st.subheader("3D Neural Activity")
            
            # Subsample control
            subsample = st.slider("Visualization Resolution", 1, 10, 5, 
                              help="Higher values improve performance but reduce resolution")
            
            fig = create_3d_activity_plot(
                results['response'],
                time_points=results['time_points'],
                subsample=subsample,
                title="3D Neural Activity Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Frequency Analysis":
            # Display frequency analysis
            st.subheader("Frequency Analysis")
            
            # Calculate average response
            avg_response = np.mean(results['response'], axis=1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stimulus frequency analysis
                fig = plot_frequency_analysis(
                    results['stimulus'], 
                    sampling_rate=1.0,
                    title="Stimulus Frequency Analysis"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Response frequency analysis
                fig = plot_frequency_analysis(
                    avg_response,
                    sampling_rate=1.0,
                    title="Neural Response Frequency Analysis"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Stimulus and Response
        st.subheader("Stimulus and Response")
        col1, col2 = st.columns(2)

        with col1:
            # Stimulus pattern
            st.subheader("Stimulus Pattern")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['time_points'],
                y=results['stimulus'],
                mode='lines',
                name='Stimulus'
            ))
            fig.update_layout(title="Input Stimulus")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Response pattern
            st.subheader("Network Response")
            fig = go.Figure()
            avg_response = np.mean(results['response'], axis=1)
            fig.add_trace(go.Scatter(
                x=results['time_points'],
                y=avg_response,
                mode='lines',
                name='Average Response'
            ))
            fig.update_layout(title="Organoid Response")
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.warning("No simulation data available. Please run a simulation first in the Simulation page.")
        
with tab2:
    # Network structure visualization
    st.header("Network Structure")
    
    if 'organoid_sim' in st.session_state and 'latest_results' in st.session_state:
        # Get simulation and results
        sim = st.session_state['organoid_sim']
        results = st.session_state['latest_results']
        
        # Network visualization controls
        col1, col2 = st.columns(2)
        
        with col1:
            threshold = st.slider(
                "Connection Threshold", 
                0.0, 0.5, 0.05, 
                step=0.01,
                help="Filter connections by strength"
            )
            
        with col2:
            max_neurons = st.slider(
                "Maximum Neurons to Display", 
                10, 500, 150,
                help="Limit the number of neurons for better visualization"
            )
        
        # Display network graph
        st.subheader("Neural Network Connectivity")
        fig = create_network_graph(
            sim.weights, 
            neuron_types=results['neuron_types'], 
            threshold=threshold,
            max_neurons=max_neurons
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display weight distribution
        st.subheader("Connection Weight Distribution")
        
        # Filter weights for visualization
        weights = sim.weights.flatten()
        weights = weights[weights != 0]  # Remove zero weights
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=weights,
            nbinsx=50,
            marker_color='blue',
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Connection Weight Distribution",
            xaxis_title="Weight Value",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display neuron type distribution
        if 'neuron_types' in results:
            st.subheader("Neuron Type Distribution")
            
            excitatory_count = np.sum(results['neuron_types'])
            inhibitory_count = len(results['neuron_types']) - excitatory_count
            
            fig = go.Figure(data=[go.Pie(
                labels=['Excitatory', 'Inhibitory'],
                values=[excitatory_count, inhibitory_count],
                marker_colors=['#3498db', '#e74c3c']
            )])
            
            fig.update_layout(
                title="Neuron Type Distribution"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No simulation data available. Please run a simulation first in the Simulation page.")
        
with tab3:
    # Learning dynamics visualization
    st.header("Learning Dynamics")
    
    st.write("This section visualizes the learning dynamics of the simulation.")
    
    # Check for simulation results first
    if 'latest_results' in st.session_state:
        results = st.session_state['latest_results']
        
        # Check if learning progress is available
        if 'learning_progress' in results and len(results['learning_progress']) > 0:
            # Plot learning progress
            st.subheader("Learning Progress")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['time_points'],
                y=results['learning_progress'],
                mode='lines',
                name='Learning Rate'
            ))
            
            # Add trend line (moving average)
            window_size = min(20, len(results['learning_progress']) // 5)
            if window_size > 1:
                trend = np.convolve(
                    results['learning_progress'], 
                    np.ones(window_size)/window_size, 
                    mode='valid'
                )
                trend_x = results['time_points'][window_size-1:]
                
                fig.add_trace(go.Scatter(
                    x=trend_x,
                    y=trend,
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', width=3)
                ))
            
            fig.update_layout(
                title="Synaptic Plasticity Over Time",
                xaxis_title="Time Step",
                yaxis_title="Learning Rate"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No learning progress data available. Run a simulation with learning enabled.")
    
    # Check for reinforcement learning results
    if 'rl_results' in st.session_state and 'rl_model' in st.session_state:
        # Show reinforcement learning dynamics
        st.subheader("Reinforcement Learning Dynamics")
        
        rl_results = st.session_state['rl_results']
        rl_model = st.session_state['rl_model']
        
        # Plot reward history
        fig = plot_reinforcement_learning(
            rl_results['rewards'],
            rl_results.get('loss', None),
            title="Reinforcement Learning Progress"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional RL metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Final Exploration Rate (ε)", f"{rl_model.epsilon:.4f}")
        
        with col2:
            # Calculate average reward for last 10 episodes
            if len(rl_results['rewards']) >= 10:
                recent_avg = np.mean(rl_results['rewards'][-10:])
                st.metric("Recent Reward (last 10)", f"{recent_avg:.2f}")
            else:
                st.metric("Recent Reward", "N/A")
        
        with col3:
            # Calculate improvement rate
            if len(rl_results['rewards']) >= 2:
                first_rewards = np.mean(rl_results['rewards'][:10]) if len(rl_results['rewards']) >= 10 else rl_results['rewards'][0]
                last_rewards = np.mean(rl_results['rewards'][-10:]) if len(rl_results['rewards']) >= 10 else rl_results['rewards'][-1]
                improvement = last_rewards - first_rewards
                st.metric("Improvement", f"{improvement:.2f}")
            else:
                st.metric("Improvement", "N/A")
        
    else:
        if not ('latest_results' in st.session_state and 'learning_progress' in st.session_state.get('latest_results', {})):
            st.info("No learning data available. Run a simulation with learning enabled or a reinforcement learning experiment.")
    
# Add export functionality
if 'latest_results' in st.session_state:
    st.sidebar.header("Export Options")
    st.sidebar.info("This feature will be implemented in future updates.")