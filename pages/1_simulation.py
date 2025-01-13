import streamlit as st
import numpy as np
import plotly.graph_objects as go
from models.organoid_model import OrganoidSimulation

st.set_page_config(page_title="Simulation Configuration", page_icon="⚙️", layout="wide")

st.title("⚙️ Simulation Configuration")

# Simulation parameters
st.header("Simulation Parameters")

col1, col2 = st.columns(2)

with col1:
    num_neurons = st.slider("Number of Neurons", 100, 1000, 500)
    stimulus_duration = st.slider("Stimulus Duration (time steps)", 100, 1000, 500)

with col2:
    connectivity = st.slider("Neural Connectivity (%)", 1, 100, 20)
    noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1)

# Stimulus pattern configuration
st.header("Stimulus Pattern")
pattern_type = st.selectbox(
    "Select Stimulus Pattern",
    ["Simple Pulse", "Complex Pattern", "Random Sequence"]
)

# Initialize or run simulation button
if st.button("Initialize/Run Simulation"):
    try:
        with st.spinner("Running simulation..."):
            # Create simulation instance
            sim = OrganoidSimulation(
                num_neurons=num_neurons,
                connectivity=connectivity/100,
                noise_level=noise_level
            )

            # Run simulation
            results = sim.run_simulation(
                pattern_type=pattern_type,
                duration=stimulus_duration
            )

            # Store results
            st.session_state['latest_results'] = results
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

            # Neural activity heatmap
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
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")