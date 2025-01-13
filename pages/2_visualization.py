import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.visualization import create_activity_heatmap, plot_learning_curve

st.set_page_config(page_title="Real-time Visualization", page_icon="📊")

st.title("📊 Real-time Visualization")

# Activity visualization
st.header("Neural Activity")

# Check if simulation results exist
if 'latest_results' in st.session_state:
    results = st.session_state['latest_results']

    # Display heatmap
    st.subheader("Neural Activity Heatmap")
    fig = create_activity_heatmap(results['response'].T)
    st.plotly_chart(fig, use_container_width=True)

    # Stimulus and Response
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