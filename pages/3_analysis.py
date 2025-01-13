import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from utils.metrics import calculate_performance_metrics

st.set_page_config(page_title="Analysis Tools", page_icon="🔍")

st.title("🔍 Analysis Tools")

if 'latest_results' in st.session_state:
    results = st.session_state['latest_results']

    # Performance metrics
    st.header("Performance Metrics")

    # Calculate metrics based on actual simulation results
    avg_response = np.mean(results['response'], axis=1)
    response_accuracy = np.corrcoef(results['stimulus'], avg_response)[0, 1] * 100
    memory_retention = np.mean(np.abs(avg_response)) * 100
    learning_efficiency = np.mean(np.gradient(avg_response)) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Learning Efficiency", f"{learning_efficiency:.2f}%")
    with col2:
        st.metric("Response Accuracy", f"{response_accuracy:.2f}%")
    with col3:
        st.metric("Memory Retention", f"{memory_retention:.2f}%")

    # Detailed analysis
    st.header("Response Analysis")

    # Create time series data
    data = pd.DataFrame({
        'Time': results['time_points'],
        'Stimulus': results['stimulus'],
        'Response': avg_response
    })

    # Plot comparison
    fig = px.line(data, x='Time', y=['Stimulus', 'Response'], 
                 title='Stimulus-Response Comparison')
    st.plotly_chart(fig, use_container_width=True)

    # Statistical summary
    st.header("Statistical Summary")
    summary_data = {
        'Metric': ['Mean', 'Std Dev', 'Max', 'Min'],
        'Stimulus': [
            np.mean(results['stimulus']),
            np.std(results['stimulus']),
            np.max(results['stimulus']),
            np.min(results['stimulus'])
        ],
        'Response': [
            np.mean(avg_response),
            np.std(avg_response),
            np.max(avg_response),
            np.min(avg_response)
        ]
    }
    st.dataframe(pd.DataFrame(summary_data))

else:
    st.warning("No simulation data available. Please run a simulation first in the Simulation page.")