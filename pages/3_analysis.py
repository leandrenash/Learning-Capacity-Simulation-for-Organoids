import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.metrics import calculate_performance_metrics
from utils.visualization import plot_frequency_analysis, plot_response_distribution, create_3d_activity_plot
from scipy.stats import pearsonr

st.set_page_config(page_title="Analysis Tools", page_icon="🔍", layout="wide")

st.title("🔍 Analysis Tools")

if 'latest_results' in st.session_state:
    results = st.session_state['latest_results']
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Advanced Analysis", "Comparative"])
    
    with tab1:
        # Performance metrics section
        st.header("Performance Metrics")
        
        # Calculate comprehensive metrics
        metrics = calculate_performance_metrics(results)
        
        # Display metrics in rows
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Learning Efficiency", f"{metrics['learning_efficiency']:.2f}%")
            st.metric("Response Accuracy", f"{metrics['response_accuracy']:.2f}%")
        
        with col2:
            st.metric("Memory Retention", f"{metrics['memory_retention']:.2f}%")
            st.metric("Response Latency", f"{metrics['response_latency']:.2f} ms")
        
        with col3:
            st.metric("Adaptation Rate", f"{metrics['adaptation_rate']:.2f}")
            st.metric("Neural Diversity", f"{metrics['neural_diversity']:.2f}")
        
        # Information theory metrics
        st.subheader("Information Theory Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Information Complexity", f"{metrics.get('information_complexity', 0):.2f}")
        
        with col2:
            st.metric("Pattern Recognition", f"{metrics.get('pattern_recognition', 0):.2f}%")
        
        # Detailed metrics display
        st.subheader("Metrics Breakdown")
        
        # Create formatted metrics dataframe
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': [f"{v:.2f}" for v in metrics.values()],
            'Description': [
                "How efficiently the organoid learns from stimulus",
                "How accurately organoid responds to stimulus",
                "How well organoid retains learned patterns",
                "Time delay between stimulus and response",
                "How quickly organoid adapts to changing patterns",
                "Variety in neuron firing patterns",
                "Complexity of information processing",
                "Ability to recognize specific patterns"
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True)
    
    with tab2:
        # Advanced analysis section
        st.header("Neural Activity Analysis")
        
        # Extract response data
        response_data = results.get('response', np.array([]))
        spike_data = results.get('spikes', np.array([]))
        stimulus = results.get('stimulus', np.array([]))
        time_points = results.get('time_points', np.array([]))
        
        if len(response_data) > 0:
            avg_response = np.mean(response_data, axis=1)
            
            # Plot comparison
            st.subheader("Stimulus-Response Comparison")
            
            # Create time series data
            data = pd.DataFrame({
                'Time': time_points,
                'Stimulus': stimulus,
                'Response': avg_response
            })
            
            fig = px.line(data, x='Time', y=['Stimulus', 'Response'], 
                         title='Stimulus-Response Comparison')
            st.plotly_chart(fig, use_container_width=True)
            
            # Frequency analysis
            st.subheader("Frequency Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stimulus frequency analysis
                fig = plot_frequency_analysis(
                    stimulus, 
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
            
            # Distribution analysis
            st.subheader("Response Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Response distribution
                fig = plot_response_distribution(
                    response_data,
                    title="Neural Response Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Correlation analysis
                if len(stimulus) == len(avg_response):
                    corr, p_value = pearsonr(stimulus, avg_response)
                    
                    # Correlation scatter plot
                    fig = px.scatter(
                        x=stimulus,
                        y=avg_response,
                        title=f"Stimulus-Response Correlation (r={corr:.3f}, p={p_value:.3f})",
                        labels={'x': 'Stimulus', 'y': 'Response'}
                    )
                    
                    # Add regression line
                    fig.update_layout(
                        xaxis_title="Stimulus Strength",
                        yaxis_title="Neural Response"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # 3D visualization
            st.subheader("3D Activity Visualization")
            fig = create_3d_activity_plot(
                response_data,
                time_points=time_points,
                subsample=5,
                title="3D Neural Activity"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Comparative analysis
        st.header("Comparative Analysis")
        st.info("Run multiple simulations and compare their performance.")
        
        # Create a placeholder for past simulations
        if 'saved_simulations' not in st.session_state:
            st.session_state['saved_simulations'] = []
        
        # Option to save current simulation
        sim_name = st.text_input("Simulation Name", value=f"Simulation {len(st.session_state['saved_simulations']) + 1}")
        
        if st.button("Save Current Simulation for Comparison"):
            # Save the current simulation with a name
            sim_data = {
                'name': sim_name,
                'results': results,
                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state['saved_simulations'].append(sim_data)
            st.success(f"Simulation '{sim_name}' saved for comparison!")
        
        # Display comparison if we have saved simulations
        if len(st.session_state['saved_simulations']) > 0:
            # Comparison analysis
            st.subheader("Saved Simulations")
            
            # Show saved simulations
            saved_names = [sim['name'] for sim in st.session_state['saved_simulations']]
            saved_df = pd.DataFrame({
                'Name': [sim['name'] for sim in st.session_state['saved_simulations']],
                'Timestamp': [sim['timestamp'] for sim in st.session_state['saved_simulations']]
            })
            
            st.dataframe(saved_df, use_container_width=True)
            
            # Allow selecting simulations to compare
            compare_options = saved_names + ['Current Simulation']
            selected_sims = st.multiselect("Select Simulations to Compare", compare_options)
            
            if len(selected_sims) >= 2:
                # Get data for comparison
                comparison_data = []
                
                for sim_name in selected_sims:
                    if sim_name == 'Current Simulation':
                        sim_results = results
                        comparison_data.append({
                            'name': 'Current Simulation',
                            'results': sim_results
                        })
                    else:
                        # Find in saved simulations
                        for saved_sim in st.session_state['saved_simulations']:
                            if saved_sim['name'] == sim_name:
                                comparison_data.append({
                                    'name': saved_sim['name'],
                                    'results': saved_sim['results']
                                })
                
                # Create comparison metrics
                if len(comparison_data) >= 2:
                    st.subheader("Metrics Comparison")
                    
                    # Calculate metrics for all selected simulations
                    comparison_metrics = {}
                    
                    for sim in comparison_data:
                        metrics = calculate_performance_metrics(sim['results'])
                        comparison_metrics[sim['name']] = metrics
                    
                    # Create comparison dataframe
                    metrics_to_compare = [
                        'learning_efficiency', 'response_accuracy', 
                        'memory_retention', 'adaptation_rate',
                        'information_complexity', 'pattern_recognition'
                    ]
                    
                    comp_data = []
                    
                    for metric in metrics_to_compare:
                        row = {'Metric': metric.replace('_', ' ').title()}
                        for sim_name, sim_metrics in comparison_metrics.items():
                            row[sim_name] = f"{sim_metrics.get(metric, 0):.2f}"
                        comp_data.append(row)
                    
                    comp_df = pd.DataFrame(comp_data)
                    st.dataframe(comp_df, use_container_width=True)
                    
                    # Create bar chart comparison
                    st.subheader("Visual Comparison")
                    
                    # Prepare data for bar chart
                    chart_data = []
                    
                    for metric in metrics_to_compare:
                        for sim_name, sim_metrics in comparison_metrics.items():
                            chart_data.append({
                                'Simulation': sim_name,
                                'Metric': metric.replace('_', ' ').title(),
                                'Value': sim_metrics.get(metric, 0)
                            })
                    
                    chart_df = pd.DataFrame(chart_data)
                    
                    # Create bar chart
                    fig = px.bar(
                        chart_df, 
                        x='Metric', 
                        y='Value', 
                        color='Simulation',
                        barmode='group',
                        title="Metrics Comparison"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Response comparison
                    st.subheader("Response Patterns Comparison")
                    
                    fig = go.Figure()
                    
                    for sim in comparison_data:
                        response = sim['results'].get('response', np.array([]))
                        if len(response) > 0:
                            avg_response = np.mean(response, axis=1)
                            time_points = sim['results'].get('time_points', np.arange(len(avg_response)))
                            
                            fig.add_trace(go.Scatter(
                                x=time_points,
                                y=avg_response,
                                mode='lines',
                                name=sim['name']
                            ))
                    
                    fig.update_layout(
                        title="Average Response Comparison",
                        xaxis_title="Time",
                        yaxis_title="Average Response"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
else:
    st.warning("No simulation data available. Please run a simulation first in the Simulation page.")