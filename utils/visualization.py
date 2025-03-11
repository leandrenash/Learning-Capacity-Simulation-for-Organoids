import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.signal import spectrogram

def create_activity_heatmap(activity_data, title="Neural Activity Heatmap"):
    """Create a heatmap of neural activity"""
    fig = go.Figure(data=go.Heatmap(
        z=activity_data,
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Neuron Index X",
        yaxis_title="Neuron Index Y"
    )
    
    return fig

def plot_learning_curve(epochs, accuracy, title="Learning Curve"):
    """Plot the learning curve"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=accuracy,
        mode='lines+markers',
        name='Learning Progress'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1]
    )
    
    return fig

def plot_response_pattern(time, response, title="Response Pattern"):
    """Plot stimulus response pattern"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time,
        y=response,
        mode='lines',
        name='Response'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Response"
    )
    
    return fig

def plot_neuron_spike_raster(spike_data, time_points=None, title="Neuron Spike Raster Plot"):
    """Create a raster plot of neuron spikes"""
    if time_points is None:
        time_points = np.arange(spike_data.shape[0])
    
    fig = go.Figure()
    
    # Plot spikes for each neuron
    for i in range(min(100, spike_data.shape[1])):  # Limit to 100 neurons for visibility
        # Get spike times for this neuron
        spike_times = time_points[spike_data[:, i] > 0.5]
        
        # Add scatter points for each spike
        if len(spike_times) > 0:
            y_values = np.ones_like(spike_times) * i
            fig.add_trace(go.Scatter(
                x=spike_times,
                y=y_values,
                mode='markers',
                marker=dict(size=4, color='blue'),
                showlegend=False
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Neuron Index",
        height=600
    )
    
    return fig

def create_network_graph(weights, neuron_types=None, threshold=0.1, max_neurons=100):
    """Create a graph visualization of neural network connectivity"""
    # Limit number of neurons for better visualization
    n_neurons = min(max_neurons, weights.shape[0])
    
    # Filter weights by threshold
    weights_filtered = weights[:n_neurons, :n_neurons].copy()
    weights_filtered[np.abs(weights_filtered) < threshold] = 0
    
    # Create positions for neurons in a circle
    theta = np.linspace(0, 2*np.pi, n_neurons)
    x_pos = np.cos(theta)
    y_pos = np.sin(theta)
    
    # Create figure
    fig = go.Figure()
    
    # Add edges for excitatory connections
    excitatory_mask = weights_filtered > 0
    if np.any(excitatory_mask):
        for i in range(n_neurons):
            for j in range(n_neurons):
                if weights_filtered[i, j] > 0:
                    fig.add_trace(go.Scatter(
                        x=[x_pos[i], x_pos[j], None],
                        y=[y_pos[i], y_pos[j], None],
                        mode='lines',
                        line=dict(color='rgba(0,0,255,0.3)', width=1),
                        hoverinfo='none',
                        showlegend=False
                    ))
    
    # Add edges for inhibitory connections
    inhibitory_mask = weights_filtered < 0
    if np.any(inhibitory_mask):
        for i in range(n_neurons):
            for j in range(n_neurons):
                if weights_filtered[i, j] < 0:
                    fig.add_trace(go.Scatter(
                        x=[x_pos[i], x_pos[j], None],
                        y=[y_pos[i], y_pos[j], None],
                        mode='lines',
                        line=dict(color='rgba(255,0,0,0.3)', width=1),
                        hoverinfo='none',
                        showlegend=False
                    ))
    
    # Add nodes
    if neuron_types is not None:
        neuron_types = neuron_types[:n_neurons]
        node_colors = ['blue' if t else 'red' for t in neuron_types]
    else:
        node_colors = ['blue'] * n_neurons
    
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers',
        marker=dict(
            size=10,
            color=node_colors,
            line=dict(width=1, color='black')
        ),
        text=[f"Neuron {i}" for i in range(n_neurons)],
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Neural Network Connectivity",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    return fig

def plot_frequency_analysis(signal, sampling_rate=1.0, title="Frequency Analysis"):
    """Perform frequency analysis of neural activity"""
    # Calculate spectrogram
    f, t, Sxx = spectrogram(signal, fs=sampling_rate)
    
    # Add small constant to avoid log(0)
    Sxx = Sxx + 1e-10
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=10 * np.log10(Sxx),  # Convert to dB
        x=t,
        y=f,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Frequency",
        yaxis_type="log"
    )
    
    return fig

def plot_response_distribution(response_data, title="Response Distribution"):
    """Plot the distribution of neural responses"""
    # Calculate distribution statistics
    mean_response = np.mean(response_data, axis=1)
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=mean_response,
        nbinsx=30,
        marker_color='blue',
        opacity=0.7
    ))
    
    # Add mean line
    fig.add_vline(
        x=np.mean(mean_response),
        line_color="red",
        line_width=2,
        annotation_text="Mean"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Response Value",
        yaxis_title="Frequency"
    )
    
    return fig

def create_3d_activity_plot(activity_data, time_points=None, subsample=5, title="3D Neural Activity"):
    """Create a 3D surface plot of neural activity over time"""
    if time_points is None:
        time_points = np.arange(activity_data.shape[0])
    
    # Subsample for better visualization performance
    activity = activity_data[::subsample, :min(100, activity_data.shape[1])]
    times = time_points[::subsample]
    
    # Create neuron indices
    neurons = np.arange(activity.shape[1])
    
    # Create meshgrid
    time_grid, neuron_grid = np.meshgrid(times, neurons)
    
    # Create 3D surface
    fig = go.Figure(data=[go.Surface(
        z=activity.T,
        x=time_grid,
        y=neuron_grid,
        colorscale='Viridis'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Neuron Index",
            zaxis_title="Activity",
        ),
        width=800,
        height=800
    )
    
    return fig

def plot_reinforcement_learning(rewards, loss=None, title="Reinforcement Learning Progress"):
    """Plot reinforcement learning progress"""
    fig = go.Figure()
    
    # Plot rewards
    fig.add_trace(go.Scatter(
        x=np.arange(len(rewards)),
        y=rewards,
        mode='lines',
        name='Reward',
        line=dict(color='blue')
    ))
    
    # Plot loss if available
    if loss is not None:
        # Create secondary y-axis for loss
        fig.add_trace(go.Scatter(
            x=np.arange(len(loss)),
            y=loss,
            mode='lines',
            name='Loss',
            line=dict(color='red'),
            yaxis="y2"
        ))
        
        # Update layout with second y-axis
        fig.update_layout(
            yaxis2=dict(
                title="Loss",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Reward",
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

def create_comparative_visualization(results1, results2, metric='response', title="Comparative Analysis"):
    """Create a comparative visualization between two simulation results"""
    fig = go.Figure()
    
    # Get data to compare
    data1 = None
    data2 = None
    
    if metric == 'response':
        if 'response' in results1 and 'response' in results2:
            data1 = np.mean(results1['response'], axis=1)
            data2 = np.mean(results2['response'], axis=1)
    elif metric == 'spikes':
        if 'spikes' in results1 and 'spikes' in results2:
            data1 = np.mean(results1['spikes'], axis=1)
            data2 = np.mean(results2['spikes'], axis=1)
    elif metric == 'learning':
        if 'learning_progress' in results1 and 'learning_progress' in results2:
            data1 = results1['learning_progress']
            data2 = results2['learning_progress']
    
    if data1 is not None and data2 is not None:
        # Define x-axis
        x1 = results1.get('time_points', np.arange(len(data1)))
        x2 = results2.get('time_points', np.arange(len(data2)))
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=x1,
            y=data1,
            mode='lines',
            name='Simulation 1'
        ))
        
        fig.add_trace(go.Scatter(
            x=x2,
            y=data2,
            mode='lines',
            name='Simulation 2'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=metric.capitalize()
        )
    
    return fig
