import plotly.graph_objects as go
import numpy as np

def create_activity_heatmap(activity_data):
    """Create a heatmap of neural activity"""
    fig = go.Figure(data=go.Heatmap(
        z=activity_data,
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title="Neural Activity Heatmap",
        xaxis_title="Neuron Index X",
        yaxis_title="Neuron Index Y"
    )
    
    return fig

def plot_learning_curve(epochs, accuracy):
    """Plot the learning curve"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=accuracy,
        mode='lines+markers',
        name='Learning Progress'
    ))
    
    fig.update_layout(
        title="Learning Curve",
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
