import numpy as np

def calculate_performance_metrics():
    """Calculate various performance metrics for the simulation"""
    # Sample metrics (replace with actual calculations)
    metrics = {
        'learning_efficiency': np.random.uniform(0.6, 0.9),
        'response_accuracy': np.random.uniform(70, 95),
        'memory_retention': np.random.uniform(60, 90),
        'response_latency': np.random.uniform(10, 50),
        'adaptation_rate': np.random.uniform(0.4, 0.8)
    }
    
    return metrics

def calculate_learning_rate(history):
    """Calculate the learning rate from training history"""
    if len(history) < 2:
        return 0
    
    # Calculate the rate of improvement
    improvements = np.diff(history)
    return np.mean(improvements)

def calculate_response_accuracy(predictions, targets):
    """Calculate accuracy of organoid responses"""
    if len(predictions) != len(targets):
        return 0
    
    errors = np.abs(predictions - targets)
    accuracy = 1 - np.mean(errors)
    return accuracy * 100

def calculate_memory_retention(initial_response, current_response):
    """Calculate memory retention rate"""
    if len(initial_response) != len(current_response):
        return 0
    
    similarity = 1 - np.mean(np.abs(initial_response - current_response))
    return similarity * 100
