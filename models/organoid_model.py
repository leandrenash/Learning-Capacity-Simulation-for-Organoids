import numpy as np

class OrganoidSimulation:
    def __init__(self, num_neurons=500, connectivity=0.2, noise_level=0.1):
        self.num_neurons = num_neurons
        self.connectivity = connectivity
        self.noise_level = noise_level
        self.initialize_network()

    def initialize_network(self):
        """Initialize the neural network structure"""
        # Create connectivity matrix
        self.connections = np.random.rand(self.num_neurons, self.num_neurons)
        self.connections = (self.connections < self.connectivity).astype(float)

        # Initialize neural states
        self.states = np.zeros(self.num_neurons)

    def generate_stimulus_pattern(self, pattern_type="Simple Pulse", duration=100):
        """Generate different types of stimulus patterns"""
        time_points = np.linspace(0, duration, duration)

        if pattern_type == "Simple Pulse":
            stimulus = np.zeros(duration)
            stimulus[10:20] = 1.0
        elif pattern_type == "Complex Pattern":
            stimulus = 0.5 * (np.sin(2 * np.pi * time_points / 20) + 
                            np.sin(2 * np.pi * time_points / 10))
        else:  # Random Sequence
            stimulus = np.random.rand(duration)

        return stimulus

    def apply_stimulus(self, stimulus):
        """Apply external stimulus to the network"""
        # Add noise to the stimulus
        noise = np.random.normal(0, self.noise_level, self.num_neurons)
        self.states = stimulus * np.ones(self.num_neurons) + noise
        return self.states

    def update_state(self):
        """Update neural states based on connections"""
        new_states = np.dot(self.connections, self.states)
        self.states = np.tanh(new_states)  # Apply activation function
        return self.states

    def run_simulation(self, pattern_type="Simple Pulse", duration=100):
        """Run a complete simulation with the specified pattern"""
        # Generate stimulus
        stimulus = self.generate_stimulus_pattern(pattern_type, duration)

        # Initialize results storage
        response_history = np.zeros((duration, self.num_neurons))

        # Run simulation for each time step
        for t in range(duration):
            # Apply stimulus for this time step
            self.apply_stimulus(stimulus[t])
            # Update network state
            self.update_state()
            # Store response
            response_history[t] = self.states

        return {
            'stimulus': stimulus,
            'response': response_history,
            'time_points': np.arange(duration)
        }