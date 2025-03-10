import numpy as np

class OrganoidSimulation:
    def __init__(self, num_neurons=500, connectivity=0.2, noise_level=0.1, 
                 excitatory_ratio=0.8, learning_rate=0.01, 
                 neuron_model="LIF"):
        self.num_neurons = num_neurons
        self.connectivity = connectivity
        self.noise_level = noise_level
        self.excitatory_ratio = excitatory_ratio
        self.learning_rate = learning_rate
        self.neuron_model = neuron_model
        self.initialize_network()
        
    def initialize_network(self):
        """Initialize the neural network structure with biologically-inspired properties"""
        # Create neuron types (excitatory or inhibitory)
        self.neuron_types = np.random.rand(self.num_neurons) < self.excitatory_ratio
        
        # Create connectivity matrix
        self.connections = np.random.rand(self.num_neurons, self.num_neurons)
        self.connections = (self.connections < self.connectivity).astype(float)
        
        # Initialize weights with excitatory (positive) and inhibitory (negative) values
        weights = np.random.normal(0, 1, (self.num_neurons, self.num_neurons))
        # Excitatory neurons have positive weights
        weights[self.neuron_types] = np.abs(weights[self.neuron_types])
        # Inhibitory neurons have negative weights
        weights[~self.neuron_types] = -np.abs(weights[~self.neuron_types])
        
        # Apply connectivity mask
        self.weights = weights * self.connections
        
        # Initialize neural states and other properties
        self.states = np.zeros(self.num_neurons)
        self.membrane_potential = np.zeros(self.num_neurons)
        self.refractory_count = np.zeros(self.num_neurons)
        
        # Neuron model parameters
        if self.neuron_model == "LIF":  # Leaky Integrate-and-Fire
            self.threshold = 1.0
            self.rest_potential = 0.0
            self.refractory_period = 5
            self.leak_rate = 0.1
        elif self.neuron_model == "Izhikevich":
            self.a = np.random.uniform(0.02, 0.1, self.num_neurons)
            self.b = np.random.uniform(0.2, 0.25, self.num_neurons)
            self.c = np.random.uniform(-65, -50, self.num_neurons)
            self.d = np.random.uniform(2, 8, self.num_neurons)
            self.v = np.zeros(self.num_neurons) - 65
            self.u = self.b * self.v
            
        # Learning history
        self.learning_history = []
        self.spikes_history = []

    def generate_stimulus_pattern(self, pattern_type="Simple Pulse", duration=100):
        """Generate different types of stimulus patterns"""
        time_points = np.linspace(0, duration, duration)

        if pattern_type == "Simple Pulse":
            stimulus = np.zeros(duration)
            stimulus[10:20] = 1.0
        elif pattern_type == "Complex Pattern":
            stimulus = 0.5 * (np.sin(2 * np.pi * time_points / 20) + 
                            np.sin(2 * np.pi * time_points / 10))
        elif pattern_type == "Oscillatory":
            stimulus = 0.8 * np.sin(2 * np.pi * time_points / 15)
        elif pattern_type == "Multi-frequency":
            stimulus = (0.5 * np.sin(2 * np.pi * time_points / 10) + 
                       0.3 * np.sin(2 * np.pi * time_points / 5) +
                       0.2 * np.sin(2 * np.pi * time_points / 20))
        else:  # Random Sequence
            stimulus = np.random.rand(duration)

        return stimulus

    def apply_stimulus(self, stimulus):
        """Apply external stimulus to the network"""
        # Add noise to the stimulus
        noise = np.random.normal(0, self.noise_level, self.num_neurons)
        stimulus_with_noise = stimulus * np.ones(self.num_neurons) + noise
        
        # Apply stimulus based on neuron model
        if self.neuron_model == "LIF":
            # Add stimulus to membrane potential
            self.membrane_potential += stimulus_with_noise
        elif self.neuron_model == "Izhikevich":
            # Stimulus is added to membrane voltage
            self.v += stimulus_with_noise
        
        return stimulus_with_noise

    def update_state(self):
        """Update neural states based on neuron model"""
        if self.neuron_model == "LIF":
            return self._update_lif()
        elif self.neuron_model == "Izhikevich":
            return self._update_izhikevich()
        else:
            # Simple model
            new_states = np.dot(self.weights, self.states)
            self.states = np.tanh(new_states)
            return self.states
    
    def _update_lif(self):
        """Update using Leaky Integrate-and-Fire model"""
        # Apply leak
        self.membrane_potential = self.membrane_potential * (1 - self.leak_rate)
        
        # Check for spiking neurons
        spiking = (self.membrane_potential > self.threshold) & (self.refractory_count <= 0)
        
        # Record spikes
        self.states = spiking.astype(float)
        
        # Reset membrane potential for spiking neurons
        self.membrane_potential[spiking] = self.rest_potential
        
        # Set refractory period for spiking neurons
        self.refractory_count[spiking] = self.refractory_period
        
        # Decrease refractory counter
        self.refractory_count = np.maximum(0, self.refractory_count - 1)
        
        return self.states
    
    def _update_izhikevich(self):
        """Update using Izhikevich model"""
        # Izhikevich model equations
        v = self.v
        u = self.u
        
        # Neurons that spike
        spiking = v >= 30
        
        # Record spikes
        self.states = spiking.astype(float)
        
        # Reset after spike
        v[spiking] = self.c[spiking]
        u[spiking] = u[spiking] + self.d[spiking]
        
        # Update membrane voltage and recovery variable
        dv = 0.04 * v**2 + 5 * v + 140 - u
        du = self.a * (self.b * v - u)
        
        self.v = v + dv
        self.u = u + du
        
        return self.states
    
    def apply_learning(self, prev_states):
        """Apply synaptic plasticity using STDP rule"""
        if np.sum(self.states) == 0 or np.sum(prev_states) == 0:
            return
        
        # STDP (Spike-Timing-Dependent Plasticity)
        # Neurons that fired together have their connections strengthened
        pre_post = np.outer(prev_states, self.states)
        post_pre = np.outer(self.states, prev_states)
        
        # Strengthen connections where pre-synaptic neuron fired before post-synaptic
        self.weights += self.learning_rate * pre_post * self.connections
        
        # Weaken connections where post-synaptic neuron fired before pre-synaptic
        self.weights -= 0.5 * self.learning_rate * post_pre * self.connections
        
        # Respect neuron types (excitatory vs inhibitory)
        self.weights[self.neuron_types] = np.maximum(0, self.weights[self.neuron_types])
        self.weights[~self.neuron_types] = np.minimum(0, self.weights[~self.neuron_types])
        
        # Normalize weights to prevent runaway connections
        sum_in = np.sum(np.abs(self.weights), axis=0)
        sum_in[sum_in == 0] = 1  # Avoid division by zero
        self.weights = self.weights / sum_in * self.num_neurons * self.connectivity
        
        # Track learning progress
        self.learning_history.append(np.mean(np.abs(pre_post - post_pre)))

    def run_simulation(self, pattern_type="Simple Pulse", duration=100, learning=True):
        """Run a complete simulation with the specified pattern"""
        # Generate stimulus
        stimulus = self.generate_stimulus_pattern(pattern_type, duration)

        # Initialize results storage
        response_history = np.zeros((duration, self.num_neurons))
        spikes_history = np.zeros((duration, self.num_neurons))
        learning_progress = np.zeros(duration)

        # Run simulation for each time step
        for t in range(duration):
            # Save previous state for learning
            prev_states = self.states.copy()
            
            # Apply stimulus for this time step
            self.apply_stimulus(stimulus[t])
            
            # Update network state
            self.update_state()
            
            # Apply learning if enabled
            if learning:
                self.apply_learning(prev_states)
                if len(self.learning_history) > 0:
                    learning_progress[t] = self.learning_history[-1]
            
            # Store response
            response_history[t] = self.membrane_potential if self.neuron_model == "LIF" else self.v
            spikes_history[t] = self.states

        return {
            'stimulus': stimulus,
            'response': response_history,
            'spikes': spikes_history,
            'time_points': np.arange(duration),
            'learning_progress': learning_progress,
            'neuron_types': self.neuron_types
        }