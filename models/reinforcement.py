import numpy as np
import tensorflow as tf
from collections import deque
import random

class ReinforcementLearner:
    def __init__(self, state_size, action_size, learning_rate=0.01, 
                 memory_size=10000, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        self.memory = deque(maxlen=memory_size)
        self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        
        self.cumulative_reward = 0
        self.rewards_history = []
        self.loss_history = []
        
    def build_model(self):
        """Build the reinforcement learning model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
        
    def update_target_model(self):
        """Update the target network to match the main network"""
        self.target_model.set_weights(self.model.get_weights())
        
    def memorize(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
        self.cumulative_reward += reward
        
    def get_action(self, state):
        """Get action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            # Exploration: select random action
            return random.randrange(self.action_size)
            
        # Exploitation: select best action
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values)
    
    def replay(self):
        """Train the model using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0  # Not enough samples
            
        # Sample a batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])
            
            # Calculate target Q value
            if done:
                target = reward
            else:
                # Double DQN: select action using policy network, evaluate using target network
                next_action = np.argmax(self.model.predict(next_state, verbose=0)[0])
                target = reward + self.gamma * self.target_model.predict(next_state, verbose=0)[0][next_action]
            
            # Build target vector
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            states.append(state[0])
            targets.append(target_f[0])
            
        # Train the network
        history = self.model.fit(np.array(states), np.array(targets), 
                        epochs=1, verbose=0, batch_size=self.batch_size)
        
        # Record loss
        if history.history.get('loss'):
            self.loss_history.append(history.history['loss'][0])
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return np.mean(self.loss_history[-10:]) if self.loss_history else 0
        
    def integrate_with_organoid(self, organoid_simulation, stimulus_patterns, episodes=100):
        """Integrate reinforcement learning with organoid simulation"""
        rewards_per_episode = []
        
        for episode in range(episodes):
            # Reset environment
            organoid_simulation.initialize_network()
            pattern_idx = np.random.randint(0, len(stimulus_patterns))
            pattern_type = stimulus_patterns[pattern_idx]
            
            # Get initial state (use a sample of neurons as state)
            state = organoid_simulation.states[:self.state_size]
            
            # Run episode
            total_reward = 0
            
            for t in range(100):  # 100 time steps per episode
                # Select action
                action = self.get_action(state)
                
                # Convert action to stimulus strength
                stimulus_strength = (action / (self.action_size - 1)) * 2.0  # Scale to 0-2
                
                # Apply stimulus
                stimulus = organoid_simulation.generate_stimulus_pattern(pattern_type, 1)[0] * stimulus_strength
                organoid_simulation.apply_stimulus(stimulus)
                organoid_simulation.update_state()
                
                # Get new state
                next_state = organoid_simulation.states[:self.state_size]
                
                # Define reward function based on desired behavior
                reward = self._calculate_reward(organoid_simulation, pattern_idx)
                
                # Check if done
                done = (t == 99)
                
                # Remember experience
                self.memorize(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                
                # Accumulate reward
                total_reward += reward
                
                # Train the model
                if len(self.memory) > self.batch_size:
                    self.replay()
            
            # Update target network every few episodes
            if episode % 10 == 0:
                self.update_target_model()
                
            rewards_per_episode.append(total_reward)
            self.rewards_history.append(total_reward)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(rewards_per_episode[-10:])
                print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        return {
            'rewards': rewards_per_episode,
            'loss': self.loss_history
        }
        
    def _calculate_reward(self, organoid_simulation, pattern_idx):
        """Calculate reward based on organoid behavior"""
        # Get current states
        spikes = organoid_simulation.states
        
        # Calculate rewards based on different objectives
        
        # 1. Reward for having a certain activity level (not too high, not too low)
        activity_level = np.mean(spikes)
        activity_reward = -((activity_level - 0.3) ** 2) + 1.0
        
        # 2. Reward for synchronized firing of excitatory neurons for certain patterns
        if pattern_idx == 0:  # For simple pulse, reward synchronization
            synchrony = np.std(spikes[organoid_simulation.neuron_types])
            synchrony_reward = synchrony * 2.0
        else:
            synchrony_reward = 0
            
        # 3. Reward for specific spatial patterns (e.g., groups of neurons firing together)
        pattern_reward = 0
        if np.sum(spikes) > 0:
            neuron_clusters = 3
            cluster_size = organoid_simulation.num_neurons // neuron_clusters
            for i in range(neuron_clusters):
                start_idx = i * cluster_size
                end_idx = (i + 1) * cluster_size
                cluster_activity = np.mean(spikes[start_idx:end_idx])
                pattern_reward += 0.5 * (1 - np.abs(cluster_activity - (i+1)/neuron_clusters))
                
        # Combine rewards
        total_reward = 0.4 * activity_reward + 0.3 * synchrony_reward + 0.3 * pattern_reward
        
        return total_reward
