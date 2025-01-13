import numpy as np
import tensorflow as tf

class ReinforcementLearner:
    def __init__(self, state_size, action_size, learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.build_model()
        
    def build_model(self):
        """Build the reinforcement learning model"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
    def get_action(self, state):
        """Predict action based on current state"""
        state = np.reshape(state, [1, self.state_size])
        return self.model.predict(state)[0]
    
    def train(self, state, action, reward, next_state):
        """Train the model using experience replay"""
        target = reward + 0.95 * np.max(self.get_action(next_state))
        target_f = self.get_action(state)
        target_f[action] = target
        
        state = np.reshape(state, [1, self.state_size])
        self.model.fit(state, np.reshape(target_f, [1, self.action_size]), epochs=1, verbose=0)
