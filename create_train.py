import numpy as np
import pickle
import os

class QLearning:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_values = np.zeros((num_states, num_actions))

    def choose_action(self, state, explore=True):
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_values[state])

    def update_q_values(self, state, action, reward, next_state):
        current_q = self.q_values[state, action]
        max_next_q = np.max(self.q_values[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_values[state, action] = new_q

    def train(self, states, rewards, actions, next_states):
        for state, action, reward, next_state in zip(states, rewards, actions, next_states):
            self.update_q_values(state, action, reward, next_state)

class AIModel:
    def __init__(self):
        self.state_map = ['state_0', 'state_1']
        self.action_map = ['action_0', 'action_1']
        self.q_learning_model = QLearning(
            num_states=len(self.state_map),
            num_actions=len(self.action_map),
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1
        )

    def get_next_action(self, current_state, explore=True):
        return self.q_learning_model.choose_action(current_state, explore=explore)

    def update_q_values(self, current_state, action_taken, reward_received, next_state):
        self.q_learning_model.update_q_values(current_state, action_taken, reward_received, next_state)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"AI model saved to {filename}")

    @classmethod
    def load_model(cls, filename):
        instance = cls()
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            instance.__dict__.update(data)
        return instance

if __name__ == "__main__":
    # Initialize AI model
    ai = AIModel()
    
    # Training parameters
    episodes = 100000  # Increased for meaningful learning
    save_interval = 10000  # Save every 100 episodes
    max_steps_per_episode = 100  # Prevent infinite loops
    model_dir = "./model"
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Training loop
    for episode in range(episodes):
        current_state = 0
        done = False
        step = 0
        
        while not done and step < max_steps_per_episode:
            # Choose action with exploration
            action = ai.get_next_action(current_state, explore=True)
            
            # Simulate environment
            if current_state == 0:
                next_state = 1 if action == 0 else 0
                reward = 1.0 if action == 0 else -1.0
            else:
                next_state = 0 if action == 1 else 1
                reward = -1.0 if action == 1 else 1.0
                
            # Update Q-values
            ai.update_q_values(current_state, action, reward, next_state)
            
            # Move to next state
            current_state = next_state
            
            # Termination condition: episode ends when reaching state 1
            done = current_state == 1
            step += 1
        
        # Save model at intervals
        if (episode + 1) % save_interval == 0:
            save_path = os.path.join(model_dir, f"ai_model_epoch_{episode+1}.pkl")
            ai.save_model(save_path)
    
    # Final save
    ai.save_model(os.path.join(model_dir, "ai_model_final.pkl"))
    
    # Test the trained model
    loaded_ai = AIModel.load_model(os.path.join(model_dir, "ai_model_final.pkl"))
    print("\nTrained Q-values:")
    print(loaded_ai.q_learning_model.q_values)
    
    # Test policy
    test_state = 0
    print(f"\nOptimal action for state {test_state}:", 
          loaded_ai.action_map[loaded_ai.get_next_action(test_state, explore=False)])