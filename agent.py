import torch
import torch.nn as nn # Neural network
import torch.optim as optim # Optimizer (gradient descent, adam etc.)
import numpy as np
from collections import deque # Experience replay buffer


class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_shape, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, num_actions)
        
    def forward(self, x):
        # Maybe this line should be uncommented
        # x = x.view(x.size(0), -1) # Flattenning input
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class Agent:
    def __init__(self, state_shape, num_actions, batch_size=64, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, memory_size=10000):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # passing 42 (shape -> 6 * 7) and num_actions to model
        self.model = NeuralNetwork(np.prod(state_shape), num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        
    def choose_action(self, state):
        """
        Choose action in current state.
        Arguments:
            state (list[int]) - flattened state (environment board)
        """
        if np.random.rand() < 0.1: # Replace with self.epsilon
            # Random action
            return np.random.randint(0, self.num_actions)
        else:
            # Convert state to tensor of type float32
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            
            # Get Q-values from neural network
            q_values = self.model(state)
            
            # Get action with the highest value from the model
            return torch.argmax(q_values).item()
            
            
    def store_experience(self, state, action, reward, next_state, terminal):
        """
        Store experience (state, action, reward, next_state, terminal) in experience replay buffer.
        """
        self.memory.append((state, action, reward, next_state, terminal))
        
    
    def train(self):
        """
        Train neural network. Sample from experience buffer.
        """
        
        # Prevent sampling when memory does not have enough samples (lower than batch_size)
        if len(self.memory) < self.batch_size:
            return
        
        # Sampling from experience buffer
        sampled_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[sample_index] for sample_index in sampled_indices]
        
        states, actions, rewards, next_states, terminals = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device) # (batch_size, num states)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device) # (batch_size, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device) # (batch_size, 1)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device) # (batch_size, num next states)
        terminals = torch.tensor(terminals, dtype=torch.float32).unsqueeze(1).to(self.device) # (batch_size, 1)
        
        # Calculating Q-values using neural network
        # Using .gather() to select actions that has been taken by the agent
        q_values = self.model(states).gather(1, actions)
        
        # Calculate the Q-values for the next states using the current model
        # Select the maximum Q-value for each next state and add an extra dimension
        # .max(1)[0] selects the max Q-value among all actions in each batch
        # .unsqueeze(1) adds an extra dimension to match the shape of the rewards tensor [batch_size, 1]
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        
        # Calculate the target Q-values using the Bellman equation
        # The target Q-value is the reward plus the discounted maximum Q-value of the next state
        # If the episode is done (terminals == 1), the next Q-value is not considered
        target_q_values = rewards + (self.gamma * next_q_values * (1 - terminals))
        
        # Compute loss. Compare q_values predicted by our current model to target_q_values - values which should be predicted
        loss = self.loss_fn(q_values, target_q_values)
        
        # Clear gradients of all optimized torch.Tensor's
        self.optimizer.zero_grad()
        
        # Perform backpropagation, computation of gradients of the loss with respect to the network parameters (weights & biases)
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        
    def save_model(self, path):
        """
        Save model to .pth file using provided path.
        """
        torch.save(self.model.state_dict(), path)
        
        
    def load_model(self, path):
        """
        Load model from specified path.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

# Testing
agent = Agent(state_shape=np.zeros((6, 7)). shape, num_actions=6)

print(agent.choose_action(np.zeros((6, 7)).flatten()))