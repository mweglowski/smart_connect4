import torch
import torch.nn as nn # Neural network
import torch.optim as optim # Optimizer (gradient descent, adam etc.)
import numpy as np
from collections import deque


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
    def __init__(self, state_shape, num_actions, batch_size=64, learning_rate=0.001, gamma=0.95, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01, memory_size=10000):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # passing 42 (shape -> 6 * 7) and num_actions to model
        self.model = NeuralNetwork(np.prod(state_shape), num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
    # choose action
    def choose_action(self, state):
        print(state.flatten())


print(torch.cuda.is_available())