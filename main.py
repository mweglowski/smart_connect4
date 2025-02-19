import numpy as np
from agent import Agent
from environment import Environment

# Initializing environment and agent parameters
env = Environment()
state_shape = env.board.shape
num_actions = env.board.shape[1]

# Initializing agent
agent = Agent(state_shape, num_actions)


# Training
episodes = 1000
for episode in range(episodes):
    # Get initial state of environment
    state = env.reset()
    
    # Flatten this state
    state = state.flatten()
    
    # Initialize total reward
    total_reward = 0
    
    # Initialize terminal flag
    terminal = False
    
    
    # Training loop
    while not terminal:
        # Agent takes an action
        env.current_player = 1
        action = agent.choose_action(state)
        
        # Observe environment and get next_state, reward and terminal flag after agent takes action
        next_state, reward, terminal = env.step(action)
        
        # Flatten state, because agents like flattened arrays
        next_state = next_state.flatten()
        
        # Store experience in agent's experience buffer
        agent.store_experience(state, action, reward, next_state, terminal)
        
        # Train neural network to better predict future moves
        agent.train()
        
        # Update state
        state = next_state
        
        # Increase cumulative reward for this episode
        total_reward += reward
        
        # print(env.board, reward)


        env.current_player = 2
        # Random action (training agent on random agent)
        # if episode < 100:
        #     action = np.random.randint(0, num_actions)
        #     next_state, _, _ = env.step(action)
        #     state = next_state.flatten()
        # # Move of the second player which is also agent. Action based on agent's neural network
        # else:
        action = agent.choose_action(state)
        next_state, _, _ = env.step(action)
        state = next_state.flatten()
        
    print(f"Episode {episode}, Total Reward: {total_reward}")
        
agent.save_model("model.pth")