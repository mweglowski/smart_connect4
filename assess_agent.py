import numpy as np
from agent import Agent
from environment import Environment
from utils import check_strike

# Initializing environment and agent parameters
env = Environment()
state_shape = env.board.shape
num_actions = env.board.shape[1]

# Initializing agent
agent = Agent(state_shape, num_actions)
agent.load_model("model.pth")

# Battle with random agent
# Training
episodes = 1000
win_count = 0
lose_count = 0
for episode in range(episodes):
    print(episode)
    # Get initial state of environment
    state = env.reset()
    
    # Flatten this state
    state = state.flatten()
    
    # Initialize terminal flag
    terminal = False

    while not terminal:
        # Agent takes an action
        env.current_player = 1
        action = agent.choose_action(state)
        
        # Observe environment and get next_state, reward and terminal flag after agent takes action
        next_state, reward, terminal = env.step(action)
        if check_strike(env.board, 1, 4):
            print("win")
            win_count += 1
            break
        
        # Flatten state, because agents like flattened arrays
        next_state = next_state.flatten()
        
        # Store experience in agent's experience buffer
        agent.store_experience(state, action, reward, next_state, terminal)
        
        # Update state
        state = next_state


        env.current_player = 2
        action = np.random.randint(0, num_actions)
        next_state, _, terminal = env.step(action)
        state = next_state.flatten()
        if check_strike(env.board, 2, 4):
            print("lose")
            lose_count += 1
            
print("WIN:", win_count)
print("LOSE:", lose_count)
print("DRAW:", episodes - win_count - lose_count)
print("ACCURACY:", win_count / episodes)