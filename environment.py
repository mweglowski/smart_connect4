import numpy as np

NUM_ROWS = 7
NUM_COLUMNS = 6

class Environment:
    def __init__(self):
        self.board = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=int)
        self.current_player = 1
        
    def step(self, action):
        # action -> column_index
        
        for row_index in range(NUM_ROWS - 1, -1, -1):
            # Check first empty space starting from the bottom
            if self.board[row_index][action] == 0:
                # Placing current player "coin" in free space
                self.board[row_index][action] = self.current_player
                break
            
        print(self.board)
        reward = 0
        return reward
    
    # def get_reward(self):
    
env = Environment()