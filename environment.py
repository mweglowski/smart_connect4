import numpy as np
from utils import check_strike

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
        reward = self.get_reward()
        self.current_player = 3 - self.current_player
        return reward
    
    def get_reward(self):
        # check for player 2, 3, 4 (win)
        current_player_strike = {str(quantity): check_strike(self.board, self.current_player, quantity) for quantity in [2, 3, 4]}
        
        # check for opponent 2, 3
        second_player_strike = {str(quantity): check_strike(self.board, 3 - self.current_player, quantity) for quantity in [2, 3]}
        
        # Check for win
        if current_player_strike["4"]:
            return 1000
        
        # Calculate rewards
        
        print('current_player ->', current_player_strike)
        print('second_player ->', second_player_strike)
        return 0
    
env = Environment()
env.step(1)
env.step(2)
env.step(1)
env.step(1)
