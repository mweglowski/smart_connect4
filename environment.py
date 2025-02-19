import numpy as np
from utils import check_strike

NUM_ROWS = 6
NUM_COLUMNS = 7

class Environment:
    def __init__(self):
        self.board = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=int)
        self.current_player = 1
        self.terminal = False
        
    # Reset board (fill again with zeros)
    def reset(self):
        """
        Resets the game.
        
        Returns:
            board (int[NUM_ROWS][NUM_COLUMNS]) - current state of the game
        """
        self.board = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=int)
        self.current_player = 1
        self.terminal = False
        
        return self.board
        
    # Perform step in environment, take certain action and observe what happens
    def step(self, action):
        # action -> column_index
        reward = 0

        # Invalid move (end game)
        if self.board[0][action] != 0:
            reward -= 100
            self.terminal = True
        
        for row_index in range(NUM_ROWS - 1, -1, -1):
            # Check first empty space starting from the bottom
            if self.board[row_index][action] == 0:
                # Placing current player "coin" in free space
                self.board[row_index][action] = self.current_player
                break

            
        reward = self.get_reward()
        
        current_player_won = check_strike(self.board, self.current_player, 4)
        second_player_won = check_strike(self.board, 3 - self.current_player, 4)

        # Victory
        if current_player_won:
            reward += 100
            self.terminal = True
        # Defeat
        elif second_player_won:
            reward -= 100
            self.terminal = True
        # Draw
        elif self.is_draw():
            self.terminal = True
        # else:
        #     self.current_player = 3 - self.current_player
        
        # print('in env.py', self.board, reward, self.terminal)
        
        return self.board, reward, self.terminal
    
    # Calculate reward
    def get_reward(self):
        # check for player 2, 3, 4 (win)
        current_player_strike = {str(quantity): check_strike(self.board, self.current_player, quantity) for quantity in [2, 3, 4]}
        
        # check for opponent 2, 3
        second_player_strike = {str(quantity): check_strike(self.board, 3 - self.current_player, quantity) for quantity in [2, 3]}
        
        # Calculate rewards
        reward = 0
        # Current player
        if current_player_strike["3"]:
            reward += 1
        
        # Second player
        if second_player_strike["3"]:
            reward -= 1
            
        # Check if second_player can end the game
        # print('current_player', self.current_player)
        board_copy = np.copy(self.board)
        
        # can_second_player_win = False
        # for action in range(NUM_COLUMNS):
        #     if can_second_player_win:
        #         break
            
        #     for row_index in range(NUM_ROWS - 1, -1, -1):
        #         # Check first empty space starting from the bottom
        #         if board_copy[row_index][action] == 0:
        #             # Placing current player "coin" in free space
        #             board_copy[row_index][action] = 3 - self.current_player
        #             # print('placing\n',  board_copy)
                    
        #             # Check if this move is winning
        #             if check_strike(board_copy, 3 - self.current_player, 4):
        #                 # print("second_player move can win!")
        #                 reward -= 100
        #                 break
                    
        #             # Remove coin from this space
        #             board_copy[row_index][action] = 0
        #             # print('removed\n', board_copy)
        #             can_second_player_win = True
        #             break
            
        return reward
    

    def is_draw(self):
        return np.all(self.board != 0)
    

# If this file is called not from another file
if __name__ == "__main__":
    env = Environment()
    env.step(0)
    env.step(5)
    env.step(1)
    env.step(5)
    env.step(2)
    env.step(5)
    env.step(3)
