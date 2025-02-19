import pygame
import sys
import numpy as np
from utils import check_strike
from agent import Agent

BLOCK_SIZE = 70
NUM_COLUMNS = 7
NUM_ROWS = 6
COIN_STRIKE_TO_WIN = 4
SCREEN_WIDTH = BLOCK_SIZE * NUM_COLUMNS
SCREEN_HEIGHT = BLOCK_SIZE * NUM_ROWS

class Game:
	def __init__(self):
		pygame.init()
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
		self.clock = pygame.time.Clock()
		self.draw_grid()
		self.running = True
		self.board = [[0 for _ in range(NUM_COLUMNS)] for _ in range(NUM_ROWS)]
		self.coins_strike_quantity = COIN_STRIKE_TO_WIN
  
		# Agent configuration
		self.agent = Agent(np.array(self.board).shape, NUM_COLUMNS)
		self.agent.load_model("model.pth")

	# Run the game
	def run(self):
		# Main game loop
		while self.running:
			# Listen for user choosing column
			for event in pygame.event.get():
				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_1:
						self.place_coin(0, 1)
					elif event.key == pygame.K_2:
						self.place_coin(1, 1)
					elif event.key == pygame.K_3:
						self.place_coin(2, 1)
					elif event.key == pygame.K_4:
						self.place_coin(3, 1)
					elif event.key == pygame.K_5:
						self.place_coin(4, 1)
					elif event.key == pygame.K_6:
						self.place_coin(5, 1)
					elif event.key == pygame.K_7:
						self.place_coin(6, 1)
					# Quit
					elif event.key == pygame.K_q:
						sys.exit()
					else:
						continue

					# Make agent move
					# random_column = np.random.randint(0, NUM_COLUMNS)
					action = self.agent.choose_action(np.array(self.board).flatten())
					self.place_coin(action, 2)
     

			# After placing coin there is time to update pygame display
			self.update_display()

			# Check for win after move and display update
			print(self.check_for_win())
			if self.check_for_win():
				print("Someone won!")
				sys.exit()
    
	# Throwing coin
	def place_coin(self, column_index, player_id):
		# If coin is thrown into full column
		if self.board[0][column_index] != 0:
			sys.exit()

		# Finding first free space starting from the bottom
		for row_index in range(NUM_ROWS):
			if self.board[NUM_ROWS - row_index - 1][column_index] == 0:
				# Place coin in empty space
				self.board[NUM_ROWS - row_index - 1][column_index] = player_id
				break

	# Checking is any player won
	def check_for_win(self):
		for player_id in [1, 2]:
			if check_strike(self.board, player_id, self.coins_strike_quantity):
				return True
		return False
   

	# Drawing coins
	def draw_coins(self):
		for row_index in range(NUM_ROWS):
			for col_index in range(NUM_COLUMNS):
				current_space = self.board[row_index][col_index]

				if current_space != 0:
					# Create coin rectangle
					if current_space == 1:
						color = "green"
					else:
						color = "red"

					# Create coin object
					coin = pygame.Rect(col_index * BLOCK_SIZE, row_index * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
					
					# Place coin on the screen
					pygame.draw.rect(self.screen, color, coin)

	# Drawing grid lines
	def draw_grid(self):
		for y in range(0, SCREEN_HEIGHT, BLOCK_SIZE):
			for x in range(0, SCREEN_WIDTH, BLOCK_SIZE):
				box = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
				pygame.draw.rect(self.screen, "#333333", box, 1)

	# Updating display
	def update_display(self):
		# Painting screen with black
		self.screen.fill("black")

		# Drawing grid
		self.draw_grid()

		# Drawing placed coins
		self.draw_coins()

		# Updating display in pygame
		pygame.display.update()
	
game = Game()
game.run()