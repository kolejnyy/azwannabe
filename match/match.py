from time import time
from agents import Agent
from games import Game
import numpy as np



class Match:

	def __init__(self, game : Game, player_1 : Agent, player_2 : Agent):
		# Initialize the game
		self.game = game
		# Initialize the players
		self.player_1 = player_1
		self.player_2 = player_2

	def play(self, display = False):
		# Initialize the starting state and game conditions
		state = self.game.initial_state()
		side = 1
		# Print some stuff if needed
		if display:
			print("Playing match:  {}  vs  {}\n".format(self.player_1.name, self.player_2.name))

		while True:
			begin_time = time()
			# Choose an action depending on whose turn it is
			if side == 1:
				action = self.player_1.play(state)
			else:
				action = self.player_2.play(state)

			# Display if asked to
			if display:
				print("Current position:")
				self.game.print_board(state)
				print("Chosen moves: ", action)
				print("Round time:   ", time()-begin_time)
				print("\n")


			# Evaluate reward for the chosen move
			reward = self.game.evaluate(state, action) 
			if not reward is None:
				if display:
					print("Game result:   ", side*reward)
				return side*reward
			next_state, swap = self.game.next_state(state, action)
			if swap:
				side *= -1
				next_state = self.game.swap_perspective(next_state)
			state = next_state