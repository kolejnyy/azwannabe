from os import system
from agents import Agent
from games import Game
from sys import platform

class SelfPlayAgent(Agent):

	def __init__(self, game : Game):
		super().__init__(game)
		self.name = 'Self-Play Agent'

	def play(self, state):
		if platform == 'win32':
			system('cls')
		else:
			system('clear')
		self.game.print_board(state)
		print("\nPossible actions:")
		poss_moves = self.game.possible_actions(state)
		print(" ".join([str(x) for x in poss_moves]), '\n')
		while True:
			print("Your move is:")
			move = input()
			try:
				move = int(move)
			except Exception:
				print("Invalid move!")
				pass
			else:
				if move in poss_moves:
					return move
				else:
					print("Invalid move!")

		