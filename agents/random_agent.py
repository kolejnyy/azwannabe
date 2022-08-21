from agents import Agent
from games import Game
from random import choice

class RandomAgent(Agent):
	def __init__(self, game : Game):
		super().__init__(game)
		self.name = "Random Agent"
	
	def play(self, state):
		possible_actions = self.game.possible_actions(state)
		return choice(possible_actions)

	def eval(self, state):
		return 0