from agents import Agent
from games import Game

class MCTSAgent(Agent):
	def __init__(self, game : Game):
		super().__init__(game)
