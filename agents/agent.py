from games import Game

class Agent:

	def __init__(self, game : Game):
		self.name = 'Agent'
		self.game = game

	def play(self, state):
		raise Exception("Move choosing not implemented!")

	def eval(self, state):
		raise Exception("Evaluation of position not implemented!")