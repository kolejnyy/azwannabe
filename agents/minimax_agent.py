from agents import Agent
from games import Game
from searches.minimax import MiniMax
from copy import deepcopy

class MiniMaxAgent(Agent):
	def __init__(self, game : Game, heuristic_function, depth, decay=0.999):
		super().__init__(game)
		self.name = 'MiniMax Agent'
		self.heuristics = heuristic_function
		self.depth = depth
		self.decay = decay
		self.minimax = MiniMax(self.game, self.heuristics, self.depth, self.decay)

		self.cached_state 	= None
		self.cached_action 	= None
		self.cached_eval	= None

	def play(self, state):
		if self.game.equal_state(self.cached_state,state):
			return self.cached_action
		eval, action = self.minimax.run(state)
		self.cached_state = deepcopy(state)
		self.cached_action = action
		self.cached_eval = eval
		return action
	
	def eval(self, state):
		if  self.game.equal_state(self.cached_state,state):
			return self.cached_eval
		eval, action = self.minimax.run(state)
		self.cached_state = deepcopy(state)
		self.cached_action = action
		self.cached_eval = eval
		return eval