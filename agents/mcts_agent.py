from copy import deepcopy
from agents import Agent
from games import Game
from searches import MCTS
import numpy as np

class MCTSAgent(Agent):
	def __init__(self, game : Game, value_f, prior_f, num_playouts = 50, lmbd = 0.5, c_puct = 0.1):
		self.name = 'MCTS Agent'
		self.game = game
		self.num_playouts = num_playouts
		self.mcts = MCTS(game, value_f, prior_f, lmbd, c_puct)

		# Cached variables to speed up processing
		self.cached_state 	= None
		self.cached_Q		= None
		self.cached_move	= None

	def eval(self, state):
		# If we call for the cached state, retunrn immediately
		if self.game.equal_state(self.cached_state, state):
			return np.max(self.cached_Q)
		# otherwise, run MCTS
		probs, Q = self.mcts.run_playouts(state, self.num_playouts)
		self.cached_state = deepcopy(state)
		self.cached_Q = Q
		self.cached_probs = probs
		return np.max(self.cached_Q)

	def play(self, state):
		# If we call for the cached state, retunrn immediately
		if self.game.equal_state(self.cached_state, state):
			return self.cached_move
		# otherwise, run MCTS
		poss_actions = self.game.possible_actions(state)
		probs, Q = self.mcts.run_playouts(state, self.num_playouts)
		self.cached_state = deepcopy(state)
		self.cached_Q = Q
		self.cached_move = poss_actions[np.argmax(probs)]
		return self.cached_move