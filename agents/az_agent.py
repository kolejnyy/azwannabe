from agents import Agent
from games import Game
from searches import MCTS
import torch
from torch import functional as F, nn
import numpy as np

class AZAgent(Agent):

	def __init__(self, game : Game, network_arch : nn.Module, prepare_f, num_playouts, lmbd : float = 0.5, c_puct : float = 0.1):
		super().__init__(game)
		self.name = "AZ Agent"

		# Neural network
		self.model = network_arch
		self.prepare_state = prepare_f

		# Search
		self.num_playouts = num_playouts
		self.mcts = MCTS(game, self.eval, self.prior, lmbd=lmbd, c_puct=c_puct)


	def load_model(self, path):
		self.model.load_state_dict(torch.load(path))

	def save_model(self, path):
		torch.save(self.model.state_dict(), path)

	def eval(self, state, action):
		n_state, swap = self.game.next_state(state, action)
		v, p = self.run_prediction(n_state, gradient = False)
		if swap:
			return -v[0]
		return v[0]
	
	def prior(self, state):
		v, p = self.run_prediction(state, gradient = False)
		return p[0][self.game.possible_actions(state)]

	def run_prediction(self, state, gradient = True):
		prepared_state = self.prepare_state(state)
		val = None
		priors = None
		if gradient:
			val, priors = self.model(prepared_state)
		else:
			with torch.no_grad():
				val, priors = self.model(prepared_state)
		return val, priors

	def play(self, state):
		poss_actions = self.game.possible_actions(state)
		probs, Q = self.mcts.run_playouts(state, self.num_playouts)
		return poss_actions[np.argmax(probs)]