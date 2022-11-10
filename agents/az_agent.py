from agents import Agent
from games import Game
from searches import MCTS
import torch
from torch import functional as F, nn, optim
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

	def training_playout(self):
		state = self.game.initial_state()
		history = []
		player = 1
		score = None

		while 1:
			# Evaluate the possible actions and move probabilities
			poss_actions = self.game.possible_actions(state)
			probs, _ = self.mcts.run_playouts(state, self.num_playouts)
			# and choose the move according to the obtained distribution
			action = np.random.choice(poss_actions, p=probs)
			history.append([state, probs, player])
			
			# Check if the action ends the game
			action_eval = self.game.evaluate(state, action)
			if action_eval != None:
				score = player*action_eval
				break
			
			# Otherwise, proceed to next state
			state, swap = self.game.next_state(state, action)
			player = -player if swap else player
			if swap: state = self.game.swap_perspective(state)

		for i in range(len(history)):
			history[i][2] *= score

		return history

	def train(self, epochs = 1000, lr = 0.001):
		# Initialize optimizers and loss functions
		optimizer = optim.Adam(self.model.parameters(), lr, weight_decay=1e-5)
		mse_crit  = nn.MSELoss()
		softmax	  = nn.Softmax()

		# Run the training loop
		for i in range(epochs):
			history = self.training_playout()
			
			loss = torch.tensor([0.0], requires_grad=True)

			debug = True
			for state, probs, val in history:
				pred_val, priors = self.run_prediction(state)
				poss_actions = self.game.possible_actions(state)
				priors = priors[0][poss_actions]
				priors = priors / torch.sum(priors)
				if debug:
					print(val, priors.data)
					debug = False
				loss = loss + (pred_val-val)**2 - torch.dot(torch.from_numpy(probs).float(), torch.log(priors))

			print("Epoch {}:   loss = {:.5f}".format(i+1, loss.item()))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()


