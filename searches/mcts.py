from games import Game


# Class implementation of Monte Carlo Tree Search

class MCTS:
	
	# Initialize
	def __init__(self, game : Game, value_function, prior_function, lmbd : float = 0.5):
		# The game that is being played
		self.game = game
		# Heuristic evaluation function
		self.val_f = value_function
		# Heuristic prior distribution function
		self.prior_f = prior_function
		# Lambda parameter
		self.lmbd = lmbd
	
	def run_playouts(self, state, num_rollouts):
		pass


	def print_parameters(self):
		print(self.lmbd)