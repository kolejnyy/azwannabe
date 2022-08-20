from ast import Num
from games import Game


class MiniMax():

	# Initialize the MiniMax class
	# 	Input:		game		a Game instance with the releavet game
	#				heuristics	a function state -> float/int returning heuristic evaluation
	#							of a position
	#				depth		the depth of the search
	#	
	#	(optional)	decay		the decay factor applied to reward after each move, encouraging
	#							faster wins
	def __init__(self, game : Game, heuristics, depth, decay = 0.999) -> None:
		# Set the game
		self.game = game
		# and the heuristic evaluation function
		self.eval = heuristics
		# and the depth of search
		self.depth = depth
		# and reward decay
		self.decay = decay

	# The iterative part of MiniMax, equipped with alpha-beta prunning
	def minimax_iter(self, state, alpha, beta, depth, debug = False):
		# Print debug notes is debug mode is on
		if debug:
			print("depth = {}\talpha = {}\t beta = {}\t state = {}".format(depth, alpha, beta, state))

		# If the depth is 0, return the evaluation
		if depth == 0:
			return self.eval(state), -1

		# Get the possible actions and set variable for the best action
		poss_actions 	= self.game.possible_actions(state)
		best_action 	= -1
		evaluation 		= -1e9

		# Iterate over all possible actions
		for action in poss_actions:
			# If the action ends the game, get the reward
			reward = self.game.evaluate(state, action)
			# otherwise
			if reward is None:
				# Evaluate the next state and player
				next_state, player_swap = self.game.next_state(state, action)
				if player_swap:
					next_state =  self.game.swap_perspective(next_state)
					reward, _ = self.minimax_iter(next_state, -beta, -alpha, depth-1, debug)
					reward *= -1
				else:
					reward, _ = self.decay*self.minimax_iter(next_state, alpha, beta, depth-1, debug)
				reward *= self.decay
			# Pruning time
			if beta <= reward:
				if debug:
					print("Pruned action {} for state {}".format(action, state))
				return reward, action
			alpha = max(alpha, reward)
			
			# Update variables
			if reward > evaluation:
				evaluation = reward
				best_action = action
		
		return evaluation, best_action


	# Evaluate the given state using MiniMax algorithm
	#	Input:		state
	#	Output:		val			- evaluation of the given state
	# 				best_action	- the action that achieves val		
	def run(self, state, debug = False):
		return self.minimax_iter(state, -1e9, 1e9, self.depth, debug=debug)
