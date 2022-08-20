from games import Game
from math import sqrt
import numpy as np
from copy import deepcopy

# Class implementation of Monte Carlo Tree Search

class MCTS:
	
	# Initialize
	def __init__(self, game : Game, value_function, prior_function, lmbd : float = 0.5, c_puct : float = 0.1):
		# The game that is being played
		self.game = game
		# Heuristic evaluation function
		self.val_f = value_function
		# Heuristic prior distribution function
		self.prior_f = prior_function
		# Lambda parameter
		self.lmbd = lmbd
		# Exploration constant
		self.c_puct = c_puct
	
	def run_playouts(self, state, num_rollouts):
		# Initialize memory and lists for all variables
		memory 	= {}
		curr_idx = 0
		actions = []; priors = []
		N_v = []; N_r = []; N_sum = []
		W_v = []; W_r = []
		Q	= []

		def expand(_state, curr_idx):
			memory[self.game.hashable_state(_state)] = curr_idx
			poss_actions = list(self.game.possible_actions(_state))
			actions_num = len(poss_actions)
			actions.append(poss_actions)
			priors.append(np.array(self.prior_f(_state)))
			N_v.append(np.zeros(actions_num));N_r.append(np.zeros(actions_num));N_sum.append(0)
			W_v.append(np.zeros(actions_num));W_r.append(np.zeros(actions_num))
			Q.append(np.array([self.val_f(_state, action) for action in poss_actions]))

		def update(history, reward_r, reward_v):
			# update the variables basing on history and rewards 
			for (state_id, action_id, swap) in reversed(history):
				# swap rewards if needed
				if swap:
					reward_r *= -1
					if not reward_v is None:
						reward_v *= -1
				# update everything
				N_sum[state_id] += 1
				N_r[state_id][action_id] += 1
				W_r[state_id][action_id] += reward_r
				if not reward_v is None:
					N_v[state_id][action_id] += 1
					W_v[state_id][action_id] += reward_v
				Q[state_id][action_id] = W_r[state_id][action_id]/N_r[state_id][action_id]
				if N_v[state_id][action_id] != 0: 
					Q[state_id][action_id] = self.lmbd*Q[state_id][action_id] + (1 - self.lmbd)*W_v[state_id][action_id]/N_v[state_id][action_id]

		def rollout(state):
			# return the result of the rollout
			multiplier = 1
			curr_state = deepcopy(state)
			
			while True:
				poss_actions = self.game.possible_actions(curr_state)
				p_dist = np.array(self.prior_f(curr_state))
				move = poss_actions[np.argmax(p_dist)]
				# if the move gave a reward, return it
				_reward = self.game.evaluate(curr_state, move)
				if not _reward is None:
					return multiplier*_reward
				# otherwise, keep playing
				curr_state, _swap = self.game.next_state(curr_state, move)
				if _swap:
					multiplier *= -1  
				

		expand(state, curr_idx)
		curr_idx += 1

		for _ in range(num_rollouts):
			# Save history as list of tuples (state_id, action_id, swap)
			history = []
			curr_state = deepcopy(state)
			idx = 0
			while True:
				# Calculate the scores and find the best action
				scores = Q[idx]+self.c_puct*priors[idx]*sqrt(N_sum[idx])/(N_r[idx]+1)
				best_id   = np.argmax(scores)
				best_move = actions[idx][best_id]

				# Check if the action ends the game
				reward = self.game.evaluate(curr_state, best_move)
				if not reward is None:
					# if so, call the update immediately and break
					history.append((idx, best_id, False))
					update(history, reward, None)
					break

				# otherwise, check if the state has been visited
				next_state, swapped = self.game.next_state(curr_state, best_move)
				next_state = self.game.swap_perspective(next_state) if swapped else next_state
				next_idx = memory.get(self.game.hashable_state(next_state), -1)
				if next_idx != -1:
					# if it has not been visited, add the current state to history,
					# overwrite with the next state and continue
					history.append((idx, best_id, swapped))
					curr_state = next_state
					idx = next_idx
					continue
				else:
					# otherwise, we reached a leaf state, so we run rollout
					# according to the current policy and save the value
					expand(next_state, curr_idx)
					curr_idx += 1
					reward = rollout(next_state)
					reward = -reward if swapped else reward
					history.append((idx, best_id, False))
					update(history, reward, self.val_f(curr_state, best_move))
					break
		
		return N_r[0]/N_sum[0]

	def print_parameters(self):
		print(self.lmbd)