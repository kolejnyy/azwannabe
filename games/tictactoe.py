from games import Game
import numpy as np
from copy import deepcopy

class TicTacToe(Game):

	def __init__(self) -> None:
		super().__init__()
		self.name = 'TicTacToe'
		self.lines = [
			[[1,2],[3,6],[4,8]],
			[[0,2],[4,7]],
			[[0,1],[5,8],[4,6]],
			[[0,6],[4,5]],
			[[0,8],[1,7],[2,6],[3,5]],
			[[3,4],[2,8]],
			[[0,3],[7,8],[2,4]],
			[[1,4],[6,8]],
			[[0,4],[2,5],[6,7]]
		]

	def initial_state(self):
		return np.array([0]*9)

	def possible_actions(self, state):
		return np.array([0,1,2,3,4,5,6,7,8])[state==0]

	def next_state(self, state, action):
		if state[action]!=0:
			raise Exception('Invalid move!')
		new_state = deepcopy(state)
		new_state[action] = 1
		return new_state, True

	def swap_perspective(self, state):
		return state*(-1)

	def evaluate(self, state, action):
		if state[action]!=0:
			raise Exception('Invalid action to evaluate: {} in {}'.format(action, state))
		for line in self.lines[action]:
			if state[line].min()==1:
				return 1
		if np.sum(state==0)==1:
			return 0
		return None

	def print_board(self, state):
		char_list = np.array(["   "]*9)
		char_list[state==1] = ' X '
		char_list[state==-1] = ' O '
		print('{}|{}|{}\n---+---+---\n{}|{}|{}\n---+---+---\n{}|{}|{}'.format(
			char_list[0],char_list[1],char_list[2],
			char_list[3],char_list[4],char_list[5],
			char_list[6],char_list[7],char_list[8]
		))

	def equal_state(self, state1, state2):
		return (state1==state2).all()

	def hashable_state(self, state):
		return tuple(state)