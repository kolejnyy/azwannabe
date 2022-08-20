from games import Game
import numpy as np
from copy import deepcopy

class ToxicToads(Game):

	def __init__(self) -> None:
		super().__init__()
		self.name = "Toxic Toads"
		self.length = 10
		self.neighbours = [
			[1,5], [0,2,6], [1,3,7], [2,4,8], [3,9],
			[0,6,10], [1,5,7,11], [2,6,8,12], [3,7,9,13], [4,8,14],
			[5,11,15],[6,10,12,16],[7,11,13,17],[8,12,14,18],[9,13,19],
			[10,16,20],[11,15,17,21],[12,16,18,22],[13,17,19,23],[14,18,24],
			[15,12],[16,20,22],[17,21,23],[18,22,24],[19,23]
		]

	def initial_state(self):
		board = np.zeros(25).astype(np.int64)
		board[6:9]-=1
		board[16:19]+=1
		return (board, self.length)
	
	def possible_actions(self, state):
		board, _ = state
		return np.array([i for i in range(25) if board[i]==0 and np.max(board[self.neighbours[i]])==1])
	
	def next_state(self, state, action):
		new_state, moves = deepcopy(state)
		new_state[action] = 1
		new_state[self.neighbours[action]]=0
		return (new_state, moves-1), True

	def swap_perspective(self, state):
		board, moves = deepcopy(state)
		return (board*-1, moves)

	def evaluate(self, state, action):
		if np.min(self.next_state(state, action)[0][0])>-1:
			return 1
		if state[1]==1:
			return 0
		return None

	def print_board(self, state):
		board, moves = state
		print("Moves left:\t", moves)
		print("Position on board:\n", board.reshape(5,5))


	def draw_board(self, state):
		return super().draw_board(state)

	def equal_state(self, state1, state2):
		if state1 is None or state2 is None:
			return False
		return state1[1]==state2[1] and (state1[0]==state2[0]).all()

	def hashable_state(self, state):
		board, moves = deepcopy(state)
		board = tuple(board)
		return board+(moves,)