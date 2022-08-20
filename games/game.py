
class Game:

	def __init__(self) -> None:
		self.name = 'Game'
	
	def initial_state(self):
		raise Exception("Initial state not defined!")

	def possible_actions(self, state):
		raise Exception("Possible actions function not defined!")
	
	def next_state(self, state, action):
		raise Exception("The transition function not defined!")

	def swap_perspective(self, state):
		raise Exception("Perspective swapping not defined!")

	def evaluate(self, state, action):
		raise Exception("Action on state evaluation not declared!")

	def draw_board(self, state):
		raise Exception("Drawing board state not defined!")

	def print_board(self, state):
		print(state)

	def equal_state(self, state1, state2):
		raise Exception("State comparison not implemented!")