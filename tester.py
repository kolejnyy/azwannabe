from random import choice
from time import time
from heuristics import lineHeuristic
from searches import MCTS, MiniMax
from games import TicTacToe, ToxicToads
from agents import *

import numpy as np

game =  ToxicToads()


func = lineHeuristic
print(func(np.array([0,0,0,0,0,0,0,0,0])))
print(func(np.array([1,1,-1,0,1,0,0,0,-1])))
print(func(np.array([0,0,1,0,1,0,1,-1,-1])))

mmagent = MiniMaxAgent(game, (lambda x : 0), 9)

b_t = time()
state = game.initial_state()
for i in range(10):
	action = mmagent.play(state)
	eval = mmagent.eval(state)
	print(action, eval)
	state = game.swap_perspective(game.next_state(state, action)[0])
print(time()-b_t)