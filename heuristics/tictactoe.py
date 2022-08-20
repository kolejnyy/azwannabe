import numpy as np

pairs = [	[0,1],[0,4],[0,3],[1,3],[1,4],[1,5],[1,2],[2,4],[2,5],
			[3,4],[3,7],[3,6],[4,6],[4,7],[4,8],[4,5],[5,7],[5,8],
			[6,7],[7,8]			
		]

lines = [	[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[1,4,8],[2,4,6] ]

def lineHeuristic(state):
	res = 0
	for [i,j] in pairs:
		if state[i]==state[j]:
			res += state[i]
	for line in lines:
		if np.min(state[line])==np.max(state[line]):
			res += state[line[0]]*100
	return np.tanh(res/5)