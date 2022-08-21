from agents.agent import Agent
from games.game import Game
from match.match import Match
import numpy as np
from os import system


class Tournament():

	def __init__(self, game : Game, players, matches_per_side : int) -> None:
		
		self.game = game
		self.players = players
		self.matches = matches_per_side
		self.num_players = len(players)

	
	def match_result(self, match):
		result = match.play()
		if result > 0:
			return np.array([1,0,0])
		if result == 0:
			return np.array([0,1,0])
		return np.array([0,0,1])

	def run_tournament(self):
		
		# Print the list of players 
		system('cls')
		print("Game:   {}\n".format(self.game.name))
		print("Players:")
		for i, agent in enumerate(self.players):
			print("Player {}: {}".format(i, agent.name))
		
		print('\n\n')

		for i in range(self.num_players):
			for j in range(i+1, self.num_players):
				print("Playing match:  Player {}  vs  Player {}".format(i,j))

				# Play match from one side
				match_1 = Match(self.game, self.players[i], self.players[j])
				result = np.zeros(3)
				for _ in range(self.matches):
					result += self.match_result(match_1)
				print("Side one: {} : {} : {}".format(result[0],result[1],result[2]))
				
				# Swap sides and play again
				match_2 = Match(self.game, self.players[j], self.players[i])
				result = np.zeros(3)
				for _ in range(self.matches):
					result += self.match_result(match_2)
				print("Side two: {} : {} : {}".format(result[2],result[1],result[0]))