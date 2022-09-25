import torch
from torch import functional as F, nn, optim
import numpy as np


class Amphisbaena(nn.Module):

	def __init__(self, f_extractor, heads) -> None:
		super().__init__()
		# The first part of the network
		self.feature_extractor = f_extractor
		# Head
		self.head = heads

	# For a given state x, return value and prior distribution of moves
	def forward(self, x):
		# Run the state through the feature extractor
		x = self.feature_extractor(x)
		x = x.flatten(1)
		# and through both heads
		value, priors = self.head(x)

		return value, priors