from turtle import forward
import torch
from torch import functional as F, nn, optim
import numpy as np


class SimpleHeads(nn.Module):

	def __init__(self, in_channels, out_moves) -> None:
		super().__init__()
		# Define heads
		self.value_head 	= nn.Linear(in_channels, 1)
		self.policy_head 	= nn.Linear(in_channels, out_moves)

		self.softmax		= nn.Softmax(dim=1)

	def forward(self, x):
		value 	= self.value_head(x)
		priors	= self.softmax(self.policy_head(x))
		return value, priors