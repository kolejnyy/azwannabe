from copy import deepcopy
import torch
from torch import functional as F, nn

import numpy as np



class ResBlock_Classic(nn.Module):

	def __init__(self, in_channels, out_channels) -> None:
		super().__init__()

		if out_channels % in_channels != 0:
			raise Exception("ResBlock Error: number of output channels {} not divisible by the number of input channels {}!".format(out_channels, in_channels))

		# Convolitional layers
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=1)

		# Batch normalizations
		self.batch_norm1 = torch.nn.BatchNorm2d(out_channels)
		self.batch_norm2 = torch.nn.BatchNorm2d(out_channels)

		# ReLU's
		self.relu1 = nn.ReLU()
		self.relu2 = nn.ReLU()

	def forward(self, x : torch.Tensor):
		residue = x.clone()
		x = self.conv1(x)
		x = self.batch_norm1(x)
		x = self.relu1(x)
		x = self.conv2(x)
		x = self.batch_norm2(x)
		
		# If the number of channels has changes, expand the residue
		if residue.shape[1] != x.shape[1]:
			residue = residue.repeat(1,x.shape[1]//residue.shape[1],1,1)
		return self.relu2(residue+x)