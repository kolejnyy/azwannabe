from turtle import forward
import torch
from torch import functional as F, nn, optim
import numpy as np

from .resblocks import ResBlock_Classic


class SimpleTrippleLayer(nn.Module):

	def __init__(self, in1, out1, out2, out3, pool1 = False, pool2 = False) -> None:
		super().__init__()
		self.conv1 	= nn.Conv2d(in1, out1, 3, padding=1)
		self.conv2 	= nn.Conv2d(out1, out2, 3, padding=1)
		self.conv3 	= nn.Conv2d(out2, out3, 3, padding=1)

		self.pool  	= nn.MaxPool2d(2)
		self.pool1 	= pool1
		self.pool2 	= pool2

		self.relu	= nn.ReLU()

	def forward(self, x):
		x = self.relu(self.conv1(x))
		if self.pool1:
			x = nn.MaxPool2d(x)
		x = self.relu(self.conv2(x))
		if self.pool1:
			x = nn.MaxPool2d(x)
		x = self.conv3(x)

		return x



class DoubleResBlock(nn.Module):

	def __init__(self, in_channels, res1_in, res2_in, out_channels) -> None:
		super().__init__()

		self.conv 		= nn.Conv2d(in_channels, res1_in, 3, padding=1)
		self.resblock1 	= ResBlock_Classic(res1_in, res2_in)
		self.resblock2	= ResBlock_Classic(res2_in, out_channels)

		self.relu 		= nn.ReLU()

	def forward(self, x):
		x = self.relu(self.conv(x))
		x = self.resblock1(x)
		x = self.resblock2(x)

		return x


class TrippleResBlock(nn.Module):

    def __init__(self, in_channels, res1_in, res2_in, res3_in, out_channels) -> None:
        super().__init__()

        self.conv 		= nn.Conv2d(in_channels, res1_in, 3, padding=1)
        self.resblock1 	= ResBlock_Classic(res1_in, res2_in)
        self.resblock2	= ResBlock_Classic(res2_in, res3_in)
        self.resblock3	= ResBlock_Classic(res3_in, out_channels)

        self.relu 		= nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        return x
