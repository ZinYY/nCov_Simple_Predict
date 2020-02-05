import torch
import torch.nn as nn
import torch.nn.functional as F


class SNet(nn.Module):
	def __init__(self):
		super(SNet, self).__init__()
		# self.timebias = nn.Parameter(torch.randn(1), requires_grad=True)
		
		self.linear1 = nn.Linear(1, 1)
		
		self.topnum = nn.Parameter(torch.randn(1), requires_grad=True)
		self.addnum = nn.Parameter(torch.randn(1), requires_grad=True)
	
	# self.bias = nn.Parameter(torch.randn(1) + 1, requires_grad=True)
	
	def forward(self, x):
		x = self.linear1(x)
		x = self.topnum / (self.addnum + torch.exp(x))
		
		# x = self.topnum / (self.addnum + torch.exp(-(x - self.timebias)))
		return x
