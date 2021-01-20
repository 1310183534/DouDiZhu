import torch
import torch.nn as nn
from .cnn import ConvBlock, ResidualBlock, ReLU_layer
from .rnn import RnnLayer
from config import config


class VHead(nn.Module):
	def __init__(self):
		super(VHead, self).__init__()
		self.conv = ConvBlock(config.Residual_channel, 6, kernel_size=1)
		self.fc0 = nn.Linear(60 * 6, 256, bias=True)
		self.ReLU = ReLU_layer()
		self.dropout = nn.Dropout(p=0.5)
		self.fc1 = nn.Linear(256, 1, bias=False)

	def forward(self, body_outputs):
		v = self.conv(body_outputs)
		v = torch.flatten(v, start_dim=1)
		v = self.fc0(v)
		v = self.ReLU(v)
		v = self.dropout(v)
		v = self.fc1(v)
		return torch.tanh(v)


class PHead(nn.Module):
	def __init__(self):
		super(PHead, self).__init__()
		self.conv = ConvBlock(config.Residual_channel, 12, kernel_size=1)
		self.fc = nn.Linear(60 * 12, 356)

	def forward(self, ResNet_outputs):
		p = self.conv(ResNet_outputs)
		p = torch.flatten(p, 1)
		p = self.fc(p)
		return p


class Struct(nn.Module):
	def __init__(self):
		super(Struct, self).__init__()
		self.Rnn = RnnLayer()
		self.ResNet = nn.Sequential()
		self.ResNet.add_module('Conv', ConvBlock(18 + (config.Rnn_channel >> 1), config.Residual_channel))
		for layer_id in range(config.Residual_num):
			self.ResNet.add_module('ResBlock_%d' % layer_id, ResidualBlock(config.Residual_channel))
		self.vhead = VHead()
		self.phead = PHead()

	def calc_rnn_states(self, histories, lengths, rnn_states, resets):
		rnn_outputs, rnn_states = self.Rnn(histories, lengths, rnn_states, resets)
		return rnn_states

	def forward(self, histories, lengths, rnn_states, resets):
		states = histories[torch.arange(histories.shape[0]), lengths - 1].clone()
		histories_encode, rnn_states = self.Rnn(histories, lengths, rnn_states, resets)
		body_outputs = self.ResNet(torch.cat((histories_encode, states), dim=1))
		return self.vhead(body_outputs), self.phead(body_outputs), rnn_states
