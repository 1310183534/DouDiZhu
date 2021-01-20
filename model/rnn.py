import torch
import torch.nn as nn
from config import config
from model.cnn import conv_layer, ReLU_layer, ResidualBlock
import torch.nn.utils as nn_utils


class RnnConvBlock(nn.Module):
	def __init__(self, input_channel, output_channel, kernel_size=3):
		super(RnnConvBlock, self).__init__()
		self.conv = conv_layer(input_channel, output_channel, kernel_size, bias=True)
		self.ReLU = ReLU_layer()

	def forward(self, x):
		out = self.conv(x)
		out = self.ReLU(out)
		return out


class Extractor(nn.Module):
	def __init__(self):
		super(Extractor, self).__init__()
		self.ResNet = nn.Sequential()
		self.ResNet.add_module('Conv', RnnConvBlock(18 + (config.Rnn_channel >> 1), config.Extractor_channel))
		for layer_id in range(config.Extractor_num):
			self.ResNet.add_module('ResBlock_%d' % layer_id, ResidualBlock(config.Extractor_channel))

	def forward(self, inputs):
		outputs = self.ResNet(inputs)
		return outputs


class RnnCell(nn.Module):
	def __init__(self):
		super(RnnCell, self).__init__()
		self.extractor = Extractor()
		self.gate = conv_layer(config.Extractor_channel + (config.Rnn_channel >> 1), (config.Rnn_channel >> 1) * 4, kernel_size=1, bias=True)

	@staticmethod
	def debug(h, c, f, i, j, o, channel):
		print('====================================')
		print('h', h[0, channel])
		print('c', c[0, channel])
		print('f', torch.sigmoid(f).numpy()[0, channel])
		print('i', torch.sigmoid(i).numpy()[0, channel])
		print('j', torch.tanh(j).numpy()[0, channel])
		print('o', torch.sigmoid(o).numpy()[0, channel])

	def forward(self, inputs, rnn_states):
		h, c = torch.chunk(rnn_states, 2, dim=1)
		inputs = torch.cat([h, inputs], dim=1)
		hidden = self.extractor(inputs)
		hidden = torch.cat([h, hidden], dim=1)
		gate = self.gate(hidden)

		f, i, j, o = torch.chunk(gate, 4, dim=1)

		c = c * torch.sigmoid(f) + torch.sigmoid(i) * torch.tanh(j)
		h = torch.tanh(c) * torch.sigmoid(o)
		# self.debug(h, c, f, i, j, o, 0)
		rnn_states = torch.cat([h, c], dim=1)
		return h, rnn_states


class RnnLayer(nn.Module):
	def __init__(self):
		super(RnnLayer, self).__init__()
		self.rnn = RnnCell()

	def forward(self, inputs, lengths, rnn_states, resets):
		batch, max_length, channel, w, h = inputs.shape
		if rnn_states is None:
			rnn_states = torch.zeros([batch, config.Rnn_channel, w, h], dtype=inputs.dtype, device=inputs.device)
		if resets is None:
			resets = torch.zeros([batch, max_length], dtype=inputs.dtype, device=inputs.device)
		rnn_outputs = torch.zeros([batch, config.Rnn_channel >> 1, w, h], dtype=inputs.dtype, device=inputs.device)

		lengths, idx = lengths.sort(descending=True)
		inputs, rnn_states, resets = inputs[idx], rnn_states[idx], resets[idx]

		inputs, nums = nn_utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True)[:2]

		tot = 0
		for step, num in enumerate(nums):
			_rnn_outputs, _rnn_states = self.rnn(inputs[tot: tot + num], rnn_states[:num])
			rnn_outputs = rnn_outputs.clone()
			rnn_states = rnn_states.clone()
			# print(_rnn_outputs.shape, resets[:num, step].shape)
			rnn_outputs[:num] = _rnn_outputs * (1 - resets[:num, step].view(-1, 1, 1, 1))
			rnn_states[:num] = _rnn_states * (1 - resets[:num, step].view(-1, 1, 1, 1))
			tot += num

		_, idx = idx.sort()
		return rnn_outputs[idx], rnn_states[idx]
