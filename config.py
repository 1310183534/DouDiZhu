import torch
import numpy as np

np.set_printoptions(linewidth=998, precision=3, suppress=True)
torch.set_printoptions(linewidth=998, precision=3)


class Config:
	def __init__(self):
		self.Rnn_channel = 256
		self.Extractor_num = 8
		self.Extractor_channel = 160
		self.Residual_num = 16
		self.Residual_channel = 256
		self.l2 = 1e-5
		# self.momentum = 0.9
		self.momentum = 0.0
		# self.device = torch.device('cuda:0')
		self.device = torch.device('cpu')
		self.device_ids = [3, 4]
		self.simulations = 800

		self.lottery_rate = 1.5
		self.name = 'model'

	def set_device(self, device):
		if device == -1:
			self.device = torch.device('cpu')
		else:
			self.device = torch.device('cuda:%d' % device)

	def set_device_ids(self, device_ids):
		self.device_ids = device_ids


config = Config()

if __name__ == '__main__':
	# print(config.get_device())
	print(config.device)
