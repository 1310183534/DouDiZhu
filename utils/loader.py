import sys
import os
import re

if re.search('utils$', os.getcwd()):
	os.chdir(re.sub('utils$', '', os.getcwd()))
	sys.path.append('.')

import pickle
import torch
import numpy as np

from config import config
from game import GameEnv, Game
import os
from game import Action


class Data:
	def __init__(self, path, max_size=0):
		self.path = path
		self.max_size = max_size
		self.files_set = set()
		self.data = []
		self.len = 0
		self.iter = None
		self.reload()

	def __len__(self):
		return self.len

	def __call__(self):
		if self.empty():
			return None

		file, init, player, p, v = self.data[self.iter]
		env = GameEnv(False)
		env.load(init)
		game = Game(env, player)

		self.iter += 1
		return game, p, v
	
	def empty(self):
		return self.iter is None or self.iter >= len(self)

	def reload(self):
		files = os.listdir(self.path)
		files.sort(key=lambda _file: _file.split('_')[0], reverse=True)

		for file in files:
			if file in self.files_set:
				continue
			with open(self.path + '/' + file, 'rb') as f:
				data = pickle.load(f)
				
				init0, p0, v0 = data[0]
				init1, p1, v1 = data[1]
				init2, p2, v2 = data[2]
				
				self.data.append((file, init0, 0, p0, v0))
				self.data.append((file, init1, 1, p1, v1))
				self.data.append((file, init2, 2, p2, v2))
		self.files_set = set(files[:self.max_size])
		for file in files:
			if file not in self.files_set:
				os.remove(self.path + '/' + file)
		delete_idx = []
		for idx, _data in enumerate(self.data):
			file, init, player, p, v = _data
			if file not in self.files_set:
				delete_idx.append(idx)
		for idx in delete_idx[::-1]:
			self.data.pop(idx)
		print('Reload %s %d' % (self.path, len(self.files_set)))

	def init(self):
		np.random.shuffle(self.data)
		self.len = len(self.data)
		self.iter = 0
		print('Init %s %d' % (self.path, self.len))


class Loader:
	def __init__(self, batch_size, pool_size, bptt, data):
		self.batch_size = batch_size
		self.pool_size = pool_size

		self.bptt_min, self.bptt_max = bptt
		self.delta = self.bptt_max - self.bptt_min + 1

		self.data = data
		self.data.init()

		self.games = []
		self.vs_list = []
		self.ps_list = []

		self.deep = 0
		self.lengths = np.zeros(pool_size, dtype=np.long)
		self.histories = np.zeros((pool_size, self.bptt_max, 18, 4, 15), dtype=np.int16)
		self.rnn_states = np.zeros((pool_size, config.Rnn_channel, 4, 15), dtype=np.float32)
		self.resets = np.zeros((pool_size, self.bptt_max), dtype=np.bool)

		for _ in range(self.pool_size):
			self.games.append(None)
			self.vs_list.append(None)
			self.ps_list.append([])

		self.batch = np.arange(batch_size)

	def sample(self):
		perm = np.random.permutation(self.pool_size)
		self.batch = perm[:self.batch_size]

	def update(self, calc_rnn_states):
		states = np.zeros((self.pool_size, 18, 4, 15), dtype=np.int16)
		for i in range(self.pool_size):
			if self.games[i] is not None:
				self.lengths[i] += 1
				states[i] = self.games[i].to_model_input()
		self.deep += 1

		if self.deep > self.bptt_max:
			_mask = self.lengths != -1
			_histories = torch.tensor(self.histories[_mask, :self.delta]).type(torch.float32)
			_lengths = torch.ones(_mask.sum().item(), dtype=torch.long) * self.delta
			_rnn_states = torch.tensor(self.rnn_states[_mask])
			_resets = torch.tensor(self.resets[_mask], dtype=torch.float32)
			_rnn_states = calc_rnn_states(_histories, _lengths, _rnn_states, _resets)
			self.rnn_states[_mask] = _rnn_states.cpu().numpy()
			self.histories[_mask, :-self.delta] = self.histories[_mask, self.delta:]
			self.resets[_mask, :-self.delta] = self.resets[_mask, self.delta:]
			self.deep = self.bptt_min
		self.histories[:, self.deep - 1] = states
		self.resets[:, self.deep - 1] = False

	def __call__(self, fn):
		vs_label = np.zeros(self.batch_size, dtype=np.float32)
		ps_label = np.zeros(self.batch_size, dtype=np.long)
		for i, idx in enumerate(self.batch):
			if self.games[idx] is not None:
				vs_label[i] = self.vs_list[idx]
				ps_label[i] = self.ps_list[idx][self.lengths[idx] - 1]
		_mask = self.lengths[self.batch] != -1
		_histories = torch.tensor(self.histories[self.batch][_mask], dtype=torch.float32)
		_lengths = torch.ones(_mask.sum().item(), dtype=torch.long) * self.deep
		_rnn_states = torch.tensor(self.rnn_states[self.batch][_mask])
		_resets = torch.tensor(self.resets[self.batch][_mask], dtype=torch.float32)
		_vs_label = torch.tensor(vs_label[_mask])
		_ps_label = torch.tensor(ps_label[_mask])
		return fn(_histories, _lengths, _rnn_states, _resets, _vs_label, _ps_label)

	def next(self, training=False):
		new_epoch = False
		for i in range(self.pool_size):
			if self.lengths[i] == -1:
				assert not training
				continue
			if self.lengths[i] == len(self.ps_list[i]):
				if self.ps_list[i]:
					self.resets[i, self.deep - 1] = True
				self.lengths[i] = 0
				if self.data.empty():
					if training:
						new_epoch = True
						self.data.reload()
						self.data.init()
					else:
						self.games[i] = None
						self.vs_list[i] = None
						self.ps_list[i] = []
						self.lengths[i] = -1
						continue
				self.games[i], self.ps_list[i], self.vs_list[i] = self.data()
			else:
				self.resets[i, self.deep - 1] = False
				self.games[i].move(self.ps_list[i][self.lengths[i] - 1])
		return new_epoch

	def remain(self):
		remain = 0
		for game in self.games:
			if game is not None:
				remain += 1
		return remain


def main():
	path = '../gen/test'
	files = os.listdir(path)
	for file in files:
		with open(path + '/' + file, 'rb') as f:
			try:
				pickle.load(f)
			except EOFError:
				print(path + '/' + file)
				os.remove(path + '/' + file)
	
	# data = Data('../gen/train', max_size=300000)
	# loader = Loader(6000, 6000, (1, 8), data)

	# cnt = 0
	# while True:
	# 	loader.next(training=True)
	# 	cnt += 1
	# 	for i in range(loader.pool_size):
	# 		if loader.games[i] is not None:
	# 			loader.lengths[i] += 1
	# 	print(loader.games[0], loader.ps_list[0][loader.lengths[0] - 1])
	# 	print(cnt, loader.remain())
	# 	if not loader.remain():
	# 		break
	pass


if __name__ == '__main__':
	main()
