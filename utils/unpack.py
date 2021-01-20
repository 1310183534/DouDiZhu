import numpy as np
from game import Game, GameEnv
import torch
from config import config


def history_regularize(histories):
	lengths = np.array([history.shape[0] for history in histories])
	max_length = np.max(lengths)
	histories_array = np.zeros([len(histories), max_length, 18, 4, 15], dtype=np.int8)
	for i in range(len(histories)):
		history = histories[i]
		if lengths[i]:
			histories_array[i, :lengths[i]] = history
	return histories_array, lengths


def unpack(init, ps, player):
	env = GameEnv(False)
	env.load(init)
	game = Game(env, player)
	histories = []
	for action in ps:
		state = game.to_model_input()
		histories.append((histories[-1] if len(histories) else []) + [state])
		game.move(action)
	for i in range(len(histories)):
		histories[i] = np.array(histories[i], dtype=np.int8)
	return histories


# def to_cuda(histories, lengths, vs=None, ps=None):
# 	histories = histories.astype(np.float32)
# 	histories = torch.tensor(histories, device=config.device)
# 	lengths = torch.tensor(lengths, dtype=torch.long, device=config.device)
# 	if vs == ps is None:
# 		return histories, lengths
# 	vs = torch.tensor(vs, dtype=torch.float32, device=config.device)
# 	ps = torch.tensor(ps, dtype=torch.long, device=config.device)
# 	return histories, lengths, vs, ps


def to_cuda(histories, lengths):
	histories = histories.astype(np.float32)
	histories = torch.tensor(histories, device=config.device)
	lengths = torch.tensor(lengths, dtype=torch.long, device=config.device)
	return histories, lengths
