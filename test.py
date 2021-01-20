from model import Model
import numpy as np
from utils.utils import pickle_load
from utils import history_regularize, to_cuda, unpack
from game import Game, GameEnv
from config import config
import os
import torch


def main():
	np.set_printoptions(precision=2, linewidth=128, suppress=True)
	path = 'test'
	config.device = torch.device('cuda:1')
	config.set_device(-1)
	model = Model('model_201912080009')
	model.restore()

	files = os.listdir('gen/%s/' % path)
	files.sort(key=lambda k: k.split('_')[0], reverse=True)
	# print(files[-1])
	file = np.random.choice(files[:100])
	# file = '201912101326_85d94af6fe1a588b.pkl'

	print(file)
	data = pickle_load('gen/%s/%s' % (path, file))
	# data = pickle_load('gen/test/' + files[-2])
	# np.random.shuffle(data)

	player = 2

	init, actions, _v = [None,
	                     [], -1.0]

	init = [[2, 1, 2, 1, 0, 1, 3, 1, 2, 1, 0, 1, 2, 0, 0],
	        [2, 1, 1, 2, 1, 1, 0, 3, 1, 2, 0, 2, 0, 0, 1],
	        [0, 1, 1, 1, 3, 2, 0, 0, 1, 1, 3, 1, 2, 1, 0]]
	actions = [352, 352, 353, 338, 343, 347, 123, 0, 0, 20, 22, 23, 24, 26, 0, 28, 0, 29, 0, 0, 39, 0, 0, 116, 0, 0, 76,
	           324, 0, 0, 41, 42, 0, 0, 92, 317, 320, 0, 0, 31, 42, 0, 0, 15, 18]
	init = np.array(init, dtype=np.int32)

	# actions = [353, 352, 352, 339, 349, 349, 15]

	# init = [[2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0],
	# 		[2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0],
	# 		[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 4, 4, 4, 1, 1]]
	# init = np.array(init, dtype=np.int32)
	# actions = [353, 352, 352, 344, 345, 346, 151]

	# init, actions, _v = data[player]
	print(_v, player)
	print(init)
	print(actions)
	print('=============================================')
	print('player:', player)

	histories = unpack(init, actions, player)
	histories, lengths = history_regularize(histories)
	histories, lengths = to_cuda(histories, lengths)

	vs, ps, _ = model(histories, lengths, None)

	env = GameEnv(False)
	env.load(init)
	game = Game(env, player)
	for v, p, action in zip(vs, ps, actions):
		print('----------------------------------')
		print('my_player: %d, curr_player: %d' % (player, game.curr_player()))
		# for action, _dist in enumerate(dist):
		# 	print(action, _dist)
		idx = np.argsort(p)[::-1]
		for i in range(8):
			print(game.could_move(idx[i]), end=' ')
			print('(%d, %.2f%%)' % (idx[i], p[idx[i]] * 100), end='\n')
		print('action: %d, %.2f%%' % (action, p[action] * 100))

		if idx[0] == 60 and p[idx[0]] > 0.3:
			print(game)
			print(game.policy())
			print(game.hand_cards_num())
			print(game.bottom())
			print(game.curr_player(), game.lord_player())
			return 0

		# model_input = game.to_model_input()
		# for channel in range(26, 28):
		# 	print(channel)
		# 	print(model_input[channel])
		print('%.1f, %.3f' % (_v, v[0]))

		game.move(action)
		print(game)
		print('Gauss:', game.gauss())


if __name__ == '__main__':
	main()
