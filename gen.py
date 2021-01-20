from game import GameEnv
from game import Game
from model.model import Model
from agent import Agent
from mcts import MCT
from utils import dump
from node import new_Root
from multiprocessing import Process, current_process
from config import config
import os
import numpy as np
import time


def move(agent, envs, gamess, rootss):
	MCTss = [[] for player in range(3)]
	for games, roots in zip(gamess, rootss):
		for player, MCTs in enumerate(MCTss):
			MCTs.append(MCT(games[player], roots[player]))
	for MCTs in MCTss:  # to make sure all roots are not leaf
		if MCTs:
			agent.simulates(MCTs, store_rnn_states='Lottery')
			roots = [mct.root for mct in MCTs]
			agent.store_rnn_states(roots)

	MCTs = []
	for env, games, roots in zip(envs, gamess, rootss):
		if env.period() != 3:
			curr_player = env.curr_player()
			MCTs.append(MCT(games[curr_player], roots[curr_player]))

	if MCT:
		for simulate_cnt in range(config.simulations):
			agent.simulates(MCTs, store_rnn_states='Lottery')

	cnt = 0
	for env, games, roots in zip(envs, gamess, rootss):
		# print(env.period())
		period = env.period()
		if period == 1 or period == 2:
			mct = MCTs[cnt]
			cnt += 1
			action = mct.choice(eta=1.0, t=1.0)
			for player, (game, root) in enumerate(zip(games, roots)):
				game.move(action)
				root.root(action)
			env.move(action)
		elif period == 3:
			for player, (game, root) in enumerate(zip(games, roots)):
				response = env.response(player)
				game.move(response)
				root.root(response)
			env.move(-1)
		else:
			print('move() period=?')
			exit(-4)


def gen_init(datass, envs, gamess, rootss, number):
	while len(envs) < number:
		env = GameEnv()
		envs.append(env)
		datas = [[env.hand_cards(), None, None] for _ in range(3)]
		games = [Game(env, player) for player in range(3)]
		roots = [new_Root() for _ in range(3)]
		datass.append(datas)
		gamess.append(games)
		rootss.append(roots)


def gen(agent, path, number):
	print('GEN: %s %d %s' % (path, number, current_process().name))
	datass, envs, gamess, rootss = [], [], [], []
	generated = 0
	cnt = 0
	while True:
		if (cnt + 1) % 16 == 0:  # restore model every 10 steps
			try:
				agent.model.restore()
			except:
				print('could not restore model now.')

		gen_init(datass, envs, gamess, rootss, number)
		# print(len(envs[0].policy()))

		move(agent, envs, gamess, rootss)
		# print(envs[0])
		cnt += 1
		delete_idx = []

		for idx, env in enumerate(envs):
			if env.game_over():
				delete_idx.append(idx)
		for idx in delete_idx[::-1]:
			generated += 1
			for data, game in zip(datass[idx], gamess[idx]):
				data[1] = game.policy()
				data[2] = game.eval()
			dump(datass[idx], path)
			print('gen: %d %d %s' % (generated, len(os.listdir(path)), current_process().name))
			for root in rootss[idx]:
				root.delete_tree()
			for _list in [datass, envs, gamess, rootss]:
				_list.pop(idx)


def srand():
	_seed0 = int(time.time() * 12345233) % 998244353
	_seed1 = (current_process().pid * 87654233) % 998244353
	np.random.seed((_seed0 + _seed1) % 998244353)
	for i in range(current_process().pid):
		np.random.random()


def worker(device, path, number):
	config.set_device(device)
	model = Model('model')
	# model.restore()
	agent = Agent(model)
	srand()
	gen(agent, path, number)


def main():
	config.set_device_ids([0, 1, 2, 3, 4, 5, 6, 7])
	process_per_device = 4

	path = 'gen/data'
	if not os.path.exists(path):
		os.makedirs(path)

	number = 64

	processes = []
	for device in config.device_ids:
		for i in range(process_per_device):
			process = Process(target=worker, args=(device, path, number))
			processes.append(process)
	for process in processes:
		process.start()
	for process in processes:
		process.join()


if __name__ == '__main__':
	main()
