import sys
import os
import re

sys.path.append('.')
if re.search('human', os.getcwd()):
	os.chdir(re.sub('human$', '', os.getcwd()))
	sys.path.append('.')

from game import GameEnv
from game import Game
from node import new_Root
from model import Model
from config import config
import argparse
from mcts import MCT
from agent import Agent
from abs1 import Agent as AgentD
from multiprocessing import Pipe
from threading import Thread
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=int, default=-1)
parser.add_argument('-p', '--player', type=int, default=0)
# parser.add_argument('-s', '--simulations', type=int, default=800)
# parser.add_argument('-n', '--name', type=str, default='model')
args = parser.parse_args()


def connect():
	def run(connection):
		while True:
			try:
				inputs = input()
			except EOFError:
				return

			connection.send(inputs)

	connection0, connection1 = Pipe()
	Thread(target=run, args=(connection0,)).start()
	return connection1


def game_start(agent, simulations, game, connection):
	root = new_Root()
	while not game.game_over():
		mct = MCT(game, root)
		if root.is_leaf():
			agent.simulates([mct], 'Always')
		period = game.period()
		if period == 1 or period == 2:
			if game.curr_player() == game.my_player():
				for _ in range(simulations):
					agent.simulates([mct], 'Always')
				# action = mct.choice(eta=1.0, t=1.0)
				action = mct.choice(eta=1.0, t=None)
				# mct.json('visual/mct_%d_%d.json' % (game.my_player(), time_step))
				game.move(action)
				root.root(action)
				print(action, flush=True)
			else:
				action = connection.recv()
				if action is None:
					continue
				action = int(action)
				game.move(action)
				root.root(action)
		if period == 3:
			response = connection.recv()
			if response is None:
				continue
			response = int(response)
			game.move(response)
			root.root(response)
		time.sleep(0.1)
	print(game.eval(), flush=True)
	root.delete_tree()


def deal_init(connection):
	init_str = None
	while init_str is None:
		init_str = connection.recv()
	init_str = init_str.replace('[', '').replace(']', '').replace(',', '')
	init = []
	for num_str in init_str.split(' '):
		try:
			init.append(int(num_str))
		except ValueError:
			pass
	return np.array(init, dtype=np.int32).reshape([3, 15])


def main():
	connection = connect()
	config.set_device(args.gpu)

	# while True:
	name = None
	while name is None:
		name = connection.recv()
	model = Model('model_' + name)
	model.forced_restore()
	# print(model)
	agent = Agent(model)

	simulations = None
	while simulations is None:
		simulations = connection.recv()
	simulations = int(simulations)
	config.simulations = simulations

	env = GameEnv()
	init = deal_init(connection)

	env.load(init)

	game = Game(env, args.player)
	game_start(agent, simulations, game, connection)


if __name__ == '__main__':
	main()
