import sys
import os
import re

if re.search('gen$', os.getcwd()):
	sys.path.append('..')

from game import Game
from game import GameEnv
import numpy as np
from utils import dump
import os
import shutil


path = 'train'

for cnt in range(30):
	env = GameEnv()
	games = [Game(env, player) for player in range(3)]
	datas = [[env.hand_cards(), None, None] for _ in range(3)]
	while not env.game_over():
		period = env.period()
		if period != 3:
			curr_player = env.curr_player()
			game = games[curr_player]
			action = np.random.choice(game.action_list())
			for game in games:
				game.move(action)
			env.move(action)
		else:
			for player, game in enumerate(games):
				response = env.response(player)
				game.move(response)
			env.move(-1)
	for data, game in zip(datas, games):
		data[1] = game.policy()
		data[2] = game.eval()
	dump(datas, path)

files = os.listdir(path)

for file in files:
	shutil.move(path + '/' + file, path + '/' + '000000000000_' + file)