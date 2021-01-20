import sys
import os
import re

sys.path.append('.')
if re.search('human$', os.getcwd()):
	os.chdir(re.sub('human$', '', os.getcwd()))
	sys.path.append('.')

from subprocess import Popen, PIPE
from game import GameEnv, Action, Game
import time
import pickle
import numpy as np
from human.act import to_action, to_chars, card_dict
import getpass
import argparse
import readline
from utils import rand_x16

parser = argparse.ArgumentParser()
parser.add_argument('-s', type=int, default=1000)
parser.add_argument('-p', type=int, default=0)
parser.add_argument('-d', type=int, default=0)
args = parser.parse_args()

players_attr = (('201912080009', args.s) for _ in range(3))
info = (getpass.getuser(), args.p, args.s)
filename = '_' + rand_x16()
data = [info]


def match(AIs):
	def read(popen):
		while True:
			output = popen.stdout.readline()
			if output is not None:
				return output.decode('utf-8')

	def write(popen, obj):
		if popen == AIs[args.p]:
			return
		popen.stdin.write((str(obj) + '\n').encode('utf-8'))
		popen.stdin.flush()

	(name0, simulation0), (name1, simulation1), (name2, simulation2) = players_attr
	write(AIs[0], name0)
	write(AIs[0], simulation0)

	write(AIs[1], name1)
	write(AIs[1], simulation1)

	write(AIs[2], name2)
	write(AIs[2], simulation2)

	env = GameEnv()
	hand_cards = env.hand_cards()

	data.append(env.hand_cards())
	game = Game(env, args.p)

	for player in range(3):
		write(AIs[player], list(hand_cards.flatten()))
	slot = []
	ig = False

	print('Filename:', filename)
	print('Your position:', args.p)
	print('Your hand cards:', to_chars(hand_cards[args.p]))

	while not env.game_over():
		period = env.period()
		player = env.curr_player()
		if period == 1 and player == args.p:
			print('Your position:', args.p)
			print('Your hand cards:', to_chars(env.hand_cards()[args.p]))
			print('Hand cards num of 3 players:', env.hand_cards_num())
			print('-----')
		if period == 1 or period == 2:
			if player == args.p:
				if period == 1:
					_game = game.copy()
					print('Your turn:', end=' ')
					chars = input()
					try:
						if chars == 'pass':
							actions = [0]
						else:
							chars = chars.upper().rstrip().replace(',', '').split(' ')
							while '' in chars:
								chars.remove('')
							# for char in chars:
							# 	if char not in card_dict:
							# 		raise KeyError('input error')
							actions = to_action(chars)
						for action in actions:
							if action in _game.action_list():
								_game.move(action)
							else:
								raise RuntimeError('couldn\'t move')
					except (RuntimeError, KeyError):
						print('Invalid action! Please retry.')
						print('=====')
						continue
					game = _game

					for action in actions:
						env.move(action)
						for target in range(3):
							write(AIs[target], action)
					print('=====')
				if period == 2:
					print('Your bet:', end=' ')
					action = int(input()) + 352
					if action not in game.action_list():
						print('Invalid action! Please retry.')
						continue
					env.move(action)
					game.move(action)
					for target in range(3):
						write(AIs[target], action)
			else:
				action = int(read(AIs[player]))
				# print(action)
				env.move(action)
				game.move(action)
				for target in range(3):
					if target != player:
						write(AIs[target], action)
				if period == 1:
					slot += to_chars(Action(action).to_array()).replace('[', '').replace(']', '').replace(',',
					                                                                                      '').split(' ')
					if env.curr_player() != player:
						if action == 0:
							print('Player%d pass' % player)
						else:
							print('Player%d\'s turn:' % player, str(slot).replace('\'', ''))
						print('=====')
						slot = []
				if period == 2:
					print('Player%d\'s bet: %d' % (player, action - 352))
			if period == 2 and env.period() == 3:
				print('Landlord is Player%d' % env.lord_player())
				print('Bottom cards:', to_chars(env.bottom()))
				print('===== Game Start =====')
		if period == 3:
			for player in range(3):
				write(AIs[player], env.response(player))
			game.move(env.response(args.p))
			env.move(-1)
	evals = [None, None, None]
	for player in range(3):
		if player != args.p:
			evals[player] = (float(read(AIs[player])) + 1) / 2
			AIs[player].terminate()
	# print(evals)
	data.append(game.policy())
	with open('human/%s.pkl' % filename, 'wb') as f:
		pickle.dump(data, f)

	if args.p == env.lord_player():
		is_winner = env.hand_cards_num()[args.p] == 0
	else:
		is_winner = env.hand_cards_num()[env.lord_player()] != 0
	if game.lord_player() == -1:
		print('<<<<<<<<<<<<<<< DRAW >>>>>>>>>>>>>>>')
	elif is_winner:
		print('<<<<<<<<<<<<<<< YOU WIN >>>>>>>>>>>>>>>')
	else:
		print('<<<<<<<<<<<<<<< YOU LOSE >>>>>>>>>>>>>>>')


def worker(device):
	seed = int(time.time() * 100019 % 998244353)
	# seed = 10090
	np.random.seed(seed)

	AIs = [Popen('/home/tiansuan/zys/anaconda3/bin/python3 human/player.py -p %d -g %d' % (player, device), shell=True,
	             stdin=PIPE, stdout=PIPE) if player != args.p else None for player in range(3)]

	match(AIs)


if __name__ == '__main__':
	worker(args.d)
