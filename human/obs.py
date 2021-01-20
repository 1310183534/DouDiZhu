import sys
import os
import re
import argparse

sys.path.append('.')
if re.search('human', os.getcwd()):
	os.chdir(re.sub('human$', '', os.getcwd()))
	sys.path.append('.')

import pickle
from game import GameEnv

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=str, default='_00208c67bd32af90')
args = parser.parse_args()


def main():
	filename = args.n
	with open('human/%s.pkl' % filename, 'rb') as f:
		info, init, history = pickle.load(f)
		print(str(init).replace(' ', ', '))
		print(history)
		print(info)
		env = GameEnv()
		env.load(init)
		for action in history:
			print('-----------------')
			print('action: %d' % action)
			print('player: %d' % env.curr_player())
			env.move(action)
			if env.period() == 3:
				print(env.response(0))
			print(env)


if __name__ == '__main__':
	main()
