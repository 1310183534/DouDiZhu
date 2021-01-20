import sys
import os
import re

if re.search('gen$', os.getcwd()):
	os.chdir(re.sub('gen$', '', os.getcwd()))
	sys.path.append('.')
	print(os.getcwd())

from utils import pickle_load
import time
from game import Game, GameEnv
import numpy as np

path = 'data/'

files = os.listdir(path)
file = np.random.choice(files)

print(file)
data = pickle_load(path + file)

init, policy, v = data[0]
env = GameEnv()
print(init)
env.load(init)
print(policy)
for action in policy:
	print(env.curr_player(), env.lord_player())
	print(action)
	if env.period() != 3:
		env.move(action)
	else:
		env.move(-1)
	print(env)
	print('---------------------')
	time.sleep(0.5)
print(v)

# TODO: MCTS simulate should be same as action choice


# with open('dump.pkl', 'wb') as f:
# 	pickle.dump(data, f)
# shuffle(data)

# cnt = 0
# for _data in data[:]:
# 	# if _data[0][0][1] != 4:
# 	# 	continue
# 	if _data[1][-1] < 309:
# 		continue
# 	print(_data[0].tolist())
# 	print(_data[0])
# 	print(_data[1])
# 	print(_data[2])
# 	print('------------------------')
# 	if _data[2] == 1:
# 		cnt += 1
# print(cnt, len(data))
