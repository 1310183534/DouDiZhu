from game import PyGame as Game
from game import PyGameEnv as GameEnv
from game import PyAction as Action
import numpy as np


def test_once():
	env = GameEnv()

	# cards = [[3, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
	# 		 [0, 2, 2, 2, 2, 2, 2, 2, 3, 0, 0, 0, 0, 0, 0],
	# 		 [0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 4, 4, 0, 0]]
	# env.load(np.array(cards))
	# history = [355, 352, 352]
	# history = [355, 352, 352, 337, 350, 351, 109, 124, 0, 14]
	history = []
	
	games = [Game(env, 0), Game(env, 1), Game(env, 2)]
	print(games[0].period())
	return
	
	for action in history:
		games[0].move(action)
		games[1].move(action)
		games[2].move(action)
		env.move(action)
	
	# print(games[0])
	# print(games[0].curr_player())
	# print(games[1])
	# print(games[1].curr_player())
	# print(games[2])
	# print(games[2].curr_player())
	# print(env)
	print('=================')
	
	# env.load(np.array(cards, dtype=np.int32))
	# print(games[0], games[1], games[2])
	print(env.bottom())
	
	while not env.game_over():
		period = env.period()
		curr_player = env.curr_player()
		print('-----------------------------------')
		print('period:', period)
		print('curr_player:', curr_player)
		
		if period == 3:  # bottom
			print('response:', env.response(0), env.response(1), env.response(2))
			print(games[0].action_list())
			print('remaining cards:')
			print(games[0].remaining_cards())
			print(games[1].remaining_cards())
			print(games[2].remaining_cards())
			games[0].move(env.response(0))
			games[1].move(env.response(1))
			games[2].move(env.response(2))
			print('bottom:')
			print(games[0].bottom())
			print(games[1].bottom())
			print(games[2].bottom())
			env.move(-1)
			print(env.hand_cards_num())
			model_input = games[0].to_model_input()
		# for channel in range(18):
		# 	print(channel)
		# 	print(model_input[channel])
		else:
			action_list = games[curr_player].action_list()
			action = np.random.choice(action_list)
			print('action_list:')
			print(games[0].action_list())
			print(games[1].action_list())
			print(games[2].action_list())
			action_c = Action(action)
			print('action:', action)
			print('', Action(action).to_array())
			games[0].move(action)
			games[1].move(action)
			games[2].move(action)
			env.move(action)
			print(env.hand_cards_num())
			# if action_c.need_attach() or action_c.is_attach() or action_c.is_bottom() or action_c.is_bid():
			# 	model_input = games[0].to_model_input()
			# 	for channel in range(18):
			# 		print(channel)
			# 		print(model_input[channel])
		print(env)
		print(games[0].eval())
		print(games[1].eval())
		print(games[2].eval())


while True:
	test_once()
