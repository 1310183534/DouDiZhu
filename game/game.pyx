# distutils: language = c++

from game cimport Game
from game cimport Action
from game cimport GameEnv

from libc.stdlib cimport free
from cpython cimport array
import numpy as np

__init_cards = np.array(list(range(13)) * 4 + [13, 14], dtype=np.int32)
ACTION_NUM = 356


cdef class PyAction:
	cdef Action action

	def __init__(self, action):
		self.action.action = action


	def to_array(self):
		cdef int* ptr = self.action.to_array()
		a = np.array(<int[:15]>ptr).copy()
		return a

	def num(self):
		return self.action.num()

	def type(self):
		return self.action.type()

	def attach_num(self):
		return self.action.attach_num()

	def attach_type(self):
		return self.action.attach_type()

	def need_attach(self):
		return self.action.need_attach()

	def is_attach(self):
		return self.action.is_attach()

	def is_bottom(self):
		return self.action.is_bottom()

	def is_bid(self):
		return self.action.is_bid()


cdef class PyGameEnv:
	cdef GameEnv env

	def __init__(self, init=True):
		if init:
			init_cards = __init_cards.copy()
			np.random.shuffle(init_cards)
			# print(init_cards)
			for i in range(51):
				player = i % 3 if i < 51 else 0
				self.env.hand_cards[player][init_cards[i]] += 1
			for i in range(51, 54):
				self.env.bottom[init_cards[i]] += 1
			self.env.hand_cards_num[0] = 17
			self.env.hand_cards_num[1] = 17
			self.env.hand_cards_num[2] = 17

	def __str__(self):
		return str(np.array(self.env.hand_cards))

	def policy(self):
		return self.env.policy

	def bottom(self):
		return np.array(<int[:15]>self.env.bottom)

	def hand_cards_num(self):
		return np.array(<int[:3]>self.env.hand_cards_num)

	def hand_cards(self):
		return np.array(<int[:3, :15]>self.env.hand_cards)

	def curr_player(self):
		return self.env.curr_player
	
	def lord_player(self):
		return self.env.lord_player

	def curr_player_hand_cards(self):
		return np.array(<int[:15]>(self.env.hand_cards[self.env.curr_player]))

	def move(self, action):
		self.env.move(action)

	def period(self):
		return self.env.period()

	def response(self, player):
		return self.env.response(player)

	def game_over(self):
		return self.env.game_over()

	def load(self, int[:, :] init):
		cdef int player, i
		for i in range(15):
			self.env.bottom[i] = 1 if i >= 13 else 4
			for player in range(3):
				self.env.hand_cards[player][i] = init[player][i]
				self.env.bottom[i] -= init[player][i]
		self.env.hand_cards_num[0] = 17
		self.env.hand_cards_num[1] = 17
		self.env.hand_cards_num[2] = 17

	def copy(self):
		ret = PyGameEnv(False)
		ret.env = self.env
		return ret

__PyGameEnv = PyGameEnv(False)

cdef class PyGame:
	cdef Game game

	def __init__(self, PyGameEnv env, my_player):
		if my_player is None:
			return
		self.game = Game(env.env, my_player)

	def __str__(self):
		return str(np.array(self.game.hand_cards)) + '\n' + str(np.array(self.game.remaining_cards))

	def policy(self):
		return self.game.policy

	def hand_cards_num(self):
		return np.array(<int[:3]>self.game.hand_cards_num)

	def hand_cards(self):
		return np.array(<int[:15]>self.game.hand_cards)

	def remaining_cards(self):
		return np.array(<int[:15]>self.game.remaining_cards)
	
	def bottom(self):
		return np.array(<int[:15]>self.game.bottom)

	def curr_player(self):
		return self.game.curr_player

	def my_player(self):
		return self.game.my_player
	
	def lord_player(self):
		return self.game.lord_player

	def to_model_input(self):
		cdef short* ptr = self.game.to_model_input()
		array = np.array(<short[:18, :4, :15]>ptr).copy()
		free(ptr)
		return array

	def move(self, action):
		self.game.move(action)

	def could_move(self, action):
		return self.game.could_move(action)
	
	def period(self):
		return self.game.period()

	cpdef array.array action_list(self):
		cdef array.array action_list = array.array('i', [])
		cdef int action, length
		for action in range(ACTION_NUM):
			if self.game.could_move(action):
				length = len(action_list)
				array.resize_smart(action_list, length + 1)
				action_list.data.as_ints[length] = action
		return action_list

	def game_over(self):
		return self.game.game_over()

	def eval(self):
		return self.game.eval()
	
	def gauss(self):
		return self.game.my_player != self.game.curr_player or self.game.period() == 3

	def copy(self):
		ret = PyGame(__PyGameEnv, None)
		ret.game = self.game
		return ret
