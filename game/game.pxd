# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.pair cimport pair

ctypedef pair[int, int] Pair
ctypedef vector[int] List

cdef extern from "include/Action.h":
	cdef cppclass Action:
		int action
		Action() except +
		Action(int) except +
		int*to_array()
		int num()
		int type()
		int attach_num()
		int attach_type()
		bint need_attach()
		bint is_attach()
		bint is_bottom()
		bint is_bid()

cdef extern from "include/Env.h":
	cdef cppclass GameEnv:
		int hand_cards[3][15]
		int hand_cards_num[3]
		int bottom[15]
		int curr_player
		int lord_player
		List policy
		GameEnv() except +
		int game_over()
		int period()
		void move(int)
		int response(int)

cdef extern from "include/Game.h":
	cdef cppclass Game:
		int hand_cards_num[3]
		int hand_cards[15]
		int remaining_cards[15]
		int bottom[15]
		int curr_player
		int my_player
		int lord_player
		List policy
		Game() except +
		Game(const GameEnv&) except +
		Game(const GameEnv&, int) except +
		short* to_model_input()
		int game_over()
		float eval()
		int period()
		bint could_move(int)
		void move(int)
