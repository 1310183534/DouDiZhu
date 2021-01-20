#ifndef LORD__GAME_CPP
#define LORD__GAME_CPP

#include <iostream>
#include <algorithm>
#include "../include/utils.h"
#include "../include/Game.h"


Game::Game() {}

Game::Game(const GameEnv& env, int _my_player) {
	memcpy(hand_cards_num, env.hand_cards_num, sizeof(hand_cards_num));
	memset(hand_cards, 0, sizeof(hand_cards));
	memset(remaining_cards, 0, sizeof(remaining_cards));
	memset(bottom, 0, sizeof(bottom));
	my_player = _my_player;
	curr_player = 0;
	set_period(2);
	set_eval(0.0);
	lord_player = -1;
//	history = state.history;

	for(int i = 0; i < 15; i ++) {
		hand_cards[i] += env.hand_cards[my_player][i];
	}
	for(int i = 0; i < 15; i ++) {
		remaining_cards[i] = (((i == 13) || (i == 14)) ? 1 : 4) - hand_cards[i];
	}
}

Game::~Game() {}

void Game::next_player() {
	curr_player = (curr_player + 1) % 3;
}

short* Game::to_model_input() {
	typedef short D[18][4][15];
	D& ret = *((D*)malloc(sizeof(D)));
	memset(ret, 0, sizeof(D));

	for(int j = 0; j < 4; j ++) {
		for(int i = 0; i < 15; i ++) {
			ret[0][j][i] = hand_cards[i] > j;

			if(policy.size()) {
				Action action(policy.back());
				int* array = action.to_array();

				if(action.is_move())
					ret[1][j][i] = action.attach_type();
				if(action.is_attach())
					ret[1][j][i] = action.num();

				if(action.is_move())
					ret[2][j][i] = array[i] > j;
				if(action.is_attach())
					ret[3][j][i] = array[i] > j;
				if(action.is_bottom())
					ret[4][j][i] = array[i] > j;
//				ret[4][j][i] = bottom[i] > j;
				if(action.is_bid())
					ret[5][j][i] = action.bid();
			}

			ret[6][j][i] = my_player == 0;
			ret[7][j][i] = my_player == 1;
			ret[8][j][i] = my_player == 2;

			ret[9][j][i] = curr_player == 0;
			ret[10][j][i] = curr_player == 1;
			ret[11][j][i] = curr_player == 2;

			ret[12][j][i] = lord_player == 0;
			ret[13][j][i] = lord_player == 1;
			ret[14][j][i] = lord_player == 2;

			ret[15][j][i] = hand_cards_num[0];
			ret[16][j][i] = hand_cards_num[1];
			ret[17][j][i] = hand_cards_num[2];
		}
	}

	return (short*)&ret;
}

int Game::game_over() {
	return _period == 0;
}

int Game::period() {
	return _period;
}

void Game::set_period(int __period) {
	_period = __period;
}

void Game::set_eval(float __eval) {
	_eval = __eval;
}

int Game::eval() {
	return _eval;
}

bool Game::could_move_type1(Action& action) {
	if((!action.is_move()) && (!action.is_attach())) return false;
	if(hand_cards_num[curr_player] - action.num_with_attach() < 0) return false;

	int* cards_required = action.to_array();
	int* curr_player_hand_cards = (curr_player == my_player ? hand_cards : remaining_cards);

	if(curr_player == lord_player) {
		int bottom_num = 0;
		for(int i = 0; i < 15; i ++) {
			bottom_num += max(0, bottom[i] - cards_required[i]);
		}
		if(hand_cards_num[curr_player] - action.num_with_attach() < bottom_num) return false;
	}

	int possible_attach_num = 0;
	for(int i = 0; i < 15; i ++) {
		if(curr_player == lord_player) {
			if(cards_required[i] > curr_player_hand_cards[i] + bottom[i]) {
				return false;
			}
		}
		else if(cards_required[i] > curr_player_hand_cards[i]) {
			return false;
		}
		if(action.attach_type()) {
//			cout << i << " " << ((curr_player_hand_cards[i] - cards_required[i]) / action.attach_type()) << endl;
//			cout << i << " " << ((curr_player_hand_cards[i] + bottom[i] - cards_required[i]) / action.attach_type()) << endl;
			int b = curr_player == lord_player ? bottom[i] : 0;
			possible_attach_num += (curr_player_hand_cards[i] + b - cards_required[i]) / action.attach_type();
		}
	}

	if(possible_attach_num < action.attach_num())
		return false;

	Pair last_action = get_last_action(policy);
	Action last_move(last_action.first);
	if(last_action.second) {
		if(!action.is_attach() || action.num() != last_move.attach_type())
			return false;
		return true;
	}
	return action.compare(last_move);
}

int max(int a, int b) {
	return a > b ? a : b;
}

bool Game::could_move_type2(Action& action) {
	if(!action.is_bid()) return false;
	if(action.bid() == 0) return true;
	int max_bid = 0;
	for(int player = 0; player < int(policy.size()); player ++) {
		Action bid_action(policy[player]);
		max_bid = max(max_bid, bid_action.bid());
	}
	return action.bid() > max_bid;
}

bool Game::could_move_type3(Action& action) {
	if(!action.is_bottom()) return false;
	int* bottom_array = action.to_array();
	for(int j = 0; j < 15; j ++) {
		if(bottom_array[j] > remaining_cards[j]) {
//			free(bottom_array);
			return false;
		}
	}
//	free(bottom_array);
	return true;
}

bool Game::could_move(int _action) {
	Action action(_action);

	if(period() == 1) return could_move_type1(action);
	if(period() == 2) return could_move_type2(action);
	if(period() == 3) return could_move_type3(action);
	return false;
}

void Game::move_type1(Action& action) {
	int* cards_required = action.to_array();
	int* curr_player_hand_cards = (curr_player == my_player ? hand_cards : remaining_cards);
	hand_cards_num[curr_player] -= action.num();
	for(int i = 0; i < 15; i ++) {
		curr_player_hand_cards[i] -= cards_required[i];
		if(curr_player == lord_player) {
			int bottom_used = std::min(cards_required[i], bottom[i]);
			bottom[i] -= bottom_used;
			curr_player_hand_cards[i] += bottom_used;
		}
	}
	policy.push_back(action.action);
	if(get_last_action(policy).second == 0)
		curr_player = (curr_player + 1) % 3;
	if(hand_cards_num[0] == 0 || hand_cards_num[1] == 0 || hand_cards_num[2] == 0) {
		set_period(0);
		bool i_am_lord = my_player == lord_player;
		bool lord_lose = hand_cards_num[lord_player] != 0;
		if(i_am_lord ^ lord_lose) set_eval(1.0);
		else set_eval(-1.0);
	}
}

void Game::move_type2(Action& action) {
	policy.push_back(action.action);

	if(policy.size() == 3) {
		int max_bid = 0;
		for(int player = 0; player < 3; player ++) {
			Action bid = Action(policy[player]);
			if(bid.bid() > max_bid) {
				max_bid = bid.bid();
				lord_player = player;
			}
		}
		if(lord_player == -1) {
			set_period(0);
			set_eval(0.0);
		}
		else {
			curr_player = lord_player;
			set_period(3);
		}
	}
	else {
		next_player();
	}
}

void Game::move_type3(Action& action) {
	int* bottom_array = action.to_array();
	for(int i = 0; i < 15; i ++) {
		bottom[i] += bottom_array[i];
		remaining_cards[i] -= bottom_array[i];
	}
	hand_cards_num[lord_player] += 1;

	policy.push_back(action.action);

	if(policy.size() == 6) {
		set_period(1);
	}
}

void Game::move(int _action) {
	Action action(_action);

	if(!could_move(_action)) {
		cout << "could't move";
		exit(-7);
	}

		 if(period() == 1) move_type1(action);
	else if(period() == 2) move_type2(action);
	else if(period() == 3) move_type3(action);
//	cout << "Game::move\t" << (a.action) << endl;
}

#endif
