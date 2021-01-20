#ifndef LORD__GAME_CP_CPP
#define LORD__GAME_CP_CPP

#include <iostream>
#include "../include/utils.h"
#include "../include/Env.h"


GameEnv::GameEnv() {
	memset(hand_cards, 0, sizeof(hand_cards));
	memset(hand_cards_num, 0, sizeof(hand_cards_num));
	memset(bottom, 0, sizeof(bottom));
	curr_player = 0;
	lord_player = -1;
	set_period(2);
}

GameEnv::~GameEnv() {}

void GameEnv::next_player() {
	curr_player = (curr_player + 1) % 3;
}

int GameEnv::game_over() {
	return _period == 0;
}

int GameEnv::period() {
	return _period;
}

void GameEnv::set_period(int __period) {
	_period = __period;
}

void GameEnv::move_type1(Action& action) {
	int* cards_required = action.to_array();
	hand_cards_num[curr_player] -= action.num();
	for(int i = 0; i < 15; i ++) {
		hand_cards[curr_player][i] -= cards_required[i];
	}
	policy.push_back(action.action);
	if(get_last_action(policy).second == 0)
		next_player();
	if(hand_cards_num[0] == 0 || hand_cards_num[1] == 0 || hand_cards_num[2] == 0)
		set_period(0);
//	free(cards_required);
}

void GameEnv::move_type2(Action& action) {
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

void GameEnv::move_type3(int _action=-1) {
	policy.push_back(-1);
	int idx = int(policy.size()) - 3;
	int c = 0;

	for(int i = 0; i < 15; i ++) {
		if((c + bottom[i]) >= idx) {
			if(_action != -1 && _action != 337 + i) {
				cout << "GameEnv::move action != -1" << endl;
				exit(-14);
			}
			hand_cards[lord_player][i] += 1;
			hand_cards_num[lord_player] += 1;
			break;
		}
		c += bottom[i];
	}

	if(policy.size() == 6) {
		set_period(1);
	}
}

void GameEnv::move(int _action) {
	Action action(_action);

		 if(period() == 1) move_type1(action);
	else if(period() == 2) move_type2(action);
	else if(period() == 3) {
		move_type3(_action);
	}
//	cout << "Game::move\t" << (a.action) << endl;

}

int GameEnv::response(int player) {
	int idx = int(policy.size()) - 3;
	int c = 0;

	for(int i = 0; i < 15; i ++) {
		if((c + bottom[i]) > idx) {
			return i + 337;
		}
		c += bottom[i];
	}
	cout << "GameEnv::response ??" << endl;
	exit(-14);
}

//void Game::load(int (&init)[3][15]) {
//	for(int player = 0; player < 3; player ++) {
//		for(int i = 0; i < 15; i ++) {
//			hand_cards[player][i] = init[player][i];
//		}
//	}
//	hand_cards_num[0] = 20;
//	hand_cards_num[1] = 17;
//	hand_cards_num[2] = 17;
//	current_player = 0;
//	history.clear();
//}

#endif
