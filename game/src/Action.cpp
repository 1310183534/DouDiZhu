#ifndef LORD__ACTION_CPP
#define LORD__ACTION_CPP

#include <iostream> 
#include "../include/Action.h"

int Action::_action_type[ACTION_NUM];
int Action::_cards_used[ACTION_NUM][15];
int Action::_cards_used_num[ACTION_NUM];
int Action::_attach_type[ACTION_NUM];
int Action::_attach_num[ACTION_NUM];

void Action::init() {
    static bool have_init = false;
    if(have_init) return;

	static const int num_of_type[] = {1, 14, 15, 13, 13, 13, 13, 13, 13, 36, 55, 66, 66, 66, 15, 13, 15, 4};
	int type = 0;
	int action_num = 0;
	for(int _action = 0, id = 0; _action < 444; _action ++, id ++) {
		int cards[15];
		int num = 0;
		int l = 0, r = 0;
		memset(cards, 0, sizeof(cards));
		
		if(id - num_of_type[type] >= 0) {
			id -= num_of_type[type];
			type ++;
		}
		
		if(9 <= type && type <= 13) {
			l = 0;
			int num = 0, _id = id;
				 if(type == 9) num = 8;
			else if(type == 10) num = 10;
			else num = 11;
			while(_id - num >= 0) {
				_id -= num;
				num --;
				l ++;
			}
			r = (12 - num) + _id;
		}
		
		switch(type) {
			case 0: { num = 0; break; }	                 // Pass
			case 1: {                                    // Boom
				if(id == 13) { cards[13] += 1; cards[14] += 1; num = 2; }
				else { cards[id] += 4; num = 4; }
				break;
			}
			case 2: { cards[id] += 1; num = 1; break; }  // single
			case 3: { cards[id] += 2; num = 2; break; }  // double
			case 4: { cards[id] += 3; num = 3; break; }  // three
			case 5: { cards[id] += 3; num = 3; break; }  // three + single
			case 6: { cards[id] += 3; num = 3; break; }  // three + double
			case 7: { cards[id] += 4; num = 4; break; }  // four + 2 * single
			case 8: { cards[id] += 4; num = 4; break; }  // four + 2 * double
			case 9: {   // straight single
				for(int i = l; i <= r; i ++) cards[i] += 1;
				num = (r - l + 1);
				break; 
			}
			case 10: {  // straight double
				for(int i = l; i <= r; i ++) cards[i] += 2;
				num = (r - l + 1) * 2;
				break; 
			}
			case 11: {  // straight three
				for(int i = l; i <= r; i ++) cards[i] += 3;
				num = (r - l + 1) * 3;
				break; 
			}
			case 12: {  // straight three + single
				for(int i = l; i <= r; i ++) cards[i] += 3;
				num = (r - l + 1) * 3;
				break; 
			}
			case 13: {  // straight three + double
				for(int i = l; i <= r; i ++) cards[i] += 3;
				num = (r - l + 1) * 3;
				break; 
			}
			case 14: { cards[id] += 1; num = 1; break; }  // attach 1
			case 15: { cards[id] += 2; num = 2; break; }  // attach 2
			case 16: { cards[id] += 1; break; }  // bottom
			case 17: { break; }  // bottom
		}

		switch(type) {
			case 5: {
				_attach_type[action_num] = 1;
				_attach_num[action_num] = 1;
				break;
			}
			case 6: {
				_attach_type[action_num] = 2;
				_attach_num[action_num] = 1;
				break;
			}
			case 7: {
				_attach_type[action_num] = 1;
				_attach_num[action_num] = 2;
				break;
			}
			case 8: {
				_attach_type[action_num] = 2;
				_attach_num[action_num] = 2;
				break;
			}
			case 12: {
				_attach_type[action_num] = 1;
				_attach_num[action_num] = (r - l + 1);
				break;
			}
			case 13: {
				_attach_type[action_num] = 2;
				_attach_num[action_num] = (r - l + 1);
				break;
			}
			default: {
				_attach_type[action_num] = 0;
				_attach_num[action_num] = 0;
				break;
			}
		}
		
		if(num + _attach_type[action_num] * _attach_num[action_num] > 20) continue;
		memcpy(_cards_used[action_num], cards, sizeof(cards));
		_cards_used_num[action_num] = num;
		_action_type[action_num] = type;
		action_num ++;
		
//		cout << action_num << ", " << num << ", " << _action_type[action_num - 1] << ": ";
//		for(int i = 0; i < 15; i ++) {
//			cout << cards[i] << " ";
//		}
//		cout << endl;
	}

	have_init = true;
}

Action::Action() {
	action = 0;
	init();
}

Action::Action(int _action) {
	action = _action;
	init();
}

int* Action::to_array() {
//	int* ret = (int*)malloc(sizeof(int) * 15);
//	memcpy(ret, _cards_used[action], sizeof(int) * 15);
//	return ret;

	return  _cards_used[action];

	// TODO: don't malloc, don't free
	// TODO: attach -> attach
}

int Action::bid() {
	if(!is_bid()) return -1;
	return action - 352;
}

int Action::num() {
    return _cards_used_num[action];
}

int Action::num_with_attach() {
	return num() + attach_num() * attach_type();
}

int Action::type() {
    return _action_type[action];
}

int Action::attach_num() {
    return _attach_num[action];
}

int Action::attach_type() {
    return _attach_type[action];
}

bool Action::need_attach() {
	return attach_num() != 0;
}

bool Action::is_move() {
	return type() <= 13;
}

bool Action::is_attach() {
	return type() == 14 || type() == 15;
}

bool Action::is_bottom() {
	return type() == 16;
}

bool Action::is_bid() {
	return type() == 17;
}

bool Action::compare(int b) {
	return compare(Action(b));
}

bool Action::compare(Action b) {
	// two un_attach action a(this) and b, return if a could be the next action after b
	if(is_attach() || b.is_attach())
		return false;

	if(type() == 0 && b.type() == 0)
		return false;

	if(type() == 0 || b.type() == 0)
		return true;

	if(type() == 1 && b.type() == 1)
		return action > b.action;

	if(type() == 1)
		return true;

	if(type() != b.type())
		return false;

	if(num() != b.num()) // 334455 and 44556677 have different num
		return false;

	return action > b.action;
}

#endif

//int main() {
//	freopen("table.txt", "w", stdout);
//	for(int _action = 0; _action < Action::ACTION_NUM; _action ++) {
//		Action action(_action);
//		cout << _action << "\t" << action.type() << ",\t" << action.num() << " " << action.num_with_attach() << "\t: ";
//		int* cards = action.to_array();
//		for(int i = 0; i < 15; i ++) cout << cards[i] << " ";
//		cout << endl;
//	}
//}
