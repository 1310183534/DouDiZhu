#ifndef LORD__ACTION_H
#define LORD__ACTION_H

#include <vector>
#include <cstdlib>
#include <cstring>

using namespace std;

typedef vector<int> List;

class Action {
private:
	void init();
public:
	static const int MOVE_NUM = 309;
	static const int attach_NUM = 28;
	static const int BOTTOM_NUM = 15;
	static const int BID_NUM = 4;
	static const int ACTION_NUM = 356;

	static int _action_type[ACTION_NUM];
	static int _cards_used[ACTION_NUM][15];
	static int _cards_used_num[ACTION_NUM];
	static int _attach_type[ACTION_NUM];
	static int _attach_num[ACTION_NUM];

	int action;

	Action();
	Action(int);

	int*to_array();
	int bid();
	int num();
	int num_with_attach();
	int type();

	int attach_num();
	int attach_type();
	bool need_attach();

	bool is_move();
	bool is_attach();
	bool is_bottom();
	bool is_bid();

	bool compare(int b);
	bool compare(Action b);
};

#endif
