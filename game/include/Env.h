#ifndef LORD__ENV_H
#define LORD__ENV_H

#include "include/Action.h"

class GameEnv {
private:
	void move_type1(Action&);
	void move_type2(Action&);
	void move_type3(int);

	int _period;
	void next_player();
public:
	int hand_cards[3][15];
	int hand_cards_num[3];
	int bottom[15];

	int curr_player;
	int lord_player;
	List policy;

	 GameEnv();
	~GameEnv();

//	short* to_model_input(int);
	int game_over();
	void set_period(int);
	int period();
//	int eval();
//	bool could_move(int);
	void move(int);
	int response(int);
};

#endif
