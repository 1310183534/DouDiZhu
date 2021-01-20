#ifndef LORD__GAME_ICP_H
#define LORD__GAME_ICP_H

#include "include/Env.h"
#include "include/Action.h"

class Game {
private:
	bool could_move_type1(Action&);  // MOVE or ATTACH
	bool could_move_type2(Action&);  // BID
	bool could_move_type3(Action&);  // BOTTOM

	void move_type1(Action&);
	void move_type2(Action&);
	void move_type3(Action&);

	int _period;
	float _eval;
	void next_player();
public:
	int hand_cards[15];
	int remaining_cards[15];
	int bottom[15];
	int hand_cards_num[3];

	int my_player;
	int curr_player;
	int lord_player;
	List policy;

	 Game();
	 Game(const GameEnv&, int);
	~Game();

	short* to_model_input();
	void set_period(int);
	void set_eval(float);
	int period();
	int game_over();
	int eval();
	bool could_move(int);
	void move(int);
};

#endif
