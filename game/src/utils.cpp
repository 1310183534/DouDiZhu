#ifndef LORD__UTILS_H
#define LORD__UTILS_H

#include "../include/utils.h"
#include "../include/Action.h"

Pair get_last_action(List& policy) {
	// return (action, 0) or (action, x) [x > 0]
	// (action, 0) means this action is from opponent
	// (action, x) means this action is from me, there is only one possibility: now I should choose a attach
	if(policy.size() == 6)
		return make_pair(0, 0);
	int last_idx = policy.size() - 1;
	if(policy[last_idx] == 0) {
		last_idx --;
		if(policy[last_idx] == 0)
			return make_pair(0, 0); // if other two players pass, that means a new turn beginning from you
	}

	int attach_num = 0;
	while(Action(policy[last_idx]).is_attach()) {
		last_idx --;
		attach_num ++;
	}
	return make_pair(policy[last_idx], Action(policy[last_idx]).attach_num() - attach_num);
}

//int* unpack_state(int*, int*) {}

#endif