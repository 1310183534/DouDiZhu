#ifndef LORD__NODE_CPP
#define LORD__NODE_CPP

#include <cmath>
#include <cstring>
#include "../include/node.h"

const int Rnn_channel = 256;
const int state_SIZE = 18 * 4 * 15 * sizeof(short);
const int rnn_state_SIZE = Rnn_channel * 4 * 15 * sizeof(float);

#define ASSERT(expression, signal) {if(expression) {cout << signal << endl; exit(-1); } }

Node::Node() {
	N = 0, V = 0.0, Q = 0.0;
	is_root = false;
	father = NULL;
	edges.clear();
	state = NULL;
	rnn_state = NULL;
}

Node::Node(Node* _father) {
	N = 0, V = 0.0, Q = 0.0;
	is_root = false;
	father = _father;
	edges.clear();
	state = NULL;
	rnn_state = NULL;
}

Node::~Node() {
	for(Edge& edge: edges) {
		delete get<2>(edge);
	}
	edges.clear();
	free(state);
	free(rnn_state);
}

bool Node::is_leaf() {
	return edges.size() == 0;
}

void Node::update(float v) {
	N += 1;
	V += v;
	Q = V / N;
}

void Node::expand(List& action_list, vector<float>& dist) {
	ASSERT(action_list.size() == 0, "Node::expand: action_list is empty!");
	float sum_prob = 0.0;
	for(int action: action_list) {
		sum_prob += dist[action];
		edges.push_back(make_tuple(action, dist[action], (Node*)NULL));
	}
	for(Edge& edge: edges) {
		get<1>(edge) /= sum_prob;
	}
}

pair<int, Node**> Node::guess(double rand) {
	double prob_sum = 0;
	for(int i = 0; i < int(edges.size()); i ++) {
		Edge& edge = edges[i];
		int action = get<0>(edge);
		Node** son = &get<2>(edge);
		prob_sum += get<1>(edge);
		if(prob_sum >= rand || i == int(edges.size()) - 1) {
			if(*son == NULL) {
				*son = new Node(this);
			}
			return make_pair(action, son);
		}
	}
	ASSERT(true, "Node::guess: ERROR.");
}

pair<int, Node**> Node::select(double* noise, double eps, double c) {
	int select_action = -1;
	Node** select_son = NULL;
	float MaxQU = 0.0;
	for(int i = 0; i < int(edges.size()); i ++) {
		Edge& edge = edges[i];
		int action = get<0>(edge);
		float P = get<1>(edge);
		Node** son = &get<2>(edge);
		int ni = *son == NULL ? 0: (*son) -> N;
		float Q = *son == NULL ? this -> Q: (*son) -> Q;  // init to father.Q  // default
//		float Q = *son == NULL ? -1.0: *son -> Q;  // init to lose
		float U = 0.0;
		if(noise != NULL) {
			U = c * (eps * noise[i] + (1 - eps) * P) * sqrt(N) / (1 + ni);
		}
		else {
			U = c * P * sqrt(N) / (1 + ni);
		}
		if(select_son == NULL || Q + U > MaxQU) {
			MaxQU = Q + U;
			select_action = action;
			select_son = son;
		}
	}
	ASSERT(select_son == NULL, "Node::select: no any son.");
	if(*select_son == NULL) {
		*select_son = new Node(this);
	}
	return make_pair(select_action, select_son);
}

void Node::get_model_input(History& _history, float*& _rnn_state) {
	static float zero_rnn_state[Rnn_channel][4][15];
	static bool init = false;

	if(init == false) {
		memset(zero_rnn_state, 0, rnn_state_SIZE);
		init = true;
	}

	if(rnn_state != NULL) {
		_rnn_state = rnn_state;
		return;
	}
	if(father != NULL) {
		father -> get_model_input(_history, _rnn_state);
	}
	else {
		_rnn_state = (float*)zero_rnn_state;
	}
	_history.push_back(state);
}

vector<pair<pair<int, float>, Node**>> Node::get_edges() {
	vector<pair<pair<int, float>, Node**>> ret;
	for(Edge& edge: edges) {
		int action = get<0>(edge);
		float P = get<1>(edge);
		Node** son = &get<2>(edge);
		ret.push_back(make_pair(make_pair(action, P), son));
	}
	return ret;
}

void Node::set_state(short* _state) {
	ASSERT(state != NULL, "Node::set_state: state is not NULL.");
	state = (short*)malloc(state_SIZE);
	memcpy(state, _state, state_SIZE);
}

void Node::set_rnn_state(float* _rnn_state) {
	ASSERT(rnn_state != NULL, "Node::set_rnn_state: rnn_state is not NULL.");
	rnn_state = (float*)malloc(rnn_state_SIZE);
	memcpy(rnn_state, _rnn_state, rnn_state_SIZE);
}

void Node::delete_father_state() {
	if(father != NULL) {
		free(father -> state);
		free(father -> rnn_state);
		father -> state = NULL;
		father -> rnn_state = NULL;
	}
}

bool Node::lottery(double k, double rand) {
	if(is_root) return true;
	if(father -> is_root) return true;
	for(Edge& edge: father -> edges) {
		float P = get<1>(edge);
		Node* son = get<2>(edge);
		if(son == this) {
			return P * k >= rand;
		}
	}
	ASSERT(true, "Node::lottery: My father didn't recognize me.");
}

Edge& Node::son(int action) {
	for(Edge& edge: edges) {
		if(get<0>(edge) == action) {
			return edge;
		}
	}
	ASSERT(true, "Node::son: action is not available.");
	cout << "No this son" << endl;
	exit(-1);
}

Node** Node::root(int action) {
	Node** new_root = &get<2>(son(action));
	if(*new_root == NULL) {
		*new_root = new Node(this);
	}
	for(Edge& edge: edges) {
		if(get<2>(edge) != *new_root) {
			delete get<2>(edge);
			get<2>(edge) = NULL;
		}
	}
	(*new_root) -> is_root = true;
	return new_root;
}

void Node::back_up(float v) {
	update(v);
	if(is_root) return;
	if(father != NULL) father -> back_up(v);
}

#endif
