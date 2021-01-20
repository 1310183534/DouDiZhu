#ifndef LORD__NODE_H
#define LORD__NODE_H

#include <iostream>
#include <vector>
#include <tuple>

using namespace std;

class Node;

typedef vector<int> List;
typedef tuple<int, float, Node*> Edge;
typedef vector<short*> History;

class Node {
public:
	int N;
	float V;
	float Q;
	bool is_root;

	Node* father;
	vector<Edge> edges;
	short* state;
	float* rnn_state;

	Node();
	Node(Node*);
	~Node();

	bool is_leaf();
	void update(float);
	void expand(List&, vector<float>&);
	pair<int, Node**> guess(double);
	pair<int, Node**> select(double*, double, double);
	void get_model_input(History&, float*&);

	vector<pair<pair<int, float>, Node**>> get_edges();

	void set_state(short*);
	void set_rnn_state(float*);
	void delete_father_state();
	bool lottery(double, double);

	Edge& son(int);
	Node** root(int);

	void back_up(float);
};

#endif