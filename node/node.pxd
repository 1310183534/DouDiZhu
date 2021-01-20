# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.pair cimport pair

ctypedef vector[int] List
ctypedef vector[float] Dist
ctypedef vector[short*] History
ctypedef pair[int, float] pair__int_float
ctypedef pair[pair__int_float, Node**] Edge

cdef extern from "include/node.h":
	cdef cppclass Node:
		int N
		float V
		float Q
		bint is_root

		Node* father
		Node* rnn_father
		short* state
		float* rnn_state

		Node() except +
		Node(Node*) except +

		bint is_leaf()
		void expand(List&, Dist&)
		pair[int, Node**] guess(double)
		pair[int, Node**] select(double*, double, double)
		void get_model_input(History&, float*&)

		vector[Edge] get_edges()
		void set_state(short*)
		void set_rnn_state(float*)
		void delete_father_state()
		bint lottery(double, double)

		Node** root(int)
		void back_up(float)
