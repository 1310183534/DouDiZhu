# distutils: language = c++

from node cimport Node
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdlib cimport free, malloc
import numpy as np


cdef class PyNode:
	cdef Node** node

	def __init__(self):
		self.node = NULL

	def __repr__(self):
		if self.node == NULL:
			return 'PyNode<ERROR>'
		if self.node[0] == NULL:
			return 'PyNode<NULL>'
		return 'PyNode<%s, %d, %.3f>' % (str(hex((<long>self.node[0]))), self.node[0][0].N, self.node[0][0].Q)

	def delete_tree(self):
		if self.is_NULL():
			raise RuntimeError('delete_tree(): node is NULL.')
		cdef Node* root = self.node[0]
		while root.father != NULL:
			root = root.father
		del root
		for i in range(NODE_POOL.size()):
			if NODE_POOL[i][0] == root:
				# print(str(hex((<long>NODE_POOL[i][0]))), str(hex((<long>root))))
				free(NODE_POOL[i])
				NODE_POOL[i] = NODE_POOL[int(NODE_POOL.size()) - 1]
				NODE_POOL.pop_back()
				return
		raise RuntimeError('delete_tree(): ???')

	def new(self, father):
		if father is None:
			self.node[0] = new Node(NULL)
		else:
			self.node[0] = new Node((<PyNode>father).node[0])

	def is_NULL(self):
		return self.node[0] == NULL

	def is_leaf(self):
		if self.is_NULL():
			raise RuntimeError("node is NULL")
		return self.node[0].is_leaf()

	def N(self):
		if self.is_NULL():
			return 0
		return self.node[0].N

	def Q(self):
		if self.is_NULL():
			return 0
		return self.node[0].Q

	def edges(self):
		if self.is_NULL():
			raise RuntimeError(self.__str__() + ": edges(): self.node is NULL")
		edges_c = self.node[0].get_edges()
		edges = []
		for edge in edges_c:
			node = PyNode()
			node.node = edge.second
			edges.append((edge.first.first, edge.first.second, node))
		return edges

	cpdef void expand(self, vector[int] move_list, vector[float] dist):
		self.node[0].expand(move_list, dist)

	def guess(self):
		cdef pair[int, Node**] guess_ret
		rand = np.random.random()
		son = PyNode()
		rand = np.random.random()
		guess_ret = self.node[0].guess(rand)
		son.node = guess_ret.second
		return guess_ret.first, son

	def select(self, add_noise, eps, c, a):
		cdef pair[int, Node**] select_ret
		cdef double[::1] noise
		son = PyNode()
		if add_noise:
			noise = np.random.dirichlet([a] * len(self.edges()))
			select_ret = self.node[0].select(&noise[0], eps, c)
		else:
			select_ret = self.node[0].select(NULL, eps, c)
		son.node = select_ret.second
		return select_ret.first, son

	def get_model_input(self):
		cdef vector[short*] history_c
		cdef float* rnn_state_c = NULL
		Rnn_channel = 256
		self.node[0].get_model_input(history_c, rnn_state_c)
		history = []
		for state in history_c:
			history.append(np.array(<short[:18, :4, :15]>state))
		history = np.array(history)
		rnn_state = np.array(<float[:Rnn_channel, :4, :15]>rnn_state_c)
		return history, rnn_state

	def set_state(self, state):
		cdef short[::1] pointer = state.flatten()
		self.node[0].set_state(&pointer[0])

	def set_rnn_state(self, rnn_state):
		cdef float[::1] pointer = np.array(rnn_state).flatten()
		self.node[0].set_rnn_state(&pointer[0])

	def delete_father_state(self):
		self.node[0].delete_father_state()

	def have_rnn_state(self):
		return self.node[0].rnn_state != NULL

	def root(self, action):
		self.node = self.node[0].root(action)

	def back_up(self, v):
		self.node[0].back_up(v)

	def lottery(self, k):
		rand = np.random.rand()
		return self.node[0].lottery(k, rand)


cdef vector[Node**] NODE_POOL

cpdef PyNode new_Root():
	root = PyNode()
	root.node = <Node**>malloc(sizeof(Node*))
	NODE_POOL.push_back(root.node)
	root.node[0] = new Node(NULL)
	root.node[0].is_root = True
	return root

cpdef void free_all():
	cdef Node** node
	while NODE_POOL.size():
		node = NODE_POOL.back()
		# print('free_nodes:', hex(<long>node))
		del node[0]
		free(node)
		NODE_POOL.pop_back()