from utils.utils import vector
import numpy as np


class MCT:
	def __init__(self, init_game, root):
		self.init_game = init_game.copy()
		# self.player = player  # the player I'm
		self.root = root

	def move(self, action):
		self.init_game.move(action)
		self.root.root(action)

	def reach_leaf(self):
		game = self.init_game.copy()
		if self.root.is_leaf():
			return self.root, game
		curr_node = self.root
		while not game.game_over():
			if game.gauss():
				action, son = curr_node.guess()
			else:
				add_noise = self.root == curr_node
				# eps = 0.25 if game.period() != 3 else 0.25
				# c = 1.0 if game.period() != 3 else 2.0
				action, son = curr_node.select(add_noise, eps=0.25, c=1.0, a=0.1)
			game.move(action)
			if son.is_leaf():
				return son, game
			curr_node = son
		return curr_node, game

	def _pi_inf(self):
		select_action = -1
		select_node_N = 0
		for edge in self.root.edges():
			if edge[2].N() > select_node_N:
				select_action = edge[0]
				select_node_N = edge[2].N()
		if select_action == -1:
			raise RuntimeError('???')
		return vector(select_action)

	def _pi(self, t):
		ret = np.zeros(356)
		edges = self.root.edges()
		avg_n = np.mean([edge[2].N() for edge in edges])
		for edge in edges:
			n = edge[2].N() + 0.001 - avg_n
			if n > 0:
				ret[edge[0]] = edge[2].N()
		ret **= t
		ret /= ret.sum()
		return ret

	def pi(self, t):
		if t is None:
			return self._pi_inf()
		return self._pi(t)

	def choice(self, eta=1.0, t=None):
		if np.random.random() <= eta:
			# follow MCTS policy with probability eta
			p = self.pi(t=t)
		else:
			# follow neural network policy with probability (1 - eta)
			p = np.zeros(356)
			for edge in self.root.edges():
				p[edge[0]] = edge[1]
			p /= p.sum()
		return np.random.choice(356, p=p)

	def json(self, filename='visual/mct.json'):
		#  for visualization
		def dfs(cnt, node, game, action=-1, P=0.0):
			cnt[0] += 1
			ret = '\"name\":\"%s\",\"action\":\"%d\",\"P\":\"%.1f\",\"N\":\"%d\",\"Q\":\"%.2f\"' \
			      % (str(cnt[0]), action, P * 100, node.N(), node.Q())
			if node.N() > 1 and not node.is_leaf():
				ret += ',\"children\":['
				for action, P, son in node.edges():
					if not son.is_NULL():
						if ret[-1] != '[':
							ret += ','
						_game = game.copy()
						_game.move(action)
						ret += dfs(cnt, son, _game, action, P)
				ret += ']'
			return '{' + ret + '}'

		with open(filename, 'w') as f:
			f.write(dfs([-1], self.root, self.init_game))


def main():
	pass


if __name__ == '__main__':
	main()
