from mcts import MCT
from game import GameEnv, Game
from node import new_Root
from model.model import Model
from config import config
from utils.unpack import history_regularize, to_cuda
import numpy as np
import torch
from abs1 import Agent as AgentD


class Agent:
	def __init__(self, model: Model):
		self.model = model
	
	@staticmethod
	def do(fn, histories, _rnn_states):
		rnn_states = [torch.tensor(_rnn_state) for _rnn_state in _rnn_states]
		histories, lengths = history_regularize(histories)
		histories, lengths = to_cuda(histories, lengths)
		rnn_states = torch.stack(rnn_states).to(config.device)
		return fn(histories, lengths, rnn_states)
	
	def pred(self, histories, _rnn_states):
		return self.do(self.model.forward, histories, _rnn_states)
	
	def calc_rnn_states(self, histories, _rnn_states):
		return self.do(self.model.calc_rnn_states, histories, _rnn_states)
	
	def simulates(self, mcts, store_rnn_states='Never'):
		assert store_rnn_states == 'Lottery' or store_rnn_states == 'Always' or store_rnn_states == 'Never'
		
		leaves, games, rnn_states, histories = [], [], [], []
		for mct in mcts:
			leaf, game = mct.reach_leaf()
			# print(game)
			if game.game_over():
				v = game.eval()
				leaf.back_up(v)
			else:
				leaf.set_state(game.to_model_input())
				games.append(game)
				leaves.append(leaf)
				history, rnn_state = leaf.get_model_input()
				histories.append(history)
				rnn_states.append(rnn_state)
		if games:
			vs, ps, new_rnn_states = self.pred(histories, rnn_states)
			for v, p, leaf, game in zip(vs, ps, leaves, games):
				leaf.expand(game.action_list(), p)
				leaf.back_up(v[0])
			if store_rnn_states == 'Lottery':
				for leaf, new_rnn_state in zip(leaves, new_rnn_states.cpu().numpy()):
					if leaf.lottery(config.lottery_rate):
						leaf.set_rnn_state(new_rnn_state.copy())
			if store_rnn_states == 'Always':
				for leaf, new_rnn_state in zip(leaves, new_rnn_states.cpu().numpy()):
					leaf.set_rnn_state(new_rnn_state.copy())
	
	def store_rnn_states(self, roots):
		_roots = [root for root in roots if not root.have_rnn_state()]
		if _roots:
			histories, rnn_states = [], []
			for root in _roots:
				history, rnn_state = root.get_model_input()
				histories.append(np.array(history))
				rnn_states.append(rnn_state)
			new_rnn_states = self.calc_rnn_states(histories, rnn_states)
			for root, new_rnn_state in zip(_roots, new_rnn_states.cpu().numpy()):
				root.set_rnn_state(new_rnn_state.copy())
		for root in roots:
			root.delete_father_state()
	
	def analysis(self, game):
		pass


def main():
	model: Model
	
	config.set_device(1)
	env = GameEnv()
	# model = Model('model')
	model = Model('model_201912080009')
	# model = Model('model_tencent0824')
	model.forced_restore()
	# agent = Agent(model)
	agent = Agent(model)

	init = [[2, 1, 2, 1, 0, 1, 3, 1, 2, 1, 0, 1, 2, 0, 0],
	        [2, 1, 1, 2, 1, 1, 0, 3, 1, 2, 0, 2, 0, 0, 1],
	        [0, 1, 1, 1, 3, 2, 0, 0, 1, 1, 3, 1, 2, 1, 0]]
	actions = [352, 352, 353, 338, 343, 347, 123, 0, 0, 20, 22, 23, 24, 26, 0, 28, 0, 29, 0, 0, 39, 0, 0, 116, 0, 0, 76,
	           324, 0, 0, 41, 42, 0, 0, 92, 317, 320, 0, 0, 31]#, 42, 0, 0, 15, 18]
	init = np.array(init, dtype=np.int32)

	env.load(init)
	print(env)
	
	root, player = new_Root(), 2

	game = Game(env, player)
	for action in actions:
		mct = MCT(game, root)
		agent.simulates([mct], store_rnn_states='Always')
		root.root(action)
		game.move(action)
		print(game)
		print('GAUSS', game.gauss())
		print(game.curr_player(), game.my_player())

	print('====')
	print(game.curr_player())
	print(game.my_player())
	print(game.lord_player())
	print(game.hand_cards_num())
	print(game.bottom())
	print('====')
	mct = MCT(game, root)
	for cnt in range(2000):
		agent.simulates([mct], store_rnn_states='Always')
	
		# if cnt == 0:
		# 	history, rnn_state = root.get_model_input()
		# 	print(history)
		# if (cnt + 1) % 10 == 0:
		if (cnt + 1) % 10 == 0:
			print(cnt + 1)
			for action, P, son in mct.root.edges():
				print('%d: %.8f %d %.3f' % (action, P, son.N(), son.Q()))
	print('-------------------------')
	t = 1.0
	s = np.array([son.N() for action, P, son in mct.root.edges()])
	p = np.array([P for action, P, son in mct.root.edges()])
	print(s)
	print(np.mean(s))
	w = s + 0.001 - np.mean(s)
	w[w < 0] = 0
	w = (w ** t) / (w ** t).sum()
	print(w)
	print(s / s.sum())
	print(p)
	mct.json()

	
if __name__ == '__main__':
	main()
