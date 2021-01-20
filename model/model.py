import torch.nn as nn
from config import config
import torch
import torch.nn.functional as F
from .struct import Struct


class Model(nn.Module):
	def __init__(self, name):
		super(Model, self).__init__()
		self.name = name
		self.model = Struct().to(config.device)

	def calc_rnn_states(self, histories, lengths, rnn_states, resets=None):
		self.model.eval()
		with torch.no_grad():
			rnn_states = self.model.calc_rnn_states(histories, lengths, rnn_states, resets)
		return rnn_states

	def forward(self, histories, lengths, rnn_states, resets=None):
		self.model.eval()
		with torch.no_grad():
			vs, ps, rnn_states = self.model(histories, lengths, rnn_states, resets)
			ps = F.softmax(ps, dim=1)
		return vs.cpu().numpy(), ps.cpu().numpy(), rnn_states

	def save(self, name=None):
		if name is None:
			name = self.name
		torch.save(self.model.cpu().state_dict(), 'save/' + name + '.pkl')
		self.model.to(config.device)

	def forced_restore(self):
		state_dict = torch.load('save/' + self.name + '.pkl')
		self.model.load_state_dict(state_dict)
		self.model.to(config.device)

	def forced_restore_abs2(self):
		state_dict = torch.load('abs2/save/' + self.name + '.pkl')
		self.model.load_state_dict(state_dict)
		self.model.to(config.device)

	def restore(self, ignore_DeleteKeyError=False, ignore_AddKeyError=False, ignore_ShapeError=False):
		try:
			load_state_dict = torch.load('save/' + self.name + '.pkl')
		except Exception as e:
			print(e)
			print('restore() failed.')
			return
		self_state_dict = self.model.state_dict()
		for key in load_state_dict:
			if key not in self_state_dict:
				if ignore_DeleteKeyError:
					print('Model(%s).restore(): DeleteKey %s' % (self.name, key))
				else:
					raise RuntimeError('Model(%s).restore(): DeleteKeyError %s' % (self.name, key))
		for key in self_state_dict:
			if key in load_state_dict:
				shape0 = self_state_dict[key].shape
				shape1 = load_state_dict[key].shape
				if shape0 != shape1:
					if ignore_ShapeError:
						print('Model(%s).restore(): ShapeChanged %s %s -> %s' % (self.name, key, str(shape1), str(shape0)))
					else:
						raise RuntimeError(
							'Model(%s).restore(): ShapeError %s %s -> %s' % (self.name, key, str(shape1), str(shape0)))
				else:
					self_state_dict[key] = load_state_dict[key]
			else:
				if ignore_AddKeyError:
					self_state_dict[key].zero_()
					print('Model(%s).restore(): AddKey %s' % (self.name, key))
				else:
					raise RuntimeError('Model(%s).restore(): AddKeyError %s' % (self.name, key))
		self.model.load_state_dict(self_state_dict)
		self.model.to(config.device)
