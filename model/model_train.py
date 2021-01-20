from .model import Model
import torch
from log import log
from torch import nn
from torch import optim
from config import config


class Loss(nn.Module):
	def __init__(self, model):
		super(Loss, self).__init__()
		self.model = model
		self.vloss = nn.MSELoss()
		self.ploss = nn.CrossEntropyLoss()

	def forward(self, histories, lengths, rnn_states, resets, vs_label, ps_label):
		batch, _, _, w, h = histories.shape
		vs, ps, _ = self.model(histories, lengths, rnn_states, resets)
		vloss = self.vloss(vs, vs_label.view(-1, 1))
		ploss = self.ploss(ps, ps_label)
		
		tot, correct = 0, 0
		for p, p_label in zip(torch.argmax(ps, dim=1).cpu().numpy(), ps_label.cpu().numpy()):
			tot += 1
			if p == p_label:
				correct += 1
		acc = torch.tensor(correct / tot, dtype=vloss.dtype, device=ploss.device)
		return torch.stack((vloss, ploss, acc)).view(1, -1)


class CalcRnnStates(nn.Module):
	def __init__(self, model):
		super(CalcRnnStates, self).__init__()
		self.model = model

	def forward(self, histories, lengths, rnn_states, resets):
		return self.model.calc_rnn_states(histories, lengths, rnn_states, resets)


class ModelTrain(Model):
	def __init__(self, name):
		super(ModelTrain, self).__init__(name)
		self.optimizer = optim.SGD(self.model.parameters(), lr=0.0, momentum=config.momentum, weight_decay=config.l2)
		
		if config.device_ids:
			self.multigpu_loss = nn.DataParallel(Loss(self.model), device_ids=config.device_ids)
			self.multigpu_calc_rnn_states = nn.DataParallel(CalcRnnStates(self.model), device_ids=config.device_ids)

	def learning_rate(self, learning_rate):
		log('learning_rate: %f' % learning_rate)
		self.optimizer.param_groups[0]["lr"] = learning_rate

	def calc_rnn_states(self, histories, lengths, rnn_states, resets=None):
		self.model.eval()
		with torch.no_grad():
			if resets is None:
				resets = torch.zeros(histories.shape[:2], dtype=torch.float)
			rnn_states = self.multigpu_calc_rnn_states(histories, lengths, rnn_states, resets)
		return rnn_states

	def learn(self, histories, lengths, rnn_states, resets, vs_label, ps_label):
		self.model.train()
		self.optimizer.zero_grad()
		loss = self.multigpu_loss(histories, lengths, rnn_states, resets, vs_label, ps_label)
		vloss = torch.mean(loss[:, 0])
		ploss = torch.mean(loss[:, 1])
		acc = torch.mean(loss[:, 2])
		(vloss + ploss).backward()
		nn.utils.clip_grad.clip_grad_norm_(self.parameters(), 5)
		self.optimizer.step()
		return vloss.item(), ploss.item(), acc.item()

	def loss(self, histories, lengths, rnn_states, resets, vs_label, ps_label):
		self.model.eval()
		with torch.no_grad():
			loss = self.multigpu_loss(histories, lengths, rnn_states, resets, vs_label, ps_label)
			vloss = torch.mean(loss[:, 0])
			ploss = torch.mean(loss[:, 1])
			acc = torch.mean(loss[:, 2])
		return vloss.item(), ploss.item(), acc.item()

	def push_down(self):
		def dfs(module):
			for child in module.children():
				if issubclass(type(child), nn.Module):
					if 'push_down' in dir(child):
						child.push_down()
					dfs(child)

		dfs(self.model)
