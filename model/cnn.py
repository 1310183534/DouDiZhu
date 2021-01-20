from torch import nn
import torch
from torch.nn import Parameter


def conv_layer(input_channel, output_channel, kernel_size=3, bias=False):
	return nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias)


class FreezeBatchNorm2d(nn.BatchNorm2d):
	def __init__(self, *args, **kwargs):
		super(FreezeBatchNorm2d, self).__init__(*args, **kwargs)
	
	def forward(self, x):
		return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class TLU(nn.Module):
	def __init__(self, num_features, inplace=True):
		super(TLU, self).__init__()
		self.inplace = inplace
		self.num_features = num_features
		self.tau = Parameter(torch.zeros([1, num_features, 1, 1]))

	def extra_repr(self):
		return 'num_features={num_features}'.format(**self.__dict__)
	
	def forward(self, x):
		return torch.max(x, self.tau)


class FRN(nn.Module):
	def __init__(self, num_features, eps=1e-6):
		super(FRN, self).__init__()
		
		self.num_features = num_features
		self.init_eps = eps
		
		self.weight = Parameter(torch.ones([1, num_features, 1, 1]))
		self.bias = Parameter(torch.zeros([1, num_features, 1, 1]))
	
	def extra_repr(self):
		return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)
	
	def forward(self, x):
		# TODO 专门实现bp
		nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
		x = x * torch.rsqrt(nu2 + self.eps)
		x = self.weight * x + self.bias
		return x


def norm_layer(channel):
	return nn.BatchNorm2d(channel, momentum=None)  # TODO use DRQN training algorithm
	# return FRN(channel)


def ReLU_layer():
	return nn.LeakyReLU(inplace=True)


def TLU_layer(channel):
	return nn.LeakyReLU(inplace=True)
	# return TLU(channel)


def push_down(conv, bn):
	with torch.no_grad():
		std = torch.sqrt(bn.running_var + bn.eps)
		conv.weight.data /= std.view([-1, 1, 1, 1])
		conv.bias.data -= bn.running_mean
		conv.bias.data /= std
		bn.running_mean.fill_(0)
		bn.running_var.fill_(1 - bn.eps)
		bn.num_batches_tracked.fill_(0)


class ConvBlock(nn.Module):
	def __init__(self, input_channel, output_channel, kernel_size=3):
		super(ConvBlock, self).__init__()
		self.conv = conv_layer(input_channel, output_channel, kernel_size, bias=True)
		self.bn = norm_layer(output_channel)
		self.ReLU = TLU_layer(output_channel)
	
	def push_down(self):
		push_down(self.conv, self.bn)
		# pass
	
	def forward(self, x):
		out = self.conv(x)
		out = self.bn(out)
		out = self.ReLU(out)
		return out


class ResidualBlock(nn.Module):
	def __init__(self, channel):
		super(ResidualBlock, self).__init__()
		self.conv0 = conv_layer(channel, channel, bias=True)
		self.bn0 = norm_layer(channel)
		self.ReLU0 = TLU_layer(channel)
		self.conv1 = conv_layer(channel, channel, bias=True)
		self.bn1 = norm_layer(channel)
		self.ReLU1 = TLU_layer(channel)
	
	def push_down(self):
		push_down(self.conv0, self.bn0)
		push_down(self.conv1, self.bn1)
		pass
	
	def forward(self, x):
		out = x
		out = self.conv0(out)
		out = self.bn0(out)
		out = self.ReLU0(out)
		out = self.conv1(out)
		out = self.bn1(out)
		return self.ReLU1(x + out)
