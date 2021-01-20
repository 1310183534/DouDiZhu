import numpy as np
import time
import os
import pickle
import torch


def get_time():
	return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def version(ver, zfill=3):
	return str(ver).zfill(zfill)


def pickle_load(path):
	with open(path, 'rb') as f:
		data = pickle.load(f)
	return data


def rand_x16():
	_0 = np.random.randint(-(1 << 31), 1 << 31) + (1 << 31)
	_1 = np.random.randint(-(1 << 31), 1 << 31) + (1 << 31)
	__0 = str(hex(_0))[2:].zfill(8)
	__1 = str(hex(_1))[2:].zfill(8)
	return __0 + __1


def dump(data, path):
	if not os.path.isdir(path):
		os.makedirs(path)
	path += '/' + rand_x16() + '.pkl'
	with open(path, 'wb') as f:
		pickle.dump(data, f)


def vector(place, size=356):
	v = np.zeros(size)
	v[place] = 1
	return v


# def mean_var(x):
# 	if type(x) is torch.Tensor:
# 		x = x.cpu().detach().numpy()
# 	return np.mean(x), np.var(x)


def main():
	pass


if __name__ == '__main__':
	main()
