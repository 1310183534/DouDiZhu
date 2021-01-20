import pickle
import torch
import collections


def model_equal(a: collections.OrderedDict, b: collections.OrderedDict):
	if a.keys() != b.keys():
		return False
	for layer in a.keys():
		if a[layer].shape != b[layer].shape:
			return False
		if (a[layer] != b[layer]).any():
			return False
	return True


with open('model.pkl', 'rb') as f:
	a = torch.load(f)

with open('model3.pkl', 'rb') as f:
	b = torch.load(f)

c = collections.OrderedDict()

print(model_equal(a, c))
