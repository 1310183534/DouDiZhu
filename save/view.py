import torch

with open('model.pkl', 'rb') as f:
	model = torch.load(f)

for layer in model:
	if 'bn' in layer:
		print(layer, list(model[layer].shape))
		print(model[layer])
