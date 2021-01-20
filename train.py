from model import ModelTrain as Model
from log import log
import numpy as np
from time import time
from config import config
from utils.loader import Data, Loader


def test(model, test_data):
	batch_size = pool_size = 3000
	test_loader = Loader(batch_size, pool_size, (1, 8), test_data)
	vloss_list, ploss_list, acc_list = [], [], []
	tot = 0

	print('Testing.')
	while True:
		test_loader.next(training=False)
		remain = test_loader.remain()
		if not remain:
			break

		test_loader.sample()
		test_loader.update(model.calc_rnn_states)
		vloss, ploss, acc = test_loader(model.loss)

		tot += remain
		vloss_list.append(vloss * remain)
		ploss_list.append(ploss * remain)
		acc_list.append(acc * remain)
		print('test: %.4f  %.4f  %.2f%%' % (vloss, ploss, acc * 100))
	vloss = float(np.sum(vloss_list)) / tot
	ploss = float(np.sum(ploss_list)) / tot
	acc = float(np.sum(acc_list)) / tot
	return vloss, ploss, acc


def train(model, train_data, test_data, save_interval, batch_size):
	vloss, ploss, acc = test(model, test_data)
	# vloss, ploss, acc = 100.0, 100.0, 100.0, 0.0
	min_loss = vloss + ploss

	it, epoch = 0, 0
	__time = time()

	log('epoch: %d  %.5f  %.5f  %.5f  %.3f%%' % (epoch, vloss, ploss, min_loss, acc * 100))
	pool_size = batch_size * 16
	train_loader = Loader(batch_size, pool_size, (14, 18), train_data)
	while True:
		it += 1
		# from utils import watch4
		# watch4.reset()
		if train_loader.next(training=True):
			epoch += 1
			test_data.reload()  # new epoch, reload train data & test data
			best_model = Model(config.name)
			best_model.restore()
			vloss, ploss, acc = test(best_model, test_data)
			min_loss = vloss + ploss
			log('epoch: %d  %.5f  %.5f  %.5f  %.3f%%' % (epoch, vloss, ploss, min_loss, acc * 100))
		train_loader.sample()
		train_loader.update(model.calc_rnn_states)
		# watch4.print('update', reset=True)
		vloss, ploss, acc = train_loader(model.learn)
		# watch4.print('learn', reset=True)
		
		if it % 20 == 0:
			model.push_down()
		
		print('%6d: %.4f  %.4f  %.2f%%  %.2fs' % (it, vloss, ploss, acc * 100, time() - __time))
		__time = time()

		if it % save_interval == 0:
			model.save(model.name + '_t')
			vloss, ploss, acc = test(model, test_data)
			if vloss + ploss < min_loss:
				min_loss = vloss + ploss
				model.save()
			log('%d: %.5f  %.5f  %.5f  %.3f%%' % ((it + 1) // save_interval, vloss, ploss, min_loss, acc * 100))


def main():
	log.set_file('log/model.txt')
	log('\n### start ' + '\t' + config.name)

	model = Model(config.name)
	print(model.model)
	# model.restore()
	model.learning_rate(0.01)

	train_data = Data('gen/train', max_size=300000)
	test_data = Data('gen/test', max_size=3000)
	train(model, train_data, test_data, batch_size=128 * len(config.device_ids), save_interval=1000)
	# train(model, train_data, test_data, batch_size=256 * len(config.device_ids), save_interval=1000)


if __name__ == '__main__':
	main()

'''
PS: remember 1201 we change learning rate to 0.001
'''
