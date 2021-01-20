import paramiko
import os
import shutil
from numpy.random import random
import time
import torch
import collections


class Node:
	def __init__(self, hostname, port, username, password):
		self.hostname = hostname
		self.port = port
		self.username = username
		self.password = password

	def __repr__(self):
		return '<%s:%d>' % (self.hostname, self.port)

	def connect(self):
		client = paramiko.SSHClient()
		client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		client.connect(hostname=self.hostname, port=self.port, username=self.username, password=self.password)
		sftp = client.open_sftp()
		return client, sftp


train_node = Node('166.111.121.42', 4721, 'tiansuan', 'password')
gen_nodes = [Node('166.111.121.42', 4719, 'tiansuan', 'password'),
             Node('166.111.121.42', 4725, 'tiansuan', 'password'),
             Node('166.111.121.42', 4726, 'tiansuan', 'password')]
project_path = '/home/tiansuan/zys/lord'


def output_err(err):
	err_str = err.read().decode('utf-8')
	if err_str:
		print('stderr:', err_str)


def safe_execute(func, *args, **kwargs):
	try:
		func(*args, **kwargs)
	except Exception as e:
		print(e)


def get_data(node):
	print('=-=-=-=-=-=-=-=-=-=')
	print('get data')
	print(node)

	_time = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
	print(_time)

	client1, sftp1 = node.connect()
	_, out, err = client1.exec_command('cd %s/gen/data; ls | wc -l;' % project_path)
	generated = int(out.read().decode('utf-8'))
	if generated > 0:
		_, out, err = client1.exec_command('cd %s/gen; tar -cf data.tar data/*; rm -r data; mkdir data' % project_path)
		output_err(err)

		sftp1.get(project_path + '/gen/data.tar', 'gen/data.tar')
		os.system('cd gen && tar -xf data.tar data')

		file_names = os.listdir('gen/data')
		print('recv:', len(file_names), 'files')

		for file_name in file_names:
			time_file_name = _time + '_' + file_name
			if random() <= 0.01:
				shutil.copy('gen/data/' + file_name, 'gen/test/' + time_file_name)
			else:
				shutil.copy('gen/data/' + file_name, 'gen/train/' + time_file_name)
			if random() <= 0.001:
				shutil.copy('gen/data/' + file_name, 'gen/sample/' + time_file_name)

		os.system('cd gen && tar -cf test.tar test && tar -cf train.tar train')

		client2, sftp2 = train_node.connect()
		sftp2.put('gen/test.tar', project_path + '/gen/test.tar')
		sftp2.put('gen/train.tar', project_path + '/gen/train.tar')
		_, out, err = client2.exec_command('cd %s/gen; tar -xmf test.tar test; tar -xmf train.tar train; rm test.tar train.tar;' % project_path)
		output_err(err)
		os.system('cd gen && rm -r data test train data.tar test.tar train.tar && mkdir data test train')
		client2.close()
		sftp2.close()

	client1.close()
	sftp1.close()
	print('done.')
	print('=-=-=-=-=-=-=-=-=-=')


_paths = ['gen', 'gen/data', 'gen/test', 'gen/train', 'gen/sample', 'save']
for _path in _paths:
	if not os.path.isdir(_path):
		os.mkdir(_path)

_models = [_file for _file in os.listdir('save') if 'model_' in _file and '.pkl' in _file]
if not _models:
	_last_time = '000000000000'
	_last_model = collections.OrderedDict()
else:
	_models.sort()
	_last_time = _models[-1].split('_')[-1].split('.')[0]
	with open('save/%s' % _models[-1], 'rb') as f:
		_last_model = torch.load(f)


# _last_time = 'YYYY-mm-dd-HH-MM'

def equal(model_a, model_b):
	if model_a.keys() != model_b.keys():
		return False
	for layer in model_a.keys():
		if model_a[layer].shape != model_b[layer].shape:
			return False
		if (model_a[layer] != model_b[layer]).any():
			return False
	return True


def save_model(_time):
	global _last_time
	if _last_time[:8] != _time[:8]:  # not in one day.
		_last_time = _time
		return True
	hour0 = int(_last_time[8:10])
	hour1 = int(_time[8:10])
	if hour0 + 4 < hour1:  # save every 4 hour, save 6 times in one day.
		_last_time = _time
		return True
	return False


def put_model(node):
	client, sftp = node.connect()
	print('uploading:', node)
	sftp.put('save/model.pkl', project_path + '/save/_model.pkl')
	_, out, err = client.exec_command('cd %s/save; mv _model.pkl model.pkl;' % project_path)
	output_err(err)
	client.close()
	sftp.close()


def get_model():
	print('=-=-=-=-=-=-=-=-=-=')
	print('get model')
	print(train_node)
	_time = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
	print(_time)

	client, sftp = train_node.connect()

	_, out, err = client.exec_command('cd %s/save; cp model.pkl _model.pkl;' % project_path)
	output_err(err)

	print('downloading...')
	sftp.get(project_path + '/save/_model.pkl', 'save/model.pkl')
	client.close()
	sftp.close()
	print('done.')

	if save_model(_time):
		shutil.copy('save/model.pkl', 'save/model_%s.pkl' % _time)
		print('model_%s.pkl saved.' % _time)

	global _last_model
	with open('save/model.pkl', 'rb') as f:
		model = torch.load(f)
	if not equal(model, _last_model):
		for node in gen_nodes:
			safe_execute(put_model, node)
		_last_model = model
	print('done.')
	print('=-=-=-=-=-=-=-=-=-=')


def work():
	for node in gen_nodes:
		safe_execute(get_data, node)
	safe_execute(get_model)


def main():
	while True:
		_time = time.time()
		work()
		while time.time() - _time < 1800:
			time.sleep(10)


if __name__ == '__main__':
	main()
