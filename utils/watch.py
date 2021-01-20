from time import time


class Watch:
	def __init__(self):
		self.time = None
		self.able = True

	def enable(self):
		self.able = True

	def disable(self):
		self.able = False

	def reset(self):
		self.time = time()

	def print(self, sign=None, reset=False):
		__time = time()
		if self.able:
			if sign is not None:
				print('%s: %.3fs' % (sign, __time - self.time))
			else:
				print('%.3fs' % (__time - self.time))
		if reset:
			self.time = __time
