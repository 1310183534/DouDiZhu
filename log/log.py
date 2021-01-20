class Logger:
	def __init__(self, file='log/log.txt'):
		self.file = file

	def set_file(self, file):
		self.file = file

	def __call__(self, string, end='\n'):
		print(string, end=end)
		with open(self.file, 'a') as f:
			f.write(string + end)


log = Logger()
