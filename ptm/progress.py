import sys

class Progress:

	def __init__(self, n_iteration):
		self.n_iteration = n_iteration
		self.reverseStr = '\r'
		self.it = 0

	def update(self):
		self.it += 1
		dispStr = ('Percentage Done %3.1f%%' % (100. * self.it / self.n_iteration))
		sys.stdout.write(self.reverseStr + dispStr)

	def end(self):
		sys.stdout.write('\n')