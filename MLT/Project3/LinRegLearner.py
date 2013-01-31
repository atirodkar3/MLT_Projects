import numpy as np

class LinRegLearner:

	def __init__(self):
		self.Xtrain = None
		self.Ytrain = None
		self.slp = None
		self.con = None

	def addEvidence(self, Xtrain, Ytrain):
		self.Xtrain = Xtrain
		self.Ytrain = Ytrain
		#Get your data sequentially in an array to run linalg on it...
		# http://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html
		A = np.hstack([self.Xtrain, np.ones((len(self.Xtrain[:, 0]), 1))])
		#Using Numpy. linalg to obtain w[0] and w[1] We return the line plotted like that
		#http://glowingpython.blogspot.com/2012/03/linear-regression-with-numpy.html w replaced by slp
		slp = np.linalg.lstsq(A, self.Ytrain)[0]
		self.slp = slp[:-1]
		self.con = slp[-1]
		
		
	def query(self, Xtest):
		# Return the corresponding value of Y for each value in Xtest considering the slope obtained from the linear regression function computed above.
		Yret = Xtest.dot(self.slp) + self.con
		return Yret