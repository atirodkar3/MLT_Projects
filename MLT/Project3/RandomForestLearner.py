import numpy
import math
from random import randint
from scipy.stats import mode
import LinRegLearner as LRL
import random

class node:
	def __init__(self):
		self.splitval_val = None
		self.feature = None
		self.lnode = None
		self.rnode = None

class TreeGen:
	def __init__(self, node,leaf_size = 1, max_depth = 999999):
		self.root = node
		self.max_depth = max_depth
		self.leaf_size = leaf_size
	
	
	def makeTree(self, xtrain, ytrain):
		self.xtrain = xtrain
		self.ytrain = ytrain
		self.recgenerate(self.root, self.xtrain, self.ytrain)
	
			
			
				

		
				
				
	def recgenerate(self, rootnode, xtrain , ytrain):
		lnode1 = list()
		lnode2 = list()
		rnode1 = list()
		rnode2 = list()
		fv = randint(0,numpy.array(xtrain).shape[1]-1)
		rootnode.feature = fv
		if len(xtrain) ==1:
			rootnode.splitval_val = numpy.array(xtrain)[0][fv]
			return
		x1 = randint(0, len(xtrain)-1)
		x2 = randint(0, len(xtrain)-1)
		splitval_val = ((numpy.array(xtrain)[x1][fv]) + (numpy.array(xtrain)[x2][fv]))/2
		rootnode.splitval_val = splitval_val
		for i in range(len(xtrain)):
			if xtrain[i][fv] <= splitval_val:
				lnode1.append(xtrain[i])
				lnode2.append(ytrain[i])
			else:
				rnode1.append(xtrain[i])
				rnode2.append(ytrain[i])
		
		if len(lnode1) > 0:
			rootnode.lnode = node()
			self.recgenerate(rootnode.lnode, lnode1, lnode2)
			
		if len(rnode1) > 0:
			rootnode.rnode = node()
			self.recgenerate(rootnode.rnode, rnode1, rnode2)
		return
	

	def Classifier(self, xdata, node, xtrain, ytrain):
		fv = node.feature
		splitval = node.splitval_val
		
		if len(xtrain) == 1:
			return ytrain[0]
		if self.leaf_size != 1 and len(xtrain) <= self.leaf_size:
			a = numpy.asarray(xtrain)
			b = numpy.asarray(ytrain)
			LL = LRL.LinRegLearner()
			LL.addEvidence(a,b)
			c = numpy.mean(a, axis = 0)
			return LL.query(c)
		xx=0
		
		for xx in range(len(xtrain)):
			xx=xx+1;
		#print xx;
		for xy in range(len(ytrain)):
			xy=xy+1;
		#print xx;
		'''
		for yx in range(len(Xtest)):
			yx=yx+1;
		#print xx;
		
		for yy in range(len(Ytest)):
			yy=yy+1;
		#print xx;
		'''		
		datax = list()
		datay = list()
		
		
		if self.leaf_size != 1 and len(xtrain) <= self.leaf_size:
			c1 = numpy.asarray(xtrain)
			d2 = numpy.asarray(ytrain)
			LL2 = LRL.LinRegLearner()
			LL2.addEvidence(c1,d2)
			c1 = numpy.mean(a, axis = 0)
			return LL.query(c1)		
		
		
		if xdata[fv] <= splitval:
			for ind, it in enumerate(xtrain):
				if it[fv] <= splitval:
					datax.append(it)
					datay.append(ytrain[ind])
			if node.lnode != None and len(datax) > 0:
				return self.Classifier(xdata, node.lnode, datax, datay)
			else:
				
				a = numpy.asarray(xtrain)
				b = numpy.asarray(ytrain)
				LL = LRL.LinRegLearner()
				LL.addEvidence(a,b)
				c = numpy.mean(a, axis = 0)
				return LL.query(c)
				
				c1 = numpy.asarray(xtrain)
				d2 = numpy.asarray(ytrain)
				LL2 = LRL.LinRegLearner()
				LL2.addEvidence(c1,d2)
				c1 = numpy.mean(a, axis = 0)
				return LL.query(c1)		
				
		
		
		else:
			for ind, it in enumerate(xtrain):
				if it[fv] > splitval:
					datax.append(it)
					datay.append(ytrain[ind])
			if node.rnode != None and len(datax)>0:
				return self.Classifier(xdata, node.rnode, datax, datay)
			else:
				
				a = numpy.asarray(xtrain)
				b = numpy.asarray(ytrain)
				LL = LRL.LinRegLearner()
				LL.addEvidence(a,b)
				c = numpy.mean(a, axis = 0)
				return LL.query(c)
				
				
				c1 = numpy.asarray(xtrain)
				d2 = numpy.asarray(ytrain)
				LL2 = LRL.LinRegLearner()
				LL2.addEvidence(c1,d2)
				c1 = numpy.mean(a, axis = 0)
				return LL.query(c1)		
				

			
class RandomForestLearner:
	def __init__(self, k = 3, leaf_size = 1):
		self.k = k
		self.leaf_size = leaf_size
		self.trees = list()
		
	def addEvidence(self, xtrain, ytrain):
		num_rows = len(xtrain[:, 0])
		self.xtrain = xtrain
		self.ytrain = ytrain
		xdata = numpy.zeros((int(0.6 * num_rows),2))
		ydata = numpy.zeros((int(0.6 * num_rows),1))
		
		for i in range(self.k):
			t = TreeGen(node(),self.leaf_size)
			m = random.sample(range(1, num_rows), int(0.6 * num_rows))
			for i in range(len(m)):
				xdata[i] = xtrain[m[i]]
				ydata[i] = ytrain[m[i]] 
			t.makeTree(xdata,ydata)
			self.trees.append(t)
		
	def query(self, Xtest):
		Yret = numpy.zeros(len(Xtest))
		#for i in range(len(Xtest)):
		#tree_val = numpy.zeros(self.k)
		for i in range(len(Xtest)):
			tree_val = numpy.zeros(self.k)
			for res in range(self.k):
				t = self.trees[res]
				#tree_val[res]=t.Class(Xtest[i]));

				tree_val[res] = t.Classifier(Xtest[i], t.root, t.xtrain, t.ytrain)
			Yret[i] = mode(tree_val)[0][0]

		return Yret
		
	def clean(self):
		self.trees = None
		self.xtrain = None
		self.ytrain = None

