import numpy as np

class KNNLearner():

	def __init__(self,k=3,method="mean"):
		self.k=k
		self.method=method
		
	def addEvidence(self,Xtrain,Ytrain):
		self.Xtrain=Xtrain.copy()
		self.Ytrain=Ytrain.copy()

	def query(self,Xtest):
		Yret=np.zeros(len(Xtest))
	
		for i in range(0,len(Xtest)):
			di=np.zeros(len(self.Xtrain))
			
			for j in range(0,len(self.Xtrain)):				
				for k in range(0,self.Xtrain.shape[1]):
					di[j]=di[j]+(Xtest[i,k]-self.Xtrain[j,k])*(Xtest[i,k]-self.Xtrain[j,k])
					
			indices=np.argsort(di)
			knval = np.zeros(self.k)
			knval[0:self.k] = self.Ytrain[indices[0:self.k]]

			Yret[i] = np.mean(knval)
			
		
		return Yret

				
				
