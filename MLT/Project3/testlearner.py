import KNNLearner as knnl
import LinRegLearner as lrl
import numpy as np
import matplotlib.pyplot as plt
import time
import RandomForestLearner as rfl
from pylab import *

def runKNNExperiment(ma,data=""):
	print '\n'
	ma=str(ma)
	print("Data set used for KNN experiment:",data)
	data=np.genfromtxt(data,delimiter=",")
	xdata=data[:,0:2]
	ydata=data[:,2]
	numrows = data.shape[0]
	xtrain=xdata[0:0.6*numrows,:]
	ytrain=ydata[0:0.6*numrows]
	xtest=xdata[0.6*numrows:,:]
	ytest=ydata[0.6*numrows:]
	learner = knnl.KNNLearner(3,"mean")
	startTraining=time.clock()
	learner.addEvidence(xtrain, ytrain)
	trainingDuration = (time.clock()-startTraining)
	averageTrainingDuration = trainingDuration/len(xtrain)
	print("Average Training time for each instance (seconds): ",averageTrainingDuration)
	startTest=time.clock()
	Yret=learner.query(xtest)
	testDuration = (time.clock()-startTest)
	averageTestDuration = testDuration/len(xtest)
	print("Average Query time for each instance (seconds): ",averageTestDuration)
	
	plt.clf()
	plt.cla()
	plt.scatter(ytest, Yret, c ='b')
	plt.xlabel('Actual Y Value')
	plt.ylabel('Predicted Y values')
	savefig('KNN Value'+ma+'.pdf',format='pdf')
	Ymatrix= np.ndarray(shape=(2,len(Yret)))
	Ymatrix[0,:]=Yret[:]
	Ymatrix[1,:]=ytest[:]
	correlator= np.corrcoef(Ymatrix)[0,1]
	print ("Correlation Coefficient for K=3 ",correlator)

	rms = np.sqrt(np.mean(np.subtract(ytest, Yret)**2))
	print 'Root Mean Squared Error(RMS) is ' , rms
	print '\n'
	corrcoeffs=np.zeros(100)
	rms_array=np.zeros(100)
	k_axis = range(1,101)
	for k in k_axis:

		learner_k = knnl.KNNLearner(k, method = "mean")
		learner_k.addEvidence(xtrain, ytrain)
		Yret_k = learner_k.query(xtest)
		Ymatrix_k = np.ndarray(shape=(2,len(Yret_k)))
		Ymatrix_k[0,:]=Yret_k[:]
		Ymatrix_k[1,:]=ytest[:]
		corrcoeffs[k-1]= np.corrcoef(Ymatrix_k)[0,1]
		rms_array[k - 1] = np.sqrt(np.mean(np.subtract(ytest, Yret_k)**2))
		
		
	return corrcoeffs, rms_array

def runLinRegExperiment(data=""):
	print '\n'
	print 'Linear Regression:'
	print("Data set used for Linear Regression Experiment:",data)
	data=np.genfromtxt(data,delimiter=",")
	xdata=data[:,0:2]
	ydata=data[:,2]
	numrows = data.shape[0]
	xtrain=xdata[0:0.6*numrows,:]
	ytrain=ydata[0:0.6*numrows]
	xtest=xdata[0.6*numrows:,:]
	ytest=ydata[0.6*numrows:]
	learner = lrl.LinRegLearner()
	startTraining=time.clock()
	learner.addEvidence(xtrain, ytrain)
	trainingDuration = (time.clock()-startTraining)
	averageTrainingDuration = trainingDuration/len(xtrain)
	print("Average Training time for each instance (seconds): ",averageTrainingDuration)
	
	
	startTest=time.clock()
	Yret=learner.query(xtest)
	testDuration = (time.clock()-startTest)
	averageTestDuration = testDuration/len(xtest)
	print("Average Query time for each instance (seconds): ",averageTestDuration)
	Ymatrix= np.ndarray(shape=(2,len(Yret)))
	Ymatrix[0,:]=Yret[:]
	Ymatrix[1,:]=ytest[:]
	correlator= np.corrcoef(Ymatrix)[0,1]
	corrcoeffs=np.zeros(100)
	lr_axis = range(1,101)
	for k in lr_axis:

		learner_k = lrl.LinRegLearner()
		learner_k.addEvidence(xtrain, ytrain)
		Yret_k = learner_k.query(xtest)
		Ymatrix_k = np.ndarray(shape=(2,len(Yret_k)))
		Ymatrix_k[0,:]=Yret_k[:]
		Ymatrix_k[1,:]=ytest[:]
		corrcoeffs[k-1]= np.corrcoef(Ymatrix_k)[0,1]

	rms = np.sqrt(np.mean(np.subtract(ytest, Yret)**2))	
	return corrcoeffs, rms

def RandomForestLearnerExperiment(data="", noleaf = 1):
	filename=data
	data=np.genfromtxt(data,delimiter=",")
	print 'Running PERT for the given data', filename
	xdata = data[:, 0:2]
	#print xdata
	ydata = data[:, 2]
	#print ydata
	row = len(data[:, 0])
	xtrain = xdata[0:0.6 * row, :]
	ytrain = ydata[0:0.6 * row]
	xtest = xdata[0.6*row:, :]
	ytest = ydata[0.6*row:]
	m=0
	n=0
	'''
	for (i in range len(XTrain[0])):
		#print XTrain[i,:]	
		m++
	for (i in range len(YTrain)[0]):
		#print YTrain[i,:]	
		n++
		
	#print m,n;	
	m=0
	n=0
	for (i in range len(xtest[0])):
		#print xtest[i,:]
		m++
	for (i in range len(ytest[0])):
		#print ytest[i,:]	
		n++
	#print m,n;
	'''
	learner = rfl.RandomForestLearner(3, noleaf)
	startTraining=time.clock()
	learner.addEvidence(xtrain, ytrain)
	trainingDuration = (time.clock()-startTraining)
	feat=1
	split_val=2;
	for i in range(len(xtrain)):
		if xtrain[i][feat] <= split_val:
			m=m+1
		# print Xtran[i][feat];
		else:
			n=n+1
		# print split_val;
				
	averageTrainingDuration = trainingDuration/len(xtrain)
	print("Average Training time for each instance (seconds): ",averageTrainingDuration)
	print 'K = 3:'
	startTest=time.clock()
	Yret=learner.query(xtest)
	testDuration = (time.clock()-startTest)
	averageTestDuration = testDuration/len(xtest)
	print("Average Query time for each instance (seconds): ",averageTestDuration)
	corr_coefficient = np.corrcoef(ytest, Yret)[0,1]
	print 'Correlation Coefficient between Actual and Predicted = ', corr_coefficient
	rms = np.sqrt(np.mean(np.subtract(ytest, Yret)**2))
	ccrfl = np.zeros(100)
	rms_array = np.zeros(100)
	print 'Root Mean Squared Error(RMS) = ' , rms
	print '\n'
	scatter_name = 'rfl_scatterplot' + filename + '.pdf'
	plt.clf()
	plt.cla()
	plt.scatter(ytest, Yret, c ='r')
	plt.xlabel('Expected Y')
	plt.ylabel('Predicted Y')
	plt.savefig(scatter_name, format='pdf')

	for k in range(1,101):
		learner_k = rfl.RandomForestLearner(k, noleaf)
		learner_k.addEvidence(xtrain, ytrain)
		Yret_k = learner_k.query(xtest)
		ccrfl[k - 1]= np.corrcoef(ytest, Yret_k)[0, 1]
		rms_array[k - 1] = np.sqrt(np.mean(np.subtract(ytest, Yret_k)**2))

	max_k = np.argmax(ccrfl) + 1
	print "Best K = ", max_k
	print "Correlation Coefficient for the best K is: ", ccrfl[max_k - 1]
	print "RMS Error for Best K = ", rms_array[max_k - 1]
	print("------------------------------------------------------")
	print("")
	plt.clf()
	plt.cla()
	plt.plot(range(1,101), ccrfl, c = "r")
	plt.xlabel('K')
	plt.ylabel('Correlation Coefficients')
	graphname = 'CorrCoeff Vs K for' + filename + '.pdf'
	plt.savefig(graphname, format = 'pdf')

print "Running Random Forests Experiment for data-classification-prob.csv"
RandomForestLearnerExperiment('data-classification-prob.csv')

print "Running Random Forests Experiment for data-ripple-prob.csv"
RandomForestLearnerExperiment('data-ripple-prob.csv')

print "Running Random Forests Experiment for data-classification-prob.csv for Leaf Size = 16"
RandomForestLearnerExperiment('data-classification-prob.csv',noleaf = 16)

print "Running Random Forests Experiment for data-ripple-prob.csv for Leaf Size = 16"
RandomForestLearnerExperiment('data-ripple-prob.csv', noleaf =16)	
		
k_axis = range(1,101)
corrcoeffs1,rms = runKNNExperiment(1,"data-classification-prob.csv")
plt.clf()
plt.cla()
plt.plot(k_axis,corrcoeffs1)
plt.xlabel('K Obtained')
plt.ylabel('Correlation Coefficient')
savefig('plot1KNN.pdf',format='pdf')
max_k1 = np.argmax(corrcoeffs1)+1
print("Best k:",max_k1)
print("Correlation Coefficient for the best K is: ",corrcoeffs1[max_k1-1])
print 'RMS Error for the Best K is ', rms[max_k1 - 1]
print("------------------------------------------------------")
print("")

corrcoeffs2,rms1 = runKNNExperiment(2,"data-ripple-prob.csv")
plt.clf()
plt.cla()
plt.plot(k_axis,corrcoeffs2)
plt.xlabel('K Obtanied')
plt.ylabel('Correlation Coefficient')
savefig('plot2KNN.pdf',format='pdf')
max_k2 = np.argmax(corrcoeffs2)+1
print("Best k:",max_k2)
print("Correlation Coefficient for the best K is: ",corrcoeffs2[max_k2-1])
print 'RMS Error for the Best K is ', rms1[max_k2 - 1]
print("------------------------------------------------------")
print("")

k_axis = range(1,101)
corrcoeffs1, rms = runLinRegExperiment("data-classification-prob.csv")
max_k1 = np.argmax(corrcoeffs1)+1
print("Correlation Coefficient is: ",corrcoeffs1[max_k1-1])
print 'RMS Error for the Best K is ', rms
print("------------------------------------------------------")
print("")

corrcoeffs2,rms = runLinRegExperiment("data-ripple-prob.csv")
max_k2 = np.argmax(corrcoeffs2)+1
print("Correlation Coefficient is: ",corrcoeffs2[max_k2-1])
print 'RMS Error for the Best K is ', rms
print("------------------------------------------------------")
print("")


