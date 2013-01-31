from qstkutil import DataAccess as da
from qstkutil import qsdateutil as du
from scipy.stats.stats import pearsonr
from qstkfeat.features import featMA, featBollinger, featMomentum
from qstkfeat.classes import class_fut_ret
import qstkfeat.featutil as featu
import RandomForestLearner as RandomForestLearner
import KNNLearner as KNNLearner
import LinRegLearner as LinRegLearner
import numpy as np
import matplotlib.pyplot as plt
import csv
import timeit
import datetime as dt

def runLinRegExperiment(data=""):
	for k in lr_axis:
		learner_k = lrl.LinRegLearner()
		learner_k.addEvidence(xtrain, ytrain)
		Yret_k = learner_k.query(xtest)
		Ymatrix_k = np.ndarray(shape=(2,len(Yret_k)))
		Ymatrix_k[0,:]=Yret_k[:]
		Ymatrix_k[1,:]=ytest[:]
		corrcoeffs[k-1]= np.corrcoef(Ymatrix_k)[0,1]


	



symbols = ['SINE_FAST']
dataobj = da.DataAccess('Yahoo')
dtStart = dt.datetime(2008,1,1)
dtEnd = dt.datetime(2008,12,31)
timestamps = du.getNYSEdays(dtStart, dtEnd, dt.timedelta(hours = 16) )

Keys = ['open', 'high', 'low', 'close', 'volume']
lldata = dataobj.get_data( timestamps, symbols, Keys )
dData = dict(zip(Keys, lldata))
Features = [featMA, featBollinger, featMomentum, class_fut_ret]
ldArgs = [ {'lLookback':20, 'bRel':True},\
{},\
{},\
{'i_lookforward':5}] 
llfeatures = featu.applyFeatures( dData, Features, ldArgs )
rLearner = RandomForestLearner.RandomForestLearner(_k = 15)
kLearner = KNNLearner.KNNLearner(3)
LinRegLearner = LinRegLearner.LinRegLearner()
Xtrain = np.zeros( ( np.size(llfeatures[1].values) - 20, 3) )
Ytrain = np.zeros( np.size(llfeatures[1].values) - 20)




def runKNNExperiment(ma,data=""):		
	for k in k_axis:

		learner_k = knnl.KNNLearner(k, method = "mean")
		learner_k.addEvidence(xtrain, ytrain)
		Yret_k = learner_k.query(xtest)
		Ymatrix_k = np.ndarray(shape=(2,len(Yret_k)))
		Ymatrix_k[0,:]=Yret_k[:]
		Ymatrix_k[1,:]=ytest[:]
		corrcoeffs[k-1]= np.corrcoef(Ymatrix_k)[0,1]
		rms_array[k - 1] = np.sqrt(np.mean(np.subtract(ytest, Yret_k)**2))
		
for i in range(20, np.size(llfeatures[1].values)):
	Xtrain[i-20][0] = llfeatures[0].values[i]
	Xtrain[i-20][1] = llfeatures[1].values[i]
	Xtrain[i-20][2] = llfeatures[2].values[i]
	Ytrain[i-20] = lldata[4].values[i]
print "Training Data" ,
rLearner.addEvidence(Xtrain, Ytrain)
kLearner.addEvidence(Xtrain, Ytrain)
LinRegLearner.addEvidence(Xtrain, Ytrain)
print "Training Over"
print "Testing Begun",
dtStart = dt.datetime(2008,1,1)
dtEnd = dt.datetime(2008,12,31)
timestamps = du.getNYSEdays(dtStart, dtEnd, dt.timedelta(hours = 16) )
Keys = ['open', 'high', 'low', 'close', 'volume']
lldata = dataobj.get_data( timestamps, symbols, Keys )
dData = dict(zip(Keys, lldata))
Features = [featMA, featBollinger, featMomentum, class_fut_ret]
ldArgs = [ {'lLookback':20, 'bRel':True},\
{},\
{},\
{'i_lookforward':5}] 
llfeatures = featu.applyFeatures( dData, Features, ldArgs )
Xtest = np.zeros( ( np.size(llfeatures[1].values) - 20, 3) )
Ytest = np.zeros( np.size(llfeatures[1].values) - 20)

def RandomForestLearnerExperiment(data="", noleaf = 1):
	for k in range(1,101):
		learner_k = rfl.RandomForestLearner(k, noleaf)
		learner_k.addEvidence(xtrain, ytrain)
		Yret_k = learner_k.query(xtest)
		ccrfl[k - 1]= np.corrcoef(ytest, Yret_k)[0, 1]
		rms_array[k - 1] = np.sqrt(np.mean(np.subtract(ytest, Yret_k)**2))
for i in range(20, np.size(llfeatures[1].values)):
	Xtest[i-20][0] = llfeatures[0].values[i]
	Xtest[i-20][1] = llfeatures[1].values[i]
	Xtest[i-20][2] = llfeatures[2].values[i]
	Ytest[i-20] = lldata[4].values[i]
print "Random Forests Training"
YPredictedR = rLearner.query(Xtest)
plt.clf()
plt.cla()
YPredictedR = np.reshape(YPredictedR,(len(YPredictedR)))
ActualY1 = np.reshape(Ytest,(len(Ytest)))
plt.xlabel('Time Elapsed in Days ')
plt.ylabel('Stock Value')
pred, = plt.plot(range(len(YPredictedR)), YPredictedR, 'r', linewidth = 1)
act, = plt.plot(range(len(YPredictedR)), ActualY1, 'b-', linewidth = 1)
plt.legend([pred, act], ["Predicted", "Actual"])
plt.title('Prediction By ' + "RandomForest")
plt.savefig("RandomForest" + '.pdf', format='pdf')
print "Random Forests Done"
print "KNN Training"
YPredictedK = kLearner.query(Xtest)
plt.clf()
plt.cla()
YPredictedK = np.reshape(YPredictedK,(len(YPredictedK)))

plt.xlabel('Time Elapsed in Days ')
plt.ylabel('Stock Value')
pred, = plt.plot(range(len(YPredictedK)), YPredictedK, 'r', linewidth = 1)
act, = plt.plot(range(len(YPredictedK)), ActualY1, 'b-', linewidth = 1)
plt.legend([pred, act], ["Predicted", "Actual"])

plt.title('Prediction By ' + "KNN")
plt.savefig("KNN" + '.pdf', format='pdf')
print "KNN Done"
print "Linear Regression Training"
YPredictedL = LinRegLearner.query(Xtest)
plt.clf()
plt.cla()
YPredictedL = np.reshape(YPredictedL,(len(YPredictedL)))

plt.xlabel('Time Elapsed in Days ')
plt.ylabel('Stock Value')
pred, = plt.plot(range(len(YPredictedL)), YPredictedL, 'r', linewidth = 1)
act, = plt.plot(range(len(YPredictedL)), ActualY1, 'b-', linewidth = 1)
plt.legend([pred, act], ["Predicted", "Actual"])
plt.title('Prediction By ' + "LinearReg")
plt.savefig("LinearReg" + '.pdf', format='pdf')
print "Linear Regression Done"







