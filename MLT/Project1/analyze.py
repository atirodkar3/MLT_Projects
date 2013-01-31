import sys
import csv
import numpy as np
import pandas
import datetime as dt
import matplotlib.pyplot as plt
import qstkutil.qsdateutil as du
import qstkutil.tsutil as tsu
import qstkutil.DataAccess as da
from pylab import *

csvval = sys.argv[1]
spxstk = sys.argv[2]
csvdata = np.loadtxt(csvval, dtype={'names': ['year', 'month', 'day', 'amount'], 'formats': ['i4', 'i4', 'i4', 'f8']}, delimiter=',')
csvlen = len(csvdata)
start_amount = csvdata['amount'][0]
starter = dt.datetime(csvdata['year'][0], csvdata['month'][0], csvdata['day'][0])
ender = dt.datetime(csvdata['year'][csvlen - 1], csvdata['month'][csvlen - 1], csvdata['day'][csvlen - 1], 16)
timeofday = dt.timedelta(hours = 16)
timestamp = du.getNYSEdays(starter, ender, timeofday)
dataobj = da.DataAccess('Yahoo')
close = dataobj.get_data(timestamp, [spxstk], 'close')
timestamp = close.index
pricedat = close.values
num_shares = start_amount/pricedat[0]
normalized = []
for i in pricedat:
    normalized.append(i * num_shares)

plt.cla()
plt.clf()
plt.plot(timestamp, csvdata['amount'], linewidth = 2)
plt.plot(timestamp, normalized, linewidth = 2)
plt.legend(('Portfolio', spxstk), 'upper left')
plt.ylabel('Portfolio Value')
plt.xlabel('Date')
plt.xticks(size='xx-small')
plt.show()
plt.savefig('chart.pdf', format = 'pdf')
	
	
pdrets = (csvdata['amount'][1:] / csvdata['amount'][0:-1]) - 1
ptotrets = (csvdata['amount'][len(csvdata['amount'])-1] / csvdata['amount'][0]) - 1
pavegardret = np.mean(pdrets)
pstd = np.std(pdrets)
psharpe = ((252**.5) * pavegardret) / pstd
print '\n'
print 'Final Portfolio Values'
print 'Total Return on Investment: ', float(ptotrets)
print 'Standard Deviation of Stock:', pstd
print 'Sharpe Ratio Obtained :', psharpe

spdrets = (pricedat[1:]/pricedat[0:-1]) - 1
sptotrets = (pricedat[len(pricedat)-1] / pricedat[0]) - 1
spavegardret = np.mean(spdrets)
spstd = np.std(spdrets)
spsharpe = ((252**.5) * spavegardret) / spstd
print '\n\n'
print 'Final Values Obtained For ',spxstk
print 'Total Return on Investment: ', float(sptotrets)
print 'Standard Deviation of Stock:', spstd
print 'Sharpe Ratio Obtained :', spsharpe





