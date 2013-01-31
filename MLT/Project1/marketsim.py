import sys
import csv
import qstkutil.qsdateutil as du
import qstkutil.tsutil as tsu
import qstkutil.DataAccess as da
import qstkfeat.features as feat
import datetime as dt
import numpy as np

secondts = None
firstts = None

csv_writer = csv.writer(open(sys.argv[3], 'wb'))

# Values Obtained From Command Line are treated as sys.argv[]. Convert into datatype with number specifying length. we strip the extra spaces from the string variables. 
csvdata = np.loadtxt(sys.argv[2],delimiter=',',dtype={'names': ('year', 'month', 'day', 'symbol', 'transaction', 'amount'), 'formats': ('i4', 'i4', 'i4', 'S10', 'S10', 'i4')})
csvdata['symbol'] = np.char.strip(csvdata['symbol'])
csvdata['transaction'] = np.char.strip(csvdata['transaction'])
csvdata = np.sort(csvdata)
tot_invest = float(sys.argv[1])
dataobj = da.DataAccess('Yahoo')
'''
symbols = dataobj.get_symbols_from_list("sp5002012")
starter = dt.datetime(2011,1,1)
ender = dt.datetime(2011,12,31)
timeday=dt.timedelta(hours=16)
timestamps = du.getNYSEdays(starter,ender,timeday)
close = dataobj.get_data(timestamps, symbols, "close")
'''
simulation = dict()



for count in csvdata:
	timestamp = du.getNextNNYSEdays( dt.datetime(count['year'],count['month'],count['day']),1,dt.timedelta(hours=16) )[0]
	if(firstts == None):
		firstts = timestamp
	if (secondts != None) and (secondts != timestamp) :
		temp = timestamp - dt.timedelta(days = 1)
		if(firstts <= temp):

			totaltime = du.getNYSEdays(firstts,temp, dt.timedelta(hours=16))
			sym = [k for k,v in simulation.iteritems()]
			data = dataobj.get_data(totaltime, sym, "close")
			totaltime = data.index
			data = data.values
			# Values Extracted from QSTK and then Modified as per the orders on the matching days from timestamp
			for ix in range(len(totaltime)):
				val = tot_invest
			for ix in range(len(sym)):
				val += (data[:,ix] * simulation[sym[ix]])
			ix = 0
			
			#Write to Values.csv
			for match in totaltime:
				csv_writer.writerow([str(match.year),str(match.month),str(match.day),str(val[ix])])
				ix += 1
			
			
		firstts = timestamp
	
	
	#First Time Stock Used.. Put 0 and Start the Stock
	if(simulation.get(count['symbol']) == None):
		simulation[count['symbol']] = 0
	close = np.sum(dataobj.get_data([timestamp], [count['symbol']], "close").values)
	#WHen Buying Stock, decrement investment and add to the total count
	if count['transaction'] == 'Buy':
		simulation[count['symbol']] += count['amount']
		tot_invest -= close * count['amount']
	elif count['transaction'] == 'Sell':
	#When Selling Stock, increment investment as per the multiplication of number * price and add to the stock
		simulation[count['symbol']] -= count['amount']
		tot_invest += close * count['amount']
		
		
	secondts = timestamp
if(firstts <= timestamp):
	totaltime = du.getNYSEdays(firstts,timestamp, dt.timedelta(hours=16))
	sym = [k for k,v in simulation.iteritems()]
	data = dataobj.get_data(totaltime, sym, "close")
	totaltime = data.index
	data = data.values
	# Values Extracted from QSTK and then Modified as per the orders on the matching days from timestamp
	for ix in range(len(totaltime)):
		val = tot_invest
	for ix in range(len(sym)):
		val += (data[:,ix] * simulation[sym[ix]])
	ix = 0
	
	#Write to Values.csv
	for match in totaltime:
		csv_writer.writerow([str(match.year),str(match.month),str(match.day),str(val[ix])])
		ix += 1

