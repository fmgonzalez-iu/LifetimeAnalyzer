#!/usr/local/bin/python3
import sys
from scipy import loadtxt, hstack
from datetime import date
import numpy as np

def weighted_mean(vals,errs):
	# Weighted Mean, provide a np array of values and uncertainties
	
	wts = 1./(errs*errs)	
	ltAvg = np.sum(vals*wts)
	ltWts = np.sum(wts)
	
	return ltAvg/ltWts, np.sqrt(1/ltWts)
	
if __name__=='__main__':
	# These are the numbers we load:
	#lists = np.array([4230,4711,5713,7332,9768,11715])
	#lists = np.array([4305,4711,5726,7332,9768,11715])
	#lists = np.array([4230,4711,5726,7332,9768,11715])
	lists = np.array([4230,4711,5713,7332,9768,11715])
	l2017 = (lists<9600)
	print(l2017)
	l2018 = (lists>= 9600)
	print(l2018)
	#-------------------------------------------------------------------
	# Should just do python ./get_likelihood_lifetime.py PATH METH PEAKS THR 
	if len(sys.argv) < 2:
		sys.exit("use python ./get_likelihood_lifetime.py PATH METH PEAKS THR")
		
	path = sys.argv[1]
	
	if len(sys.argv) >= 3:
		meth = sys.argv[2]
		meth += ' '
	else:
		meth = 'coinc '
	
	if len(sys.argv) >= 4:
		peaks = sys.argv[3]
		peaks += ' '
	else:
		peaks = '23 '
	if len(sys.argv) >= 5:
		thr = sys.argv[4]
		thr += ' '
	else:
		thr = 'low'
	
	analyzer = 'Frank '
	typ = 'global '
	
	lts = np.zeros(len(lists))
	unc = np.zeros(len(lists))
	for i,r in enumerate(lists):
		# Filename switch
		v = 4
		if meth == 'coinc ':
			fName = path+'outputCoinc'+str(r)+'.txt'
			v = 3 # Coincidence doesn't have an 'eff' variable
		elif meth == 'pmt12 ':
			fName = path+'outputSing'+str(r)+'.txt'
		elif meth == 'pmt1 ':
			fName = path+'outputPMT1'+str(r)+'.txt'
		elif meth == 'pmt2 ':
			fName = path+'outputPMT2'+str(r)+'.txt'
		else:
			continue
		try:	
			samples_full = np.loadtxt(fName) # load data from our file
		except IOError:
			unc[i] = np.inf
			print("Cannot Load Run",r)
			continue	
		
		nBreaks = int((np.shape(samples_full)[1]-1)/v)
		
		samples = []
		for s in samples_full:
			if np.isfinite(s[v*nBreaks+1]):
				samples.append(s)
		samples = np.array(samples)
		if  len(samples) == 0:
			print("Run",r,"has unconstrained initial bounds! Doing a 'best guess'!")
			tau_0 = np.average(samples_full[:,0])
			unc_low = np.percentile(samples_full[:,0],16)
			unc_hi  = np.percentile(samples_full[:,0],84)
			lts[i] = tau_0
			unc[i] = (unc_hi-unc_low)/2.
			continue
		# take means of posterior distributions as a "best fit"
		tau_0  = np.average(samples[:,0])
		print(tau_0)
		# We've found chi2 values so we can use the min as "best fit"
		chi2_min = np.amin(samples[:,v*nBreaks+1])
		chi2_max = np.amax(samples[:,v*nBreaks+1])
		chi2_delta = chi2_max-chi2_min
		chi2_min_ind = np.where(samples[:,v*nBreaks+1]==chi2_min)[0]
		if len(chi2_min_ind) > 1:
			chi2_min_ind = chi2_min_ind[0]
	
		tau_c = samples[chi2_min_ind,0]
		lts[i] = tau_0
		# uncertainty from percentiles
		unc_low = np.percentile(samples[:,0],16)
		unc_hi  = np.percentile(samples[:,0],84)
		
		# and take the average of the difference as unc. on tau
		unc[i] = (unc_hi-unc_low)/2.
	
	print(lts,unc)
	# Now calculate 2017, 2018, and both lifetimes
	l7,u7 = weighted_mean(lts[l2017],unc[l2017])
	l8,u8 = weighted_mean(lts[l2018],unc[l2018])
	lc,uc = weighted_mean(lts,unc)
	
	# and write out into our file for Adam
	print("Writing these lifetimes for ATH!")
	today  = date.today()
	todayS = today.strftime("%m-%d-%Y")	
	fileName = "FMG_Lifetimes-"+todayS+".txt"
	outfile = open(fileName,"a")
	# Format from Adam
	#outfile.write("Analyzer Type Year Peaks Method Threshold tau_raw etau_raw tau_rde etau_rde\n")
	
	write_2017 = analyzer+typ+'2017 '+peaks+meth+thr+str(l7)+' '+str(u7)+' '+'-1 -1\n'
	outfile.write(write_2017)
	write_2018 = analyzer+typ+'2018 '+peaks+meth+thr+str(l8)+' '+str(u8)+' '+'-1 -1\n'
	outfile.write(write_2018)
	write_comb = analyzer+typ+'comb '+peaks+meth+thr+str(lc)+' '+str(uc)+' '+'-1 -1\n'
	outfile.write(write_comb)
