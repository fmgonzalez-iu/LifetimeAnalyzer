#!/usr/local/bin/python3
import sys
import pdb
import csv
from math import *
import numpy as np
#from statsmodels.stats.weightstats import DescrStatsW
from scipy import stats, special
from scipy.odr import *
from scipy.optimize import curve_fit, nnls
from datetime import datetime
import matplotlib.pyplot as plt

from PythonAnalyzer.plotting import *
	
def load_run_breaks(rL, nM,nList, d1 = 3, d2 = 5):
# Need to find separations for normalization -- the RH changed around a couple of times.
# Since the GV monitor wasn't changed, I'm doing a ratio test, then generating "runbreaks".
#
# This one uses MAD values as a check.


	holdSel,w,nDet1,nDet2,maxUnl =number_list_params_list(nList)
	mR = [] # Monitor Ratio
	rB = [rL[0]] # Run Breaks
	nM1 = nM[nM['det']==d1] # Normalization Monitor 1
	nM2 = nM[nM['det']==d2] # Normalization Monitor 2
	#if breaksOn:
	# Generate Ratio vector
	for run in rL:
		mon1 = nM1[nM1['run']==run]['mon']
		mon2 = nM2[nM2['run']==run]['mon']
		if len(mon1/mon2) != 1:
			print(run)
		ratio = float(mon1 / mon2)
		
		#print run, ratio
		mR.append(ratio)
	
	# plt.figure(1)
	# plt.plot(rL,mR,'b.')
	# plt.xlabel("Run Number")
	# plt.ylabel("Ratio of (detector %d) / (detector %d)" % (nDet1, nDet2))
	
	ratioList = [1]
	#rB2 = []
	# Find our runBreaks based on this
	for i, r in enumerate(mR):
		if i > 0:
			ratioList.append(mR[i]/mR[i-1])		
			#if abs(mR[i] - mR[i-1]) > 5.0:
			if (rL[i] - rL[i-1]) > 50 and rL[i] not in rB:# and rL[i] != 10864: # Run 10864 is glitchy because it doesn't have normalization
				rB.append(rL[i])
			elif (rL[i] > 8600): # Roundhouse
				if not (0.85 <= (mR[i]/mR[i-1]) <= 1.15):
					rB.append(rL[i])	
				elif ((rL[i] > 13306) and (rL[i-1] < 13306)):# or ((rL[i] > 13209) and (rL[i-1] < 13209)) or ((rL[i] >= 14000) and (rL[i-1] < 14000)):# or ((rL[i] >= 14014) and (rL[i-1] < 14014)):
					rB.append(rL[i])
			else:
				if not (0.75 <= (mR[i]/mR[i-1]) <=  1.25):
					rB.append(rL[i])
			#if runList[i] == 4392 and runList[i] not in runBreaks:
			#	runBreaks.append(runList[i])
	#else:
	#	rB.append(13306) # Approximate position of dagger change.
		#rB.append(13209) # Approximate position of dagger change.

	# for xl in rB:
		# plt.axvline(x=xl)
		
	
	# plt.figure(2)
	# plt.plot(rL,ratioList,'r.')
	# plt.xlabel("Run Number")
	# plt.ylabel("Ratio of sequential runs")
	# plt.hlines(0.85,min(rL),max(rL))
	# plt.hlines(1.15,min(rL),max(rL))
	# for xl in rB:
		# plt.axvline(x=xl)
		
	# plt.show()
	
	# Need to have a final runBreak in here as well. First make sure it's in order.	
	rB.sort()
	rB.append(max(rL))
	
	return rB



def load_hardcoded_break(rB, rL, run):
	# Hardcoded break 
	
	if min(rL) < run and max(rL) > run:
		rB.append(run)
		rB.sort()
	return rB

def filter_runs_by_mon(rL, nM = [], nDet1 = 3, nDet2 = 4, lim = 15):
	# This filters runs by cutting out normalization monitors with too few counts
	try: # Error catching!
		len(rL)
		len(nM)
	except TypeError:
		sys.exit("No runs loaded into runBreaks generation!")
	if len(rL) <= 1:
		sys.exit("Need at least 2 runs for normalization!")
		
	# Force running with -det
	mS1  = ('mon'+str(nDet1)) # Parse strings
	mES1 = ('mon'+str(nDet1)+'E') # Also error strings
	mS2  = ('mon'+str(nDet2))
	mES2 = ('mon'+str(nDet2)+'E')	
	rF = []
	for run in rL: # Generate monitor vectors
		mon1 = nM[nM['run']==run][mS1] # Load monitor values
		mon1E = nM[nM['run']==run][mES1] # Also load monitor errors
		if not (mon1 > lim):# Too few counts?
			continue
		try:
			mon2 = nM[nM['run']==run][mS2]
			mon2E = nM[nM['run']==run][mES2]
			if not (mon2 > lim): 
				continue
		except ValueError:
			mon2 = 0.0
			mon2E = 0.0
		rF.append(run)
	return rF

def filter_runs_by_error(rL, nM = [], nDet1 = 3, nDet2 = 4, nSig = 4, vb = True):
	# Since I fit each data poit to an exponential, I can look at the 
	# uncertainty in the fit. Turns out that this is a very useful filter
	# for bad runs!
		
	try: # Error catching!
		len(rL)
		len(nM)
	except TypeError:
		sys.exit("No runs loaded into runBreaks generation!")
	if len(rL) <= 1:
		sys.exit("Need at least 2 runs for normalization!")
		
	try:# Need to load detTCData, since that has the fitting time parameters!
		tcs = np.loadtxt("/home/frank/FUCKED_MCA_Analysis/detTCData.csv", delimiter=",", dtype=[('iniRun', 'i4'), ('finRun', 'i4'), ('tc1', 'f8'), ('tc2', 'f8'),('tc3','f8')])
	except IOError: # IOError means there was no file
		if vb:
			print("No time constant file found!!")
			print("Check the function filter_runs_by_error!")
		tcs = []
	except IndexError: # IndexError means there were more than len(dtype) entries in a line
		if vb:
			print("Data file has the wrong structure!!")
		tcs = []
	
	rB_guess = []
	if len(tcs) > 0:
		for t in tcs:
			if t['iniRun'] not in rB_guess:
				rB_guess.append(t['iniRun'])
		if min(rB_guess) < min(rL): # Set limits to just runs in range
			rB_guess[0] = min(rL)
		if max(rB_guess) > max(rL):
			rB_guess[-1] = max(rL)
		if max(rB_guess) < max(rL):
			rB_guess.append(max(rL))
	else:
		rB_guess = [min(rL),max(rL)]

	rB_guess.sort() # Sorting just in case
	
	print(rB_guess)
	# Now Load our detector data files
	mS1  = ('mon'+str(nDet1)) # Parse strings
	mES1 = ('mon'+str(nDet1)+'E') # Also error strings
	mS2  = ('mon'+str(nDet2))
	mES2 = ('mon'+str(nDet2)+'E')
	
	mR1  = []
	mR2  = []
	mR1E = []
	mR2E = []
	for run in rL: # Generate monitor vectors
		mon1 = nM[nM['run']==run][mS1] # Load monitor values
		mon1E = nM[nM['run']==run][mES1] # Also load monitor errors
		try:
			mon2 = nM[nM['run']==run][mS2]
			mon2E = nM[nM['run']==run][mES2]
		except ValueError: # Can put in a "no spectral correction" check
			mon2  = []
			mon2E = []
				
		# Failure to load rates (should be covered but just in case)
		if (len(mon1) == 0 or len(mon2) == 0):
			if len(mon1) == 0: 
				mR1.append(0.0)
			else:
				mR1.append(float(mon1))
			if len(mon2) == 0:
				mR2.append(0.0)
			else:
				mR2.append(float(mon2))
		else:
			mR1.append(float(mon1))
			mR2.append(float(mon2))
		# Same thing for errors
		if (len(mon1E) == 0 or len(mon2E) == 0):
			if len(mon1E) == 0:
				mR1E.append(0.0)
			else:
				mR1E.append(float(mon1E))
			if len(mon2E) == 0:
				mR2E.append(0.0)
			else:
				mR2E.append(float(mon2E))
		else:
			mR1E.append(float(mon1E))
			mR2E.append(float(mon2E))
	rL_out = []
	for i, B in enumerate(rB_guess):
		# We have rB_guesses, so now we can filter in between these
		if i == 0:
			continue
		# We're generating an effective accuracy of the monitor -- assuming
		# Gaussian statistics, what is the error we'd expect?
		#
		# Note that weighted monitors are explicitly non-Gaussian! 
		# This means that we can beat Gaussian here
		mErr1 = [] # Effective accuracy of the monitor. 
		mErr2 = []
		for j, run in enumerate(rL):
			
			if not rB_guess[i-1] <= run < B:
				continue
			# Now calculate error / expected error: (mRE / mR) / (sqrt(mR) / mR)
			mErr1.append(mR1E[j] / np.sqrt(mR1[j]))
			mErr2.append(mR2E[j] / np.sqrt(mR2[j]))
				
		if len(mErr1) > 0: # Mean and standard deviation of reduced values
			mV1 = np.mean(mErr1)
			mE1 = np.std(mErr1)
			mV2 = np.mean(mErr2)
			mE2 = np.std(mErr2)
		
		for j, run in enumerate(rL): # Double pass because this is the easiest to code
			if not rB_guess[i-1] <= run < B:
				continue
			if (((mV1 - nSig*mE1) < ( mR1E[j] / np.sqrt(mR1[j]) ) < (mV1 + nSig*mE1))
				and ((mV2 - nSig*mE2) < ( mR2E[j] / np.sqrt(mR2[j]) ) < (mV2 + nSig*mE2))):
				rL_out.append(run)
	#print rL_out
	return rL_out, rB_guess

def filter_runs_by_asym(rL, nM = [], nDet1 = 3, nDet2 = 4, nSig = 4):
	# This does an "asymmetry check" of bad runs
	
	try: # Error catching!
		len(rL)
		len(nM)
	except TypeError:
		sys.exit("No runs loaded into runBreaks generation!")
	if len(rL) <= 1:
		sys.exit("Need at least 2 runs for normalization!")
	
	# Force running with -det
	mS1  = ('mon'+str(nDet1)) # Parse strings
	mES1 = ('mon'+str(nDet1)+'E') # Also error strings
	mS2  = ('mon'+str(nDet2))
	mES2 = ('mon'+str(nDet2)+'E')
		
	rB = [rL[0]] # More error caching -- start at first

	mR1 = [] # Monitor Reduced 1
	mR2 = [] # Monitor Reduced 2
	mR1E = [] # Monitor reduced error
	mR2E = [] # Monitor reduced error
	
	for run in rL: # Generate monitor vectors
		mon1 = nM[nM['run']==run][mS1] # Load monitor values
		mon2 = nM[nM['run']==run][mS2]
		mon1E = nM[nM['run']==run][mES1] # Also load monitor errors
		mon2E = nM[nM['run']==run][mES2]
				
		# Failure to load rates (should be covered but just in case)
		if (len(mon1) == 0 or len(mon2) == 0):
			if len(mon1) == 0: 
				mR1.append(0.0)
			else:
				mR1.append(float(mon1))
			if len(mon2) == 0:
				mR2.append(0.0)
			else:
				mR2.append(float(mon2))
		else:
			mR1.append(float(mon1))
			mR2.append(float(mon2))	
		# Same thing for errors
		if (len(mon1E) == 0 or len(mon2E) == 0):
			if len(mon1E) == 0:
				mR1E.append(0.0)
			else:
				mR1E.append(float(mon1E))
			if len(mon2E) == 0:
				mR2E.append(0.0)
			else:
				mR2E.append(float(mon2E))
		else:
			mR1E.append(float(mon1E))
			mR2E.append(float(mon2E))
		
	asymList1 = [] # Asymmetries (positive and negative spectral corrections)
	asymList2 = []	
	for i, r in enumerate(rL): # Loop through, calculating asymmetries by runs
		
		if i == 0: # First run safety check
			asym1 = 0.0
			asym2 = 0.0
		elif i == len(mR1) - 1: # Last run safety check
			asym1 = 0.0
			asym2 = 0.0
		else:
			asym1 = (mR1[i+1]-mR1[i])/mR1[i] - (mR1[i]-mR1[i-1])/mR1[i]
			asym2 = (mR2[i+1]-mR2[i])/mR2[i] - (mR2[i]-mR2[i-1])/mR2[i]
		
		asymList1.append(asym1)
		asymList2.append(asym2)
		
		
	m1 = np.mean(asymList1)
	e1 = np.std(asymList1)
	m2 = np.mean(asymList2)
	e2 = np.mean(asymList2)
	print (str(m1)+str(e1))
	print (str(m2)+str(e2))
	rFilt = []
	for i, r in enumerate(rL): # Now filter asymmetries
		#if not (mean_ratio - 3*err_ratio <= r <= 1.0/(mean_ratio - 3*err_ratio))
		if (abs(asymList1[i]) > m1 + nSig*e1 and 
			abs(asymList2[i]) > m2 + nSig*e2):
			continue
		
		else:
			rFilt.append(r)
		
	return rFilt
	
def load_run_breaks_rolling(rL,nM = [], nDet1 = 3, nDet2 = 4,  w = 5,pctBreak = 0.75,nPlt = 1):
# Need to find separations for normalization -- the RH changed around a couple of times.
# Since the GV monitor wasn't changed, I'm doing a ratio test, then generating "runbreaks".
# This one does a rolling average
#
# I've changed it here so that we're using generic nM (that is, formatted from det)

	try: # Error catching!
		len(rL)
		len(nM)
	except TypeError:
		sys.exit("No runs loaded into runBreaks generation!")
	if len(rL) <= 1:
		sys.exit("Need at least 2 runs for normalization!")
	
	# Force running with -det
	mS1  = ('mon'+str(nDet1)) # Parse strings
	mES1 = ('mon'+str(nDet1)+'E') # Also error strings
	mS2  = ('mon'+str(nDet2))
	mES2 = ('mon'+str(nDet2)+'E')
		
	rB = [rL[0]] # More error caching -- start at first

	mR1 = [] # Monitor Reduced 1
	mR2 = [] # Monitor Reduced 2
	mR1E = [] # Monitor reduced error
	mR2E = [] # Monitor reduced error
	mR = []	
	
	for run in rL: # Generate monitor vectors
		mon1 = nM[nM['run']==run][mS1] # Load monitor values
		mon2 = nM[nM['run']==run][mS2]
		mon1E = nM[nM['run']==run][mES1] # Also load monitor errors
		mon2E = nM[nM['run']==run][mES2]
				
		# Failure to load rates (should be covered but just in case)
		if (len(mon1) == 0 or len(mon2) == 0):
			if len(mon1) == 0: 
				mR1.append(0.0)
			else:
				mR1.append(float(mon1))
			if len(mon2) == 0:
				mR2.append(0.0)
			else:
				mR2.append(float(mon2))
		else:
			mR1.append(float(mon1))
			mR2.append(float(mon2))	
		# Same thing for errors
		if (len(mon1E) == 0 or len(mon2E) == 0):
			if len(mon1E) == 0:
				mR1E.append(0.0)
			else:
				mR1E.append(float(mon1E))
			if len(mon2E) == 0:
				mR2E.append(0.0)
			else:
				mR2E.append(float(mon2E))
		else:
			mR1E.append(float(mon1E))
			mR2E.append(float(mon2E))
		mR.append(float(mon1)/float(mon2))
	# Now we go and check the rolling average of mR, with "error checking"
	if mR2[0] > 0:
		prev_ratio = (mR1[0]) / (mR2[0]) # for j=1
	elif mR2[-1] > 0:
		prev_ratio = mR1[-1] / mR2[-1]
	else:
		prev_ratio = 1.0
	prev_ratio = mR[0]
	ratioList = [] # For plotting
	asymList1 = [] # Asymmetries (positive and negative spectral corrections)
	asymList2 = []
	print(str(len(mR1))+str(len(mR2)))
	for i, r in enumerate(rL): # Loop through, calculating ratios/asymmetries by runs
		if i == len(mR1) - 1: # Last run safety check
			tmp = 1.0
			ratioList.append(tmp)
			pos_asym = 1.0
			break

		#pos_asym  = []
		pos_ratio = []
		for j in range(1,w): # Calculate the window across a range
			if j + i >= len(mR1): # Catch end (positive)
				continue
			#if mR2[i+j] != 0: # Avoid divide by zero errors
			if mR1[i+j] > mR2[i+j] and mR2[i+j] > 0:
				pos_ratio.append(mR1[i+j]/mR2[i+j]) 
			elif mR1[i+j] < mR2[i+j] and mR1[i+j] > 0:
				pos_ratio.append(mR2[i+j]/mR1[i+j]) 
			else:
				pos_ratio.append(1.0)
			#pos_asym.append((mR1[i+j] - mR1[i]))# / (mR2[i+j] - mR2[i]))
		if len(pos_ratio) > 0: # Now find positive value and errors
			#pos_val = np.mean(pos_ratio)
			pos_val = max(pos_ratio)
			pos_std = np.std(pos_ratio)
		else: # We've checked to make sure there's actually data
			if mR2[i] != 0.0: # Default to instantaneous value
				pos_val = mR1[i]/mR2[i]
			else:
				pos_val = 0.0
			pos_std = 0.0
			#asym1   = 0.0
		#pos_val = mR1[i+1]/mR2[i+1]
		# Here checking "Asymmetry", which is an instantaneous spectral correction
		if mR2[i+1] - mR2[i] != 0.0: # Not rolling here, check for div. by zero
			pos_asym = (mR1[i+1] - mR1[i])/ (mR2[i+1] - mR2[i]) 
		else: # Assume spectral correction goes to zero if the 2nd monitor doesn't change
			pos_asym = 0.0
					
		#neg_asym  = []
		neg_ratio = [] # Do the same thing, but on the negative end
		for j in range(1,w):
			if i - j < 0: # Catch end (negative)
				continue
			if mR1[i-j] > mR2[i-j] and mR2[i-j] > 0:
				neg_ratio.append(mR1[i-j]/mR2[i-j]) 
			elif mR1[i-j] < mR2[i-j] and mR1[i-j] > 0:
				neg_ratio.append(mR2[i-j]/mR1[i-j]) 
			else:
				neg_ratio.append(1.0)
			#if mR2[i-j] != 0.0: # Avoid divide by zero errors
			#	neg_ratio.append(mR1[i-j]/mR2[i-j])
		if len(neg_ratio) > 0: # Now find negative values and errors
			#neg_val = np.mean(neg_ratio)
			neg_val = max(neg_ratio)
			neg_std = np.std(neg_ratio)
		else: # We've checked to make sure there's actually data
			if mR2[i] != 0.0: # Default to instantaneous value
				neg_val = mR1[i]/mR2[i]
			else:
				neg_val = 0.0
			neg_std = 0.0
			
		# Now again check asymmetry (skipping if first run and thus can't go negative)
		if i > 0: 
			if mR2[i] - mR2[i] != 0.0: # Not rolling here, check for div. by zero
				neg_asym = (mR1[i] - mR1[i-1])/ (mR2[i] - mR2[i-1])

			else:
				neg_asym = 0.0
		else:
			neg_asym = 1.0
		
		try:
			if mR1[i] > 0:
				asym1 = (mR1[i+1]-mR1[i])/mR1[i] - (mR1[i]-mR1[i-1])/mR1[i]
			else:
				asym1 = 0.0
			if mR2[i] > 0:
				asym2 = (mR2[i+1]-mR2[i])/mR2[i] - (mR2[i]-mR2[i-1])/mR2[i]
			else:
				asym2 = 0.0
			#asym2 = (mR1[i]/mR2[i]) - (mR1[i-1] / mR2[i-1])
		except IndexError:
			asym1 = 0.0
			asym2 = 0.0
		
		if mR1[i] > 0:
			asymList1.append((mR1E[i]/mR1[i]) / (np.sqrt(mR1[i])/mR1[i]))
		else: 
			asymList1.append(0.0)
		if mR2[i] > 0:
			asymList2.append((mR2E[i]/mR2[i]) / (np.sqrt(mR2[i])/mR2[i]))
		else:
			asymList2.append(0.0)
		#asymList1.append(asym1)
		#asymList2.append(asym2)
		#asymList1.append((neg_asym + pos_asym)/2)
		#asymList1.append(mR1E[i])
		#asymList2.append((neg_asym - pos_asym)/2)
		#asymList2.append(mR2E[i])
		
		if len(pos_ratio) + len(neg_ratio) > 0: # Find theh average ratio
			tmp_ratio = pos_val + neg_val / (len(pos_ratio) + len(neg_ratio))
		else:
			tmp_ratio = pos_val+neg_val/2
			
		#tmp_ratio = (mR1[i+1] - mR1[i]) / (mR2[i+1] - mR2[i]) # for i
		#tmp = (tmp_ratio / prev_ratio)
		if neg_val > 0:
			tmp = pos_val / neg_val
		else:
			tmp = 1.0
		#tmp_ratio = mR[i+1]
		
		#if pctBreak <= tmp <= 1/pctBreak:
		#if (((pos_val - pos_std) > (neg_val + neg_std)) or 
		#	((pos_val + pos_std) < (neg_val - neg_std))):
				#if r not in rB:
				#	rB.append(r)
		ratioList.append(tmp_ratio / prev_ratio)
		#ratioList.append(tmp)
		#if r not in rB and not (pctBreak <= tmp <= 1/pctBreak):
		#if r not in rB and not (-pctBreak <= tmp <= pctBreak):
		#	rB.append(r)
		prev_ratio = tmp_ratio
	
	asymList1.append(1.0)
	asymList2.append(1.0)
	print(str(np.mean(asymList1)), str(np.std(asymList1)))
	print(str(np.mean(asymList2)), str(np.std(asymList2)))
	#print np.mean(ratioList), np.std(ratioList)
	mean_ratio = np.mean(ratioList)
	err_ratio  = np.std(ratioList)
	print (mean_ratio)
	if (mean_ratio - 3*err_ratio) <= 0.0: # Error catching
		mean_ratio = 1.0
		err_ratio  = 0.25 # Arbitrary!
	for i, r in enumerate(ratioList): # Now we scan and see if we're outside 3-sigma
		#if not (mean_ratio - 3*err_ratio <= r <= 1.0/(mean_ratio - 3*err_ratio))
		#if abs(asymList1[i]) > np.mean(asymList1) + 4*np.std(asymList1) and abs(asymList2[i]) > np.mean(asymList2) + 4*np.std(asymList2) :
		#	continue
		
		#if not (mean_ratio - 3*err_ratio <= r <= 1.0/(mean_ratio - 3*err_ratio)):
		if not (pctBreak <= r <= 1.0/pctBreak):# + (1.0 - pctBreak)):
			rB.append(rL[i])
			# dRatio = []
			# for j in range(-w,w-1):
				# if i + j + 1 >= len(ratioList):
					# break
				# dRatio.append(abs(ratioList[i+j+1] - ratioList[i+j]))
			# try:
				# if ((w == np.argmin(dRatio)) or
					# (dRatio[w+1] > 2*err_ratio and dRatio[w] > 2*err_ratio and dRatio[w-1] > 2*err_ratio)):
				# #if ((ratioList[i-1] < r and ratioList[i+1] > r)
				# #	or (ratioList[i-1] > r and ratioList[i+1] < r)):
					# rB.append(rL[i+1])
			# except IndexError:
				# print(" Reached the end of the runBreaks index!")
	
	# Hardcoded in forced changes:
	#rB = load_hardcoded_break(rB, rL, 4391) # Something is broken here
	#rB = load_hardcoded_break(rB, rL, 14008) # This octet just seems broken...
	#rB = load_hardcoded_break(rB, rL, 14018) # Big jump
	rB = load_hardcoded_break(rB, rL, 11715)
	rB = load_hardcoded_break(rB, rL, 13209)
	# Need to have a final runBreak in here as well.
	
	rB.append(max(rL))	
	#nPlt = plot_run_breaks(rL,mR1, mR2,ratioList,rB, nDet1,nDet2,mean_ratio-3*err_ratio,nPlt,asymList1,asymList2)
	#plt.show()
	return rB, nPlt	
		
		# # 3 areas: min edge, max edge and middle
		# if i >= w and i + w < len(rL): # Middle
			# avg1 = np.mean(mR1[i-w:i+w])
			# std1 = np.std(mR1[i-w:i+w])
			# avg2 = np.mean(mR2[i-w:i+w])
			# std2 = np.std(mR2[i-w:i+w])
		# elif i < w: # Lower
			# avg1 = np.mean(mR1[0:i+w])
			# std1 = np.std(mR1[0:i+w])
			# avg2 = np.mean(mR2[0:i+w])
			# std2 = np.std(mR2[0:i+w])
		# else: # Upper
			# avg1 = np.mean(mR1[i-w:])
			# std1 = np.std(mR2[i-w:])
			# avg2 = np.mean(mR2[i-w:])
			# std2 = np.std(mR2[i-w:])
			
		# try:
			# #print mR1[i],mR2[i], mR1[i]/mR2[i]
			# tmp_ratio = (mR1[i]+mR2[i])
			# tmp = tmp_ratio / prev_ratio# (avg1/avg2)
		# except ZeroDivisionError:
			# tmp = np.inf
			
		# ratioList.append(tmp)
		# if r not in rB and not (pctBreak <= tmp <= 1/pctBreak):
			# avgR = []
			# print r, tmp
			# for j in range(1,w): # Move ahead to see if we can ignore "bad runs" 
				# if i + j == len(rL): # Error checking
					# break
				# skip_ratio = mR1[i+j] + mR2[i+j] # Check the next handful
				# skip = skip_ratio / prev_ratio
				# print skip
				# avgR.append(skip)
			# print np.mean(avgR), np.std(avgR) / w
			# if ((pctBreak >= tmp + np.std(avgR)/w) or
				# (1/pctBreak <= tmp - np.std(avgR)/w)):
				# rB.append(r)
			
			#rB.append(r)
		#prev_ratio = tmp_ratio
	
	
	
	
			
			
	# ratioList = [1]
	# #rB2 = []
	# # Find our runBreaks based on this
	# for i, r in enumerate(mR):
		# if i > 0:
			# ratioList.append(mR[i]/mR[i-1])		
			# #if abs(mR[i] - mR[i-1]) > 5.0:
			# if (rL[i] - rL[i-1]) > 50 and rL[i] not in rB:# and rL[i] != 10864: # Run 10864 is glitchy because it doesn't have normalization
				# rB.append(rL[i])
			# elif (rL[i] > 8600): # Roundhouse
				# if not (0.85 <= (mR[i]/mR[i-1]) <= 1.15):
					# rB.append(rL[i])	
				# elif ((rL[i] > 13306) and (rL[i-1] < 13306)):# or ((rL[i] > 13209) and (rL[i-1] < 13209)) or ((rL[i] >= 14000) and (rL[i-1] < 14000)):# or ((rL[i] >= 14014) and (rL[i-1] < 14014)):
					# rB.append(rL[i])
			# else:
				# if not (0.75 <= (mR[i]/mR[i-1]) <=  1.25):
					# rB.append(rL[i])
					
					
	# plt.figure(1)
	# plt.plot(rL,mR,'b.')
	# plt.xlabel("Run Number")
	# plt.ylabel("Ratio of (detector %d) / (detector %d)" % (nDet1, nDet2))
	
			#if runList[i] == 4392 and runList[i] not in runBreaks:
			#	runBreaks.append(runList[i])
	#else:
	#	rB.append(13306) # Approximate position of dagger change.
		#rB.append(13209) # Approximate position of dagger change.

	# for xl in rB:
		# plt.axvline(x=xl)
		
	
	# plt.figure(2)
	# plt.plot(rL,ratioList,'r.')
	# plt.xlabel("Run Number")
	# plt.ylabel("Ratio of sequential runs")
	# plt.hlines(0.85,min(rL),max(rL))
	# plt.hlines(1.15,min(rL),max(rL))
	# for xl in rB:
		# plt.axvline(x=xl)
		
	# plt.show()
	
	
	
	#return rB

