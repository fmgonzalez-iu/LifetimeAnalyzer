#!/usr/local/bin/python3
import sys
import pdb
import csv
from math import *
import numpy as np
import emcee 					#MCMC module
import matplotlib.pyplot as plt

from statsmodels.stats.weightstats import DescrStatsW
from scipy import stats, special
from scipy.odr import *
from scipy.optimize import curve_fit, nnls, leastsq, lsq_linear
from scipy.special import factorial
from datetime import datetime
from multiprocessing import Pool  #for parallelizing

from PythonAnalyzer.writeOut import write_summed_counts
from PythonAnalyzer.classes import *
from PythonAnalyzer.functions import *
from PythonAnalyzer.backgrounds import *
#from numpy.random import normal   #just a normal thing to do in any code

#-----------------------------------------------------------------------
# Global normalization fit for the arbitrary runs I want to use
def global_normalization_fit(rList,holdT,det,mon, runBreaks = []):
	
	input_data = [] # Load the data we want to resample
	for i,x in enumerate(mon): # Needs to be this format for lnL fcn
		input_data.append([rList[i],float(holdT[i]),float(det[i].val),x[0].val,x[0].err,x[1].val,x[1].err])
	
	ini_alpha = float(det[0].val / mon[0][0].val) # Initial guess for alpha is just y / m1 for the first run
	
	# now on to emcee. ndim is number of model parameters + the value of lnf
	ndim, nwalkers = 3, 12 # nwalkers is the number of tests to run simultaneously

	#generate ensemble sampler, providing likelihood function, additional arguments, and any other options
	pool = Pool(1) # For multithreading on a single computer, set Pool(num_cores) 
	sampler = emcee.EnsembleSampler(nwalkers,ndim,lnL,args=(input_data,),pool=pool)
	
	# generate initial points in parameter space for the walkers.
	# I start the walkers as spread normally about my best guess
	walker_init =	[[	np.random.normal(880.0,10.0),     # tau
						np.random.normal(ini_alpha,1.0),  # N
						np.random.normal(0.0,1e-5)        # beta
					] for j in range(nwalkers) ]

	
	then = datetime.now()
	print (str(then))

	#lnFit = []
	#outData = []
	#for x in input_data:
	#	outData.append(float(x[2]))
	#	lnFit.append([float(x[1]),float(x[3]),float(x[5])])

	#pFitE, pVarE = curve_fit(ln_multid,lnFit,  outData, p0=(880.0,ini_alpha,0.0))
	#print pFitE[0], np.sqrt(np.diag(pVarE))[0]
	
	sampler.run_mcmc(walker_init,20000) # off we go! (20k steps)

	print (datetime.now()-then).total_seconds()
	pool.close()

	# print the autocorr time (an estimate of the uncorrelated number of samples)
	# this gives a measure of how well we have converged
	# -- if the autocorr times are too long compared
	# to the total run time, we might have to run longer
	try:
		print (sampler.get_autocorr_time() )
	except Exception as e:
		print (e)

	# flatten out all the walkers into one list of walker points for each parameter
	# discarding the first 500 for "burn-in" and keeping only every so often,
	# since points too nearby are correlated
	samples = sampler.get_chain(discard=500,thin=100,flat=True)
	# here "blob" is any additional return argument from my lnL function,
	# which in my case is just a chi2 goodness of fit
	blobs   = sampler.get_blobs(discard=500,thin=100,flat=True)
		
	np.savetxt("test_out.csv",np.column_stack((samples,blobs))) # output here
	#return pFitE, pVarE
	return np.column_stack((samples, blobs))

def analyze_multivar_norm(runList,cts,nMon,nMad,runBreaks,nDet1 = 3, nDet2 = 5):

	# Counting vectors for our normalization
	normedRun= []
	holdT    = []
	unldCts  = []
	monCts   = []
	monCtsGeo= []
	
	# Load Norm Monitor name
	monStr1  = ('mon'+str(nDet1))
	monEStr1 = ('mon'+str(nDet1)+'E')
	monStr2  = ('mon'+str(nDet2))
	monEStr2 = ('mon'+str(nDet2)+'E')

	for run in runList:
		
		# Load the raw data from each run -- the file is separated by dip already.
		ctsRaw  = cts[cts['run']==run]
		nMonRaw = nMon[nMon['run']==run]
		nMadRaw = nMad[nMad['run']==run]
		
		# For each run define nDips and holdT
		tOut = (nMonRaw['ts'] - nMonRaw['td']) - 50.0 # include cleaning time
				
		# Normalize background. Separated into multiple PMTs
		if useBkgCorr:
			[normBKG1, normBKG2, normBKG, bkgTime] = extract_background(nMonRaw)
		else:
			[normBKG1, normBKG2, normBKG, bkgTime] = [0.0, 0.0, 0.0, 0.0]
		bkgRaw = [normBKG1, normBKG2, normBKG, bkgTime]
			
		[ctsSum, dtSum, bkgSum] = sum_all_dips(ctsRaw, bkgRaw,norm2All)
	
		if useDTCorr == False:
			dtSum = measurement(0.0, 0.0)
		
		# Write out normalization monitors	
		normedRun.append(run)
		holdT.append(tOut)
		if expoNorm:
			monCts.append([measurement(nMonRaw[monStr1][0],nMonRaw[monEStr1][0]),measurement(nMonRaw[monStr2][0],nMonRaw[monEStr2][0])])
		if geomNorm:
			monCtsGeo.append([measurement(nMadRaw[nMadRaw['det']==nDet1]['mon'][0],nMadRaw[nMadRaw['det']==nDet1]['monE'][0]),measurement(nMadRaw[nMadRaw['det']==nDet2]['mon'][0],nMadRaw[nMadRaw['det']==nDet2]['monE'][0])])
		unldCts.append((ctsSum + dtSum - bkgSum))
	data_out = write_summed_counts(normedRun,holdT,unldCts,monCts)
	
	if expoNorm:
		likelihood_out = global_normalization_fit(normedRun,holdT,unldCts,monCts)
	if geomNorm:
		likelihood_out = global_normalization_fit(normedRun,holdT,unldCts,monCtsGeo)

	nPlt = plot_global_normalization(data_out, likelihood_out)
	return data_out, likelihood_out

def analyze_multivar(runList,cts,nMon,nMad,runBreaks = [],nDet1 = 4, nDet2 = 8,dips = range(0,3),pmt1=True,pmt2=True): # Main function here
	expoNorm = True
	geomNorm = False

	if (runBreaks == []):
		runBreaks = [runList[0],max(runList)]
	
	normedRun, timeL, ctsL, monL, dC,dtCts,bSubVec,_N = extract_values(runList, cts, nMon, nDet1, nDet2, 0, pmt1, pmt2,dips,runBreaks)
	
	t_scan = np.linspace(880,900,21)
	chi2Out = []
	for t in t_scan:
		nCoeff,coErr, chi2 = normalization_multiple(normedRun,timeL,monL,ctsL,runBreaks,t, 9999)
		print(t,chi2)
		chi2Out.append(chi2)
	#print(chi2)
	plt.figure(t_scan,chi2)
	nFac1, nFac2,normRuns,nCorr = normalize_counts_by_nvec(runList, cts, nMon, nDet1, nDet2, normedRun,nCoeff,coErr,runBreaks)
	
	runNum, holdVec, rawCts, rawMon, dC,dtCts,bSubVec,_N = extract_values(normRuns, cts, nMon, nDet1, nDet2, 0, pmt1, pmt2,dips,runBreaks)
	
	nCtsVec = []
	normVec = []
	for i, c in enumerate(rawCts):
		nCtsVec.append(c / (nCorr[i]))
		normVec.append(nCorr[i])
	meanArr = []
	
	nPhiVec = []
	pct2 = []
	pct3 = []
	sig2noi = []
	#write_summed_counts(runNum,holdVec,rawCts,rawMon)
	print("Managed to successfully normalize "+str(len(runNum))+" runs! This is a ratio of: "+str(float(len(runNum))/float(len(runList))))
	print(" ")
	if len(runNum) < 8:
		sys.exit("Error! Unable to normalize an entire octet's worth of runs! Exiting...")
	# lol i'm just returning a bunch of stuff
	return runNum,holdVec,nCtsVec,meanArr,rawCts,bSubVec,normVec,nFac1,nFac2,nPhiVec,pct2,pct3,dC
	
def analyze_multivar_norm_condensed(runListI,cts,nMon,nMad,runBreaks = [],nDet1 = 4, nDet2 = 8, dips = range(0,3)): # Main function here
	# Multivar norm will calculate the expected yield for each set of runs,
	# then will 

	expoNorm = True
	geomNorm = False

	if (runBreaks == []):
		runBreaks = [runList[0],max(runList)]
	# Figure out the holding times needed to normalize to: (don't bother saving vals) (Filters bad timings)
	runList, holdVec, _N, _N, _N, _N, _N, _N = extract_values(runListI, cts, nMon, nDet1, nDet2, 0, True, True, dips)
	timesInRun = []
	for t in holdVec:
		if round(float(t)) not in timesInRun: # I don't remember if this is a measurement?
			timesInRun.append(round(float(t)))
	
	timesInRun.sort() # Normalized runs should be in order
	# Initialize output vectors
	normRunsU = []
	nFac1U = []
	nFac2U = []
	nCorrU = []
	nTimeU  = []	
	for t in timesInRun: # Now normalize all the runs at a given time:
		if t > 1551: # Hard coding in skip for super-long runs
			continue 
		if not t % 5 == 0: # Hard coded assumption every real time we do is divisible by 5
			continue
		normedRunT, timeT, ctsT, monT, monET,bSubVecT,_N, _N = extract_values(runList, cts, nMon, nDet1, nDet2, t, True, True,dips)	
		nCoeff,coErr = normalization_params(normedRunT,monT,ctsT,runBreaks,99999)
		nFac1T, nFac2T,tmpRuns,nCorrT = normalize_counts_by_nvec(runList, cts, nMon, nDet1, nDet2, normedRunT,nCoeff,coErr,runBreaks)
		
		# Extend normalization runs
		normRunsU.extend(tmpRuns)
		nFac1U.extend(nFac1T)
		nFac2U.extend(nFac2T)
		nCorrU.extend(nCorrT)
		nTimeU.extend(t*np.ones(len(tmpRuns)))
		
	#print nFac1U
	#print nFac2U
	#print nCorrU
	# Sort runs and re-extract
	sortIndex = np.argsort(normRunsU)
	normRuns = np.empty(len(normRunsU))
	nFac1 = np.empty(len(nFac1U))
	nFac2 = np.empty(len(nFac2U))
	nCorr = np.empty(len(nCorrU))
	nTime = np.empty(len(nTimeU))
	print(len(sortIndex))
	for j,i in enumerate(sortIndex):
		normRuns[j] = normRunsU[i]
		nFac1[j] = nFac1U[i]
		nFac2[j] = nFac2U[i]
		nCorr[j] = nCorrU[i]
		nTime[j] = nTimeU[i]
		#print nCorrU[i].val
	
	# Extract from our parsed list
	runNum, holdVec, rawCts, rawMon, monEL,bSubVec,_N,_N  = extract_values(normRuns, cts, nMon, nDet1, nDet2, 0, True, True,dips)
	
	nCtsVec = []
	normVec = []
	for i, c in enumerate(rawCts):
		nCtsVec.append(c / (nCorr[i]))
		normVec.append(nCorr[i])
	meanArr = []
	
	nPhiVec = []
	pct2 = []
	pct3 = []
	sig2noi = []
	print("Managed to successfully normalize "+str(len(runNum))+" runs! This is a ratio of: "+str(float(len(runNum))/float(len(runList))))
	print(" ")
	if len(runNum) < 8:
		sys.exit("Error! Unable to normalize an entire octet's worth of runs! Exiting...")
	# lol i'm just returning a bunch of stuff
	return runNum,holdVec,nCtsVec,meanArr,rawCts,bSubVec,normVec,nFac1,nFac2,nPhiVec,pct2,pct3,nTime

def normalization_multiple(rList = [], hold = [], mon = [], det = [], breaks = [],t=880, wt = 3, vb = True):
	#-----------------------------------------------------------------------
	# Functional form of normalizing, given 2 weighted monitor
	#-----------------------------------------------------------------------
	# Input parameters are vectors (except "window" w):
	# rList is the list of normalization runs (for troubleshooting)
	# mon is a 2xN list of weighted monitor signals -- can be "measurement" or double
	# det is a list of the counts in our given detector -- can be "measurement" or double
	# breaks is our runbreaks list  
	#
	# np.linalg.lstsq(a,b) solves the equation a x = b by computing a vector x
	# that minimizes the equation ||b-ax||^2.
	#
	# Returns solution[0], resiudual[1], rank[2], and min value of a[3]
	#
	#-----------------------------------------------------------------------
	
	
	# Initialize counters and outputs
	nC   = [] # Coeffecients (output)
	err  = [] # Residuals (also output) [element 2 is the covariance]
	chi2 = 0
	
	bC = 0 # Break Counter
	m = 0 # Window counter (low)
	n = 0 # Window counter (high)
	#c2Out = open("BadRunsChi2_2.txt", "w")
	
	# Error checking----------------------------------------------------
	# Check that we have the right number of inputs
	if len(rList) != len(det) or len(rList) != len(mon):
		print("Error: Mismatched normalization size! Returning...")
		print ("   Loading these normalizations:",len(rList),len(det),len(mon))
		return nC, err, chi2
	if len(breaks) < 2:
		print("Error: Number of runBreaks must be 2 or more! (At least first and last)! Returning...")
		return nC, err, chi2
	
	# Double check runBreak boundaries
	if rList[0] < breaks[0] and vb:
		print("Warning: First element of runList is less than the first runBreak!")
		print("       I'll try to run this but something might be un-normalized!")
		breaks.append(rList[0])
		breaks.sort()
	if max(rList) > max(breaks) and vb:
		print("Warning: Last element of runList is more than the last runBreak!")
		print("        I'll try to run this but something might be un-normalized!")
		breaks.append(max(rList))
			
	if type(det[0]) == type(mon[0][0]):
		if isinstance(det[0],measurement): # Combined for calculating errors via black box or explicitly
			meas = True
		else:
			meas = False
	else:
		print("Error! Monitor and Detector counts not same datatype!")
		return nC, err, chi2
	#-------------------------------------------------------------------
		
	if meas: # Previous value of alpha (counting) and beta (spectral correction term)
		prevAlpha = det[0]/mon[0][0]
		prevBeta = measurement(0.0,0.0) # Set initial spectral correction at zero
	else:
		prevAlpha = float(det[0]/mon[0][0])
		prevBeta = 0.0 
		
	skip = 0
	# Loop through the normalization runlist
	for i, r in enumerate(rList):
		
		det_sca = det[i]*np.exp((hold[i]-20.)/t) # Here we're scaling the detector counts
		#print(det_sca,det[i],hold[i])		
		# First, figure out which runBreaks space we're in and adjust bC.
		if 0 <= bC < len(breaks)-1: 
			bmi = breaks[bC]
			bma = breaks[bC + 1]
			
			# Check if our present run is greater than the next runBreak
			if (r >= bma):
				# Start a loop to increase bC
				while (r >= bma):
					# Make sure we don't increment too far!
					if bC == len(breaks) - 2:
						if vb:
							print("This is the maximum runBreak, proceeding!")
						bmi = breaks[bC]
						bma = breaks[bC + 1]
						break
					bC += 1
					bmi = breaks[bC]
					bma = breaks[bC + 1]
			if not (bmi <= r < bma):
				if r == bma and bC == len(breaks) - 2: # Last possible break
					alpha = (det_sca - prevBeta*mon[i][1]) / mon[i][0]
					if meas:
						nC.append([alpha.val,prevBeta.val])
						err.append([alpha.err,prevBeta.err,0.0])
					else:
						nC.append([float(alpha),float(prevBeta)])
						if mon[i][0]*det_sca > 0: # Error calculation -- make sure it's real
							alpha_err = float(np.sqrt(mon[i][0]+det_sca)/np.sqrt(mon[i][0]*det_sca))
						else:
							alpha_err = np.inf
						if mon[i][0]*mon[i][1] > 0: # Require counts in both monitors
							beta_err = float(np.sqrt(mon[i][0]+mon[i][1])/np.sqrt(mon[i][0]*mon[i][1]))
						else:
							beta_err = np.inf
						err.append([alpha_err, beta_err,0.0])
									
					bC -=1
					prevAlpha = alpha
					continue					
				else:
					if vb:
						print("Unable to normalize run", r, "with rolling normalization!")
						print(bmi,bma)
					alpha = (det_sca - prevBeta*mon[i][1]) / mon[i][0]
										
					if meas:
						nC.append([alpha.val,prevBeta.val])
						err.append([alpha.err,prevBeta.err,0.0])
					else:
						nC.append([float(alpha),float(prevBeta)])
						if mon[i][0]*det_sca > 0: # Error calculation -- make sure it's real
							alpha_err = float(np.sqrt(mon[i][0]+det_sca)/np.sqrt(mon[i][0]*det_sca))
						else:
							alpha_err = np.inf
						if mon[i][0]*mon[i][1] > 0:
							beta_err = float(np.sqrt(mon[i][0]+mon[i][1])/np.sqrt(mon[i][0]*mon[i][1]))
						else:
							beta_err = np.inf
						err.append([alpha_err, beta_err,0.0])
					prevAlpha = alpha
					continue
					
		# Make sure max run [i+wt] is callable i.e. [i+len(rList) - (i+1)] = [len(rList)-1]
		if i + wt < len(rList): 
			n = wt
		else:
			n = len(rList) - (1 + i)
		# Make sure min run [i-wt] is callable, i.e. [i-i] = [0]
		if i >= wt: 
			m = wt
		else:
			m = i
			
		# Get the upper bound
		if rList[i+n] >= bma:
			while (rList[i+n] >= bma):
				n-=1
			if n < 0:
				if vb:
					print("Somehow have a negative window on run "+str(r))
					print(i,n,bmi,bma)
				while breaks[bC+1] <= rList[i+1]: 
					bC += 1
				rList.pop(i)
				mon.pop(i)
				det.pop(i)
		# Get the lower bound
		if rList[i-m] < bmi:
			while (rList[i-m] < bmi):
				m-=1
			if m < 0:
				if vb:
					print("Somehow have a negative minimum on run "+str(r))
					print(i,n,bmi,bma)
				rList.pop(i)
				mon.pop(i)
				det.pop(i)
		
		# case of just one normalization run in bounds
		if m == 0 and n == 0:
		
			alpha = (det_sca - mon[i][1]*prevBeta) / mon[i][0]
			if meas:
				nC.append([alpha.val,prevBeta.val])
				err.append([alpha.err,prevBeta.err,0.0])
			else:
				nC.append([float(alpha),float(prevBeta)])
				
				if mon[i][0]*det_sca > 0: # Error calculation -- make sure it's real
					alpha_err = float(np.sqrt(mon[i][0]+det_sca)/np.sqrt(mon[i][0]*det_sca))
				else:
					alpha_err = np.inf
				if mon[i][0]*mon[i][1] > 0:
					beta_err = float(np.sqrt(mon[i][0]+mon[i][1])/np.sqrt(mon[i][0]*mon[i][1]))
				else:
					beta_err = np.inf
				err.append([alpha_err, beta_err,0.0])
			prevAlpha = alpha		
		# shift window to account for edge effects
		elif m == 0 or n == 0:
			if m == 0:
				# If we can call n+w, do it. Otherwise just use max n
				if i + (n+wt) < len(rList): 
					n += wt
				else:
					n = len(rList) - (1 + i)
				if rList[i+n] >= bma:
					while (rList[i+n] >= bma):
						n-=1
				if n < 0:
					if vb:
						print("Somehow have a negative window on run "+str(r))
						print(i,n,bmi,bma)
					while breaks[bC+1] <= rList[i+1]: 
						bC += 1
					rList.pop(i)
					mon.pop(i)
					det.pop(i)
			elif n == 0:	
				# If we can call m+w, do it. Otherwise use max m
				if i >= (m+wt): 
					m += wt
				else:
					m = i
				if rList[i-m] < bmi:
					while (rList[i-m] < bmi):
						m-=1
				if m < 0:
					if vb:
						print("Somehow have a negative minimum on run "+str(r))
						print(i,n,bmi,bma)
					rList.pop(i)
					mon.pop(i)
					det.pop(i)

		# Now calc. rolling norm
		if m > 0 or n > 0:
			
			m1 = []
			m2 = []
			val = []
			verr= [] # This remains empty if no meas
			
			if m + n + 1 > 2: # Curve fit doesn't like sparse matrices
				for x in range(i-m,i+n+1):
					det_sca_tmp = det[x]*np.exp((hold[x]-20.)/t)
					#m1.append(float(mon[x][0]))
					#m2.append(float(mon[x][1]))
					if meas: # I guess I have to do a full divide-by-zero safety check here...
						if mon[x][0].val > 0: # Make sure we're weighting by real values
							rE1 = (mon[x][0].err / mon[x][0].val)**2
						else: # If there's no counts in the monitor, we can skip that monitor for weighting
							rE1 = 0.0
						if mon[x][1].val > 0:
							rE2 = (mon[x][1].err / mon[x][1].val)**2
						else: 
							rE2 = 0.0
						if rE1 + rE2 > 0: # Now we should have monitor counts and errors
							weight = 1.0/(rE1+rE2)
						else: # If not, set weighting at zero since we don't know what's happening.
							weight = 0.0 
					
					else:
						weight = 1.0
					if weight == 0: 
						print("Warning! Weighting is Set To Zero!")
					if weight > 0:
						weight = 1.0
						m1.append(float(mon[x][0])*weight)
						m2.append(float(mon[x][1]/mon[x][0])*weight)
						if meas:
							val.append(float(det_sca_tmp.val)*weight)
							#verr.append(float(det[x].err)*weight)
							verr.append(1.0)#float(det[x].err)*weight)
						else:
							val.append(float(det_sca_tmp))
					
				#-------------------------------------------------------
				if float(mon[i][0]) > 0.0: # Make sure we have a rational starting spot
					if meas:
						try:
							co, cov = curve_fit(spectral_norm_meas, (m1,m2), val,sigma=verr,absolute_sigma=False, p0=(float(det_sca/mon[i][0]),0.0),bounds=([0.0,-np.inf],[np.inf,np.inf]))
							#co, cov = curve_fit(spectral_norm_meas, (m1,m2), val, p0=(float(det[i]/mon[i][0]),0.0),bounds=([0.0,-np.inf],[np.inf,np.inf]))
						except ValueError:
							co = [float(det_sca/mon[i][0]),0.0]
							cov = [[np.inf,np.inf],[np.inf,np.inf]]
							#print (str(m1)+str(m2)+str(val)+str(verr))
						#co, cov = curve_fit(spectral_norm_meas, (m1,m2), val,sigma=verr,absolute_sigma=False, p0=(float(prevAlpha),float(prevBeta)),bounds=([0.0,-np.inf],[np.inf,np.inf]))
					else:
						co, cov = curve_fit(spectral_norm, (m1,m2), val, p0=(float(prevAlpha),float(prevBeta)),bounds=([0.0,-np.inf],[np.inf,np.inf]))
				elif float(mon[i][1]) > 0.0: # Try using the secondary detector
					if meas:
						co, cov = curve_fit(spectral_norm_meas, (m1,m2), val,sigma=verr,absolute_sigma=False, p0=(0.0,float((det_sca/mon[i][1]))),bounds=([0.0,-np.inf],[np.inf,np.inf]))
					else:
						co, cov = curve_fit(spectral_norm, (m1,m2), val, p0=(0.0,float(det_sca/mon[i][1])),bounds=([0.0,-np.inf],[np.inf,np.inf]))
				else: # Uh, just skip it and move on?
					print("Error: Can't normalize to an empty bin!")
					#c2Out.write("%d," % rList[i])
					#skip += 1
					break
				if not np.isfinite(np.diag(cov)[0]):
					print("Error! Covariance matrix is infinite for run", r,"!")
					#c2Out.write("%d," % rList[i])
					#skip += 1
					#print(m1,m2,val)
				
				#print("Factors:",co)
				#print("Covariance:",cov)
				nC.append(co)
				print(co, m2)
				# Chi squared correlated fit:
				#print(spectral_norm_meas([m1,m2],co[0],co[1]),np.array(val))
				#print val
				#
				unc = (float(co[0]*float(mon[i][0])+co[1]*float(mon[i][1]/mon[i][0]))
								+ (cov[0][0]*float(mon[i][0]*mon[i][0])
								+cov[1][1]*float(mon[i][1]*mon[i][1]/mon[i][0]/mon[i][0]) 
								+ 2*cov[0][1]*float(mon[i][1])))
				chisq =((co[0]*float(mon[i][0])+co[1]*float(mon[i][1]/mon[i][0]) - float(det_sca))**2/float(unc))						#/
							#co[0]*(mon[i][0]).err+co[1]*(mon[i][1]/mon[i][0]).err)
				#print(co[0]*float(mon[i][0])+co[1]*float(mon[i][1]/mon[i][0]),float(det_sca), chisq)
				if chisq > 50:
					#print(rList[i],chisq)
					#c2Out.write("%d," % rList[i])
					skip += 1
				else:
					chi2 += chisq
				chisq = sum(((spectral_norm_meas(np.array([m1,m2]),co[0],co[1]) -np.array(val))**2/np.array(val)))
				chisq /= m+n-1
				if meas:
					prevAlpha = measurement(co[0],np.sqrt(np.diag(cov)[0])/chisq)
					prevBeta = measurement(co[1],np.sqrt(np.diag(cov)[1])/chisq)
				else:
					prevAlpha = co[0]
					prevBeta = co[1]
				if abs(cov[1][0] - cov[0][1]) > 1e-8: # Arbitrarily choosing 10^-8 precision. Python seems to be only sometimes doing double bit precision (10^-16). 
					print("Warning, poorly defined covariance! "+str(cov[1][0] - cov[0][1]))
							
				if (np.diag(cov)[0] >= 0 and np.diag(cov)[1] >= 0) and not (m+n+1) == 0:
					#err.append(np.sqrt(np.diag(cov))/chisq)#(m+n+1.0))
					#err.append([np.sqrt(cov[0][0]), np.sqrt(cov[1][1]), cov[1][0])
					err.append([np.sqrt(cov[0][0]/chisq), np.sqrt(cov[1][1]/chisq), cov[1][0]/chisq])
				else:
					err.append([np.inf,np.inf,0.0])
			else: # Two equations, two unknowns
				print("Unconstrained curve_fit on run", r)
				prevAlpha  = ((det[i+n] - det[i-m] * (mon[i+n][1]/mon[i-m][1])) 
							/ (mon[i+n][0] - mon[i-m][0] * (mon[i+n][1]/mon[i-m][1])))
				prevBeta   = (det[i-m] - prevAlpha*mon[i-m][0])/mon[i-m][1]
				nC.append([prevAlpha.val, prevBeta.val])
				err.append([prevAlpha.err,prevBeta.err,0.0])
				chi2 += 0
	
	#for e in range(0,len(err)):
	#c2Out.close()	
	#err = err/chi2
	chi2 /= len(nC)-skip - 1
	
	#print(chi2)
	return nC, err, chi2
