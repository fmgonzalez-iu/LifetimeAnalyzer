#!/usr/local/bin/python3
#import os
#import sys
#import pdb
#import csv

from math import *
import numpy as np
import emcee

#from scipy import stats, special
from scipy.special import factorial, gammaln
from scipy.optimize import curve_fit
from schwimmbad import MPIPool
from datetime import datetime
from itertools import starmap
from multiprocessing import Pool
#from scipy.odr import *
#from scipy.optimize import curve_fit, nnls
#from datetime import datetime
#import matplotlib.pyplot as plt
from PythonAnalyzer.extract import *
from PythonAnalyzer.backgrounds import *
from PythonAnalyzer.classes import *
from PythonAnalyzer.dataIO import *

#-----------------------------------------------------------------------
# This analyzer code normalizes the monitor detector counts
# and uses a likelihood function to do it. It's been optimized to run on 
# BR3 with MPI.
#-----------------------------------------------------------------------

# Detectors for normalization
#global nDet1
#global nDet2

# Here are the functions to use for optimization
#-----------------------------------------------------------------------
def exp_cts(tau,alpha,beta,dt,m1,m2):
	# Expected yields in the dagger
	# Initial counts should be estimated through alpha*m1+beta*m2/m1, then through time
	
	return (alpha*m1+beta*(m1*m1/m2))*np.exp(-(dt-20.)/tau) # Setting 20s as scaled to	
	#return (alpha*m1+beta*(m2/m1))*np.exp(-(dt-20.)/tau) # Setting 20s as scaled to	

def eff_scale(sVar,cts,s):
	# Efficiency factor likelihood
	# cts should be scaled somehow since it's not a real Gaussian
	if cts > 0:
		sigma = s * np.sqrt(1 + s) / np.sqrt(cts) # Measurement Uncertainty
		return -(s-sVar)**2 / (2*sigma*sigma) # Assuming Gaussian
	else:
		return -np.inf
		
	

def bkg_cts(bkgVar,S_mu,b):
	# "Expected" background likelihood
	# This assumes that backgrounds can be modeled as a Poisson process (with a relevant scaling factor)
		
	return (b/S_mu)*np.log(bkgVar/S_mu)-(bkgVar/S_mu) - gammaln(b/S_mu+1)#((b/S_mu)*np.log(b/S_mu) - (b/S_mu)) # Poisson background (with stirling)

global gamma_List # Define optimized gamma
def poisson_opt_cts(l,us):
	# Poisson probability with a single gammaln(m+1) factor called
	
	like = us*(np.log(l)) - l - gamma_List[us]
	return like

def poisson_opt_bkg(l,m,s):
	# Poisson probablility BUT I've optimized out the gammaln(m+1) already 
	# This shold speed up running considerably.
	# Note that "m" here is b1, which is scaled. B_mu is l and must be scaled here.
	
	like = (m)*(np.log(l/s)) - (l/s) - gamma_List[0:len(m)]
	return like
	
def poisson_opt_one(l,m,s):
	# "Expected" background likelihood
	# This assumes that backgrounds can be modeled as a Poisson process (with a relevant scaling factor)
	# Optimized so that only one likelihood is calculated
	
	ms = int(m/s)		
	return (m/s)*np.log(l/s)-(l/s) - gamma_List[ms]

def unl_cts_floatB(U,B,Y,S):
	# Use this if we want to just float B
	US = int(U/S)
	return US*np.log(Y+B/S)-(Y+B/S) - gamma_List[US]
	
# Probably should code Gaussian/Other likelihood functions but w/e
def unl_cts_summed(U,B,Y,S):
	# Poisson, summing across all possible background counts
	# For unload counts I'm using Stirling's approx for poisson counts
	#lnL = 0.0 # Start second part at 0
	
	US = int(U/S) # Cast to int for speed
	b1 = np.linspace(0,US+1,US+1) # Find the possible backgrounds to sum over
	lnLArr = poisson_opt_cts(Y+b1,US) + poisson_opt_bkg(B,b1,S)
	lnL = np.sum(np.exp(lnLArr))
	
	if lnL > 0 and np.isfinite(lnL): # Make sure we didn't break things
		return np.log(lnL)
	else: # we are below rounding error
		if max(lnLArr) < 0:
			return max(lnLArr) # Assume everything is the same as the max. There should be an ln(len(lnLArr)) added in, but I'm trying to optimize this.
		else:
			return -np.inf
#-----------------------------------------------------------------------
# Optimized log likelihoods
#-----------------------------------------------------------------------
def ln_pdf_full(tau,alpha,beta,S_mu,B_mu,
				dt,u,b,s,m1,run,m2,BE):
	
	# This is the probability distribution
	Y_mu = exp_cts(tau,alpha,beta,dt,m1,m2) # Expected UCN in unload
	if Y_mu < 0: 
		return -np.inf # Force at least 1 neutron in trap (since a negative will break log)
	
	if S_mu == 1.: # S_mu should be forced 1 for coincidences
		unlVar = unl_cts_floatB(u,B_mu,Y_mu,S_mu) # Variation in unload counts
		sVar   = 0. # Don't bother with the variance in sVar
	else:
		unlVar = unl_cts_floatB(u,B_mu+BE,Y_mu,S_mu) # Variation in unload counts
		sVar   = eff_scale(S_mu,u-(b+BE), s) # Expected scaling factor -- how many "events" in a UCN
	
	bkgVar = poisson_opt_one(B_mu,b,S_mu) # Expected background counts, optimized
	#print(run,unlVar+bkgVar+sVar)	
	if np.isfinite(unlVar+bkgVar+sVar):
		return unlVar+bkgVar+sVar
	else:
		return -np.inf

global data_arr_global # For faster parallelization, need to define a global data_arr
def lnL_global_sing(param): 
	# Breaking our log-likelihood function into singles and coincidences
	# input the data and find the number of breaks:
	
	breaks = int((len(param) - 1) / 4) # 5 parameters in param: tau, N, beta,S,B_mu. Have 2*breaks n and betas
	tau = param[0] # lifetime, number of init. counts, and "temperature correction" factor beta
		
	# Physical requirement that counts are positive and lifetime is real, 
	# and also background/efficiency positive. 
	# Efficiency is 0.9 so that I can pre-calculate the factorials, and < 100 because I can put a limit there
	if (tau < 0.0  \
		or any([n <= 0.0 for n in param[1:breaks+1]]) \
		or any([(s <= 0.9 or s > 100.) for s in param[2*breaks+1:3*breaks+1]]) \
		or any([b <= 0.0 for b in param[3*breaks+1:4*breaks+1]])):  
		return -np.inf, np.inf
		
	# Sum the log pdf for each data point
	global data_arr_global
	L_ = sum(
		starmap(
			ln_pdf_full,
					 [(tau,n,b,sm,bm,*d[1:])
						for (n,b,sm,bm,data) in zip(param[1:(breaks+1)],param[(breaks+1):(2*breaks+1)],param[(2*breaks+1):(3*breaks+1)],param[(3*breaks+1):(4*breaks+1)],data_arr_global)
						for d in data
					 ]
				)
			)
	#print(L_)
	# return the likelihood to be maximized, and the chi2 for reference later
	if np.isfinite(L_): # Assuming our likelihood worked, return it
		return L_, -2.0*L_
	else: # If something went wrong, no probability
		return -np.inf,np.inf

def lnL_global_coinc(param): 
	# Breaking our log-likelihood function into singles and coincidences
	# input the data and find the number of breaks:
	#-------------------------------------------------------------------
	
	breaks = int((len(param) - 1) / 3) # 4 parameters in param: tau, N, beta,B_mu. Have 2*breaks n and betas
	tau = param[0] # lifetime, number of init. counts, and "temperature correction" factor beta
		
	# Physical requirement that counts are positive and lifetime is real, 
	# and also background/efficiency positive.
	if (tau < 0.0  \
		or any([n <= 0.0 for n in param[1:breaks+1]]) \
		or any([b <= 0.0 for b in param[2*breaks+1:3*breaks+1]])):  
		return -np.inf, np.inf
		
	# Sum the log pdf for each data point
	# Hardcoding S_mu = 1.
	global data_arr_global
	L_ = sum(
		starmap(
			ln_pdf_full,
					 [(tau,n,b,1.,bm,*d[1:])
						for (n,b,bm,data) in zip(param[1:(breaks+1)],param[(breaks+1):(2*breaks+1)],param[(2*breaks+1):(3*breaks+1)],data_arr_global)
						for d in data
					 ]
				)
			)
	#print(L_)		
	# return the likelihood to be maximized, and the chi2 for reference later
	if np.isfinite(L_): # Assuming our likelihood worked, return it
		return L_, -2.0*L_
	else: # If something went wrong, no probability
		return -np.inf,np.inf

def curve_fit_unloads(runNum,tStore,ctsSum,mSum1,mSum2,holdT = 20, runBreaks = []):
	# We're going to guess by using curve fit since it's 	
	
	parameters=np.ones((3,2))
	parList = []
	covList = []
	
	for i, r in enumerate(runBreaks):
		if i==len(runBreaks)-1:
			continue	
		if holdT > 0:
			condition=(tStore==holdT)*(r<=runNum)*(runNum<runBreaks[i+1])
		else:
			condition=(r<=runNum)*(runNum<runBreaks[i+1])
		testMon = np.transpose(np.array([mSum1[condition],mSum2[condition]]))
		if len(testMon) == 0: # check that we loaded counts
			runCond = (r<= runNum)*(runNum<runBreaks[i+1])
			if len(tStore[runCond]) > 0:
				newHold = np.min(tStore[runCond])
			else:
				print("runBreak has zero runs!",r,runBreaks[i+1])
				continue
			condition = (tStore==newHold)*runCond
			# Guessing an 880s fit time.
			testMon = np.transpose(np.array([mSum1[condition]*np.exp(20-newHold)/880.,\
											 mSum2[condition]*np.exp(20-newHold)/880.]))
			
			#parameters = np.array([0.,0.]) # Put in zeros if we didn't
			#pov_matrix = np.array([[1.,0.],[0.,1.]])
			#print(runNum,holdT)
			
			#parList.append(parameters)
			#covList.append(pov_matrix)
			#continue
		testCts = ctsSum[condition] # Prior to this, subtract bkg
		guess=[testCts[0]/testMon[0,0],0.]
		
		def linear_inv_2det(x,a,b): # Optimization function
			try:
				x1=np.array(x[:,0]) # Must convert to two numbers
				x2=np.array(x[:,1])
			except IndexError:
				x1=x[0]
				x2=x[1]
			except ValueError:
				print(x)
				sys.exit()
			return a*x1+b*x1*x1/x2 # Doing 2 detectors
		
		parameters,pov_matrix = curve_fit(linear_inv_2det,np.float64(testMon),np.float64(testCts),bounds=([0.,-np.inf],[np.inf,np.inf]))
	
		# Check finite-ness (This crashes somewhere and IDK why)
		
		parList.append(parameters)
		covList.append(pov_matrix)
	
	return parList, covList


#def optimized_likelihood_fit(runList,cts,nMon, dips=range(0,3),runBreaks=[]):
def optimized_likelihood_fit(runList,cts,nMon,bkgsH,bkgsT,cfg):# dips=range(0,3),runBreaks=[]):
		
	# Start here with extracting our values
	pltFig = 0
	#if len(fName) == 0: 
	#	fName = 'likelihood_out.txt'
	#if len(runBreaks) == 0:	#runBreaks=[4230,4672,5713,6753,6960,7326,9768,9960,10940,10988,11085,11217,12514,13209,14508] # Added 13307 	
	# 5453 should also be in here.
	runBreaks=[4230,4391,4415,4711,5475,5713,5955,6126,6429,6754,6930,7326,7490,\
				9767,9960,10936,10988,11085,11669,12516,13209,13307,14509]
	#runBreaks=[4223,4672,5713,5955,6125,6429,6753,6960,7326,7490,\
	#			9767,9960,10936,10988,11085,11669,12516,13209,14509]
		#6367,6390,13307,
	
	#extract_values
	runObj = extract_reduced_runs(runList, cts, nMon, bkgsH,bkgsT,cfg)
	
	# Allocate numpy arrays of "extracted values" to put in our array
	run_no  = np.zeros(len(runObj),dtype='i4') # Run Numbers
	t_store = np.zeros(len(runObj),dtype='i4') # Nominal Hold
	fore    = np.zeros(len(runObj),dtype='f8') # cts + dt
	back    = np.zeros(len(runObj),dtype='f8') # Time independent bkg
	bTDep   = np.zeros(len(runObj),dtype='f8') # Time dependent part of bkg
	mat     = np.zeros(len(runObj),dtype='f8') # Mean Arrival Time
	eff     = np.zeros(len(runObj),dtype='f8') # Efficiency parameters
	#monS    = np.zeros((len(runObj),2),dtype='f8') # Monitor Counts
	low_mon = np.zeros(len(runObj),dtype='f8') # Low Monitor Counts
	hi_mon  = np.zeros(len(runObj),dtype='f8') # Efficiency parameters
	
	# Now fill these arrays from the runsObj:
	# Many of these are type "measurement", so we need to convert to float
	for i, r in enumerate(runObj):
		run_no[i]  = r.run
		t_store[i] = int(round(r.hold))       # Hold, nearest int
		fore[i]    = (r.ctsSum).val
		if cfg.useDTCorr:
			fore[i] += r.dtSum.val
		back[i]    = (r.bkgSum - r.tCSum).val # Time Independent Component
		bTDep[i]   = (r.tCSum).val            # Time Dependent Component
		if cfg.useMeanArr:
			mat[i] = r.mat.val
		else:
			mat[i] = r.hold
		if r.sing:
			eff[i] = (r.eff[0]+r.eff[1]).val
		else:
			eff[i] =  1.
		low_mon[i] = r.mon[0].val
		hi_mon[i]  = r.mon[1].val
		#monS[i]    = [r.mon[0].val,r.mon[1].val]
	
	# Convert runBreaks to run_breaks based on run numbers
	run_breaks = []
	for i in range(0,len(runBreaks)-1):
		if runBreaks[i]<=run_no[0] < runBreaks[i+1]: 
			run_breaks.append(runBreaks[i]) # find first
			for j in range(i+1,len(runBreaks)):
				cond = (run_breaks[-1] <= run_no)*(run_no < runBreaks[j])
				if runBreaks[j] <= run_no[len(run_no)-1]:
					if len(run_no[cond]) > 0:
						run_breaks.append(runBreaks[j]) # append through the last break
				else:
					break
			break
	try: # And possibly add the last run.
		if run_breaks[-1] < np.max(run_no)+1: 
			run_breaks.append(np.max(run_no)+1)
	except IndexError: # If things are weird, just go with min/max runs
		run_breaks = [np.min(run_no),np.max(run_no)+1]

	raw_data = np.empty((len(run_no),9)) # 9 parameters 
	#print(np.size(raw_data))
	for r in range(0,len(run_no)):
		#print(run_no[r],t_store[r],U[r],low_mon[r],hi_mon[r])
		raw_data[r] = np.array([(run_no[r],
								#t_store[r],
								mat[r],
								fore[r], 
								back[r],
								eff[r], 
								low_mon[r],run_no[r], # For now assume no uncertainty from monitor counts.
								hi_mon[r],bTDep[r])]) # Sliding the background correction in here so I don't have to rework things
	#raw_data = genfromtxt(argv[1])
	#runlist = list(genfromtxt(argv[2])) # List of runs -- don't need 
	#runbreaks = list(genfromtxt(argv[3])) # List of data 
	
	input_list = [] #Separate the numpy data arrays into a list for each run break
	for rb1,rb2 in zip(run_breaks[0:-1],run_breaks[1:]): 
		#print(rb1,rb2)
		input_list.append(np.array([x for x in raw_data if rb1<=x[0]<rb2]))
		#input_list.append(array([x for x in raw_data if x[0] in runlist and rb1<=x[0]<rb2]))
			
	# Optimizing b1 here by setting it as a global
	b1min = 0 
	b1max = int(max(fore)/0.9) + 1 # Scaling to 0.9 efficiency
	b1 = np.linspace(b1min,b1max,int(max(fore)/0.9)+1) # Find the possible backgrounds to sum over
	
	global gamma_List # Pre-load the lookup table of factorials
	gamma_List = gammaln(b1+1)
	
	# Start values near the averages
	global data_arr_global # Load data_arr as a global before parallel processing
	data_arr_global = input_list
	
	N_guesses = []
	N_spreads = []
	A_guesses = []
	A_spreads = []
	params, covs = curve_fit_unloads(run_no,t_store,(fore-(back+bTDep))/eff,low_mon,hi_mon,20, run_breaks)
	for k in range(len(run_breaks) - 1):
		N_guesses.append(params[k][0])
		A_guesses.append(params[k][1])
		if (covs[k][0,0] > 0) and np.isfinite(covs[k][0,0]):
			N_spreads.append(np.sqrt(covs[k][0,0]))
		else:
			N_spreads.append(np.sqrt(np.abs(params[k][0])))
		if (covs[k][1,1] > 0) and np.isfinite(covs[k][1,1]):
			A_spreads.append(np.sqrt(covs[k][1,1]))
		else:
			A_spreads.append(np.sqrt(np.abs(params[k][1])))
	#print(N_guesses,A_guesses)
	#print(N_spreads,A_spreads)
	
	S_guesses = [np.average([x[4] for x in data]) for data in input_list]
	S_spreads = [np.std([x[4] for x in data])/np.sqrt(len(data))+0.001 for data in input_list]
	B_guesses = [np.average([x[3] for x in data]) for data in input_list]
		
	# Now on to emcee. ndim is number of model parameters + the value of lnf
	#print(S_spreads)
	# generate initial points in parameter space for the walkers.
	# I start the walkers as spread normally about my best guess
	# Don't have an initial walker for S in coincidence
	if anlz.sing:
		ndim, nwalkers = 1+4*(len(run_breaks)-1), 144
		walker_init = [
						[np.random.normal(880.0,10.0)] \
						+[abs(np.random.normal(N,N_spreads[i])) for i,N in enumerate(N_guesses)] \
						+[(np.random.normal(A,A_spreads[i])) for i,A in enumerate(A_guesses)] \
						+[abs(np.random.normal(Sm,S_spreads[i])) for i,Sm in enumerate(S_guesses)] \
						+[abs(np.random.normal(Bm,np.sqrt(Bm))) for Bm in B_guesses] \
						for j in range(nwalkers) \
					  ]
	else:
		ndim, nwalkers = 1+3*(len(run_breaks)-1), 128
		walker_init = [
						[np.random.normal(880.0,10.0)] \
						+[abs(np.random.normal(N,N_spreads[i])) for i,N in enumerate(N_guesses)] \
						+[(np.random.normal(A,A_spreads[i])) for i,A in enumerate(A_guesses)] \
						+[abs(np.random.normal(Bm,np.sqrt(Bm))) for Bm in B_guesses] \
						for j in range(nwalkers) \
					  ]
	return nwalkers,ndim,walker_init # OK now let's skip this for a bit 

if __name__ == "__main__":
	
	#-------------------------------------------------------------------
	# Change these if you want to do different datasets
	# Breaks I've used: 4674, 7326, 9600, 11713, 14508
	#global nDet1 
	#global nDet2
#	iniRun = 13309
#	finRun = 99999
	#-------------------------------------------------------------------
						
	#initialization_script(nList,bData,bNorm,bPlot,bWrite)
	fileList = [''] * 10 # If you need more scale this up
	fileListV = []
	for i, f in enumerate(sys.argv):
		fileList[i] = f
		if i ==0:
			continue
		fileListV.append(f)
			
	vb,loadS,anlz, outS = initialization_script()
	runListAll, ctsSing, ctsCoinc, nMon, _x, bkgsH,bkgsT = load_flex_lists(fileListV,loadS,vb)
	ctsTrunc = []
	if len(ctsSing) == 0:
		ctsTrunc = ctsCoinc
		anlz.sing = False
		if 'OUTPUT_PATH' in os.environ:
			savePath = os.environ['OUTPUT_PATH']
			saveName=(savePath+"/outputCoinc"+str(runListAll[0])+".txt")
		else:
			saveName = ("/N/slate/frangonz/outputCoinc"+str(runListAll[0])+".txt")
		#saveName = ("/N/u/frangonz/BigRed3/New_MCA_Analysis/outputs/outputCoinc"+str(iniRun)+".txt")
	if len(ctsCoinc) == 0:
		ctsTrunc = ctsSing
		anlz.sing = True
		if 'OUTPUT_PATH' in os.environ:
			savePath = os.environ['OUTPUT_PATH']
			if anlz.pmt1 and anlz.pmt2:
				saveName=(savePath+"/outputSing"+str(runListAll[0])+".txt")
			elif anlz.pmt1:
				saveName=(savePath+"/outputPMT1"+str(runListAll[0])+".txt")
			else:
				saveName=(savePath+"/outputPMT2"+str(runListAll[0])+".txt")
		else:
			if anlz.pmt1 and anlz.pmt2:
				saveName = ("/N/slate/frangonz/outputSing"+str(runListAll[0])+".txt")
			elif anlz.pmt1:
				saveName = ("/N/slate/frangonz/outputPMT1"+str(runListAll[0])+".txt")
			else:	
				saveName = ("/N/slate/frangonz/outputPMT2"+str(runListAll[0])+".txt")
		#saveName = ("/N/u/frangonz/BigRed3/New_MCA_Analysis/outputs/outputSing"+str(iniRun)+".txt")
	if len(ctsTrunc) == 0:
		ctsTrunc = ctsSing
		saveName = "outputDebug.txt"
			
	nwalkers,ndim,walker_init = optimized_likelihood_fit(runListAll,ctsTrunc,nMon,bkgsH,bkgsT,anlz)
	
	try: # I've moved this down here so that we pickle the updated global variables
		pool = MPIPool() # This is the master MPI check
		if not pool.is_master(): # All the pools will get called after the master loads
			pool.wait() # Wait if you're not the master
			sys.exit(0)
		# Now let's do some quick  writeouts for debug
		print("Environment:")
		print("   Config:",os.environ['ANALYSIS_CONFIG'])
		print("   Data:",os.environ['DATA_DIR'])
		
		print("RUNS:",runListAll)
		if anlz.sing:
			if anlz.pmt1 and anlz.pmt2:
				print("PMT 1+2")
			elif anlz.pmt1:
				print("PMT 1")
			else:
				print("PMT 2")
		else:
			print("COINC")
				
		mpi = True
	except ValueError:
		print("Running without MPI Pool!")
		mpi = False
	if mpi == False:
		pool = Pool(48)
	if anlz.sing: # lnL_global_(STATE) should have already pickled the data 
		sampler = emcee.EnsembleSampler(nwalkers,ndim,lnL_global_sing,pool=pool) 
	else:
		sampler = emcee.EnsembleSampler(nwalkers,ndim,lnL_global_coinc,pool=pool) 
		
	then = datetime.datetime.now()
	print("Start Time:",then)

	# Run our markov chain monte carlo
	if anlz.sing: # Run for long times for singles
		sampler.run_mcmc(walker_init,75000)
	else:
		sampler.run_mcmc(walker_init,30000)
		
	print("Elapsed Time:",(datetime.datetime.now()-then).total_seconds())
	
	# Results here
	try:
		
		# flatten out all the walkers into one list of walker points for 
		# each parameter discarding the first 500 for "burn-in" and keeping
		# only every so often, since points too nearby are correlated
		samples = sampler.get_chain(discard=3000,thin=500,flat=True)
		
		# here "blob" is any additional return argument from my lnL 
		# function, which in my case is just a chi2 goodness of fit
		blobs   = sampler.get_blobs(discard=3000,thin=500,flat=True)
		
		# Save output
		np.savetxt(saveName,np.column_stack((samples,blobs)))	
		print(runListAll)
		#print(samples)
		#print(blobs)
		#for i in range(len(samples)):
		#	print(samples[i],blobs[i]) # Putting output in stderr
			
		# Print the autocorr time -- this gives a measure
		# of how well we have converged -- if the autocorr times are too 
		# long compared to the total run time, we might have to run longer
		more_time = False
		try:
			autocorr_time = sampler.get_autocorr_time()
		except AutocorrError:
			autocorr_time = sampler.get_autocorr_time(quiet=True)
			more_time = True	
		print(autocorr_time)
		
	except Exception as e:
		print(e)
	pool.close()
	print(runListAll)
	print(autocorr_time)
	if more_time:
		print("This might not be enought autocorrelation time!!")
	sys.exit()
