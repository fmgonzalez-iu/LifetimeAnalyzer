#!/usr/local/bin/python3
#import sys
#import pdb
#import csv
#from math import *
#import numpy as np
#from statsmodels.stats.weightstats import DescrStatsW
#from scipy import stats, special
#from scipy.odr import *
#from scipy.optimize import curve_fit, nnls
#from datetime import datetime
#import matplotlib.pyplot as plt

import numpy as np
from PythonAnalyzer.classes import measurement, analyzer_cfg
from PythonAnalyzer.functions import map_height

class bkgHgtDep:
	# Height dependence for background runs
	def __init__(self,r_min,r_max):
	#def __init__(self):
		self.rMin = r_min
		self.rMax = r_max
		
		# Height dependence for backgrounds that we use
		# Unfortunately you'll have to encode additional heights yourself
		# That would also require modifying dataIO.
		self.hgts  = [10,250,380,490]
		
		# Scaling factors for our different heights
		# Recall that the first is always 1
		self.pmt1  = np.ones(len(self.hgts))
		self.pmt1E = np.zeros(len(self.hgts))
		self.pmt2  = np.ones(len(self.hgts))
		self.pmt2E = np.zeros(len(self.hgts))
		self.coinc  = np.ones(len(self.hgts))
		self.coincE = np.zeros(len(self.hgts))
	
	def is_run(self,run):
		# This tells us if we're using the right run
		if self.rMin <= run < self.rMax:
			return True
		else:
			return False

	def hF(self,hgt,pmt):
		# This is our height_factor_run() but in object form.
		
		f = measurement(1.0,0.0) # Default is 1
		for i,h in enumerate(self.hgts):
			# Loop through heights
			if h-1 < hgt < h+1:
				if pmt==0: # Coinc
					f = measurement(self.coinc[i],self.coincE[i])
				elif pmt==1: # PMT 1
					f = measurement(self.pmt1[i],self.pmt1E[i])
				elif pmt==2: # PMT 2
					f = measurement(self.pmt2[i],self.pmt2E[i])
				break # If we got the right hgt don't need to loop.
		return f

class bkgTimeDep:
	# Time dependence for background runs
	def __init__(self,r_min,r_max):
		self.rMin = r_min
		self.rMax = r_max
		
		# How many time dependent curves we want to load
		# That would also require modifying dataIO.
		self.consts = 2
		
		# Time constants as "measurement" values
		self.pmt1   = []
		self.pmt1T  = []
		self.pmt2   = []
		self.pmt2T  = []
		self.coinc  = []
		self.coincT = []
		for i in range(self.consts):
			# Load factors as 0
			self.pmt1.append(measurement(0.0,0.0))
			self.pmt2.append(measurement(0.0,0.0))
			self.coinc.append(measurement(0.0,0.0))
			# and time constants infinite
			self.pmt1T.append(measurement(np.inf,0.0))
			self.pmt2T.append(measurement(np.inf,0.0))
			self.coincT.append(measurement(np.inf,0.0))
			
	def is_run(self,run):
		# This tells us if we're using the right run
		if self.rMin <= run < self.rMax:
			return True
		else:
			return False

	def tF(self,hgt,pmt):
		# This is our time_dependence_factor() but in object form.
		# Technically this is hardcoded towards 2 as an output
		fac  = []
		t    = []
		for i in range(self.consts): # Initialization loop
			fac.append(measurement(0.0,0.0))  # Default is 0
			t.append(measurement(np.inf,0.0)) # Default is inf
		for i in range(self.consts): # Forced to scale to 490
			if pmt==0:
				fac[i] = self.coinc[i] * hgt.hF(490.,0)
				t[i]   = self.coincT[i]
			if pmt==1:
				fac[i] = self.pmt1[i] * hgt.hF(490.,1)
				t[i]   = self.pmt1T[i]
			if pmt==2:
				fac[i] = self.pmt2[i] * hgt.hF(490.,2)
				t[i]   = self.pmt2T[i]
		# Here's the hardcode to do least work
		return fac[0],t[0],fac[1],t[1]
		
class bkgStruct:
	# Structure for holding background events
	def __init__(self,run):
		# Define these based on the run
		self.run = run
		
		# Create a height and time dependence object
		self.hDep = bkgHgtDep( run, run+1) # Stored inside here
		self.tDep = bkgTimeDep(run, run+1) # And inside here
		
		# Background items:
		self.pmt1  = 0.0
		self.pmt2  = 0.0
		self.coinc = 0.0
		self.time  = 0.0
		self.dt    = 0.0
		
		# "Hidden variables"
		self.hgt   = 10.0 # Assume height is 10 unless otherwise 
		self.nruns = 1 # Number of runs the average is calculated from
	
	def __str__(self):
		
		txt  = ("run   = %d\n" % self.run)
		txt += ("pmt1  = %f\n" % self.pmt1)
		txt += ("pmt2  = %f\n" % self.pmt2)
		txt += ("coinc = %f\n" % self.coinc)
		txt += ("time  = %f" % self.dt)
		
		return txt
	
	def __add__(lhs,rhs):
		# This is actually the __average__ of two backgrounds.
		# Some time-dependence is baked into here too, to compare like-like
		
		# 1) Extrapolate to infinity to get rid of time dependence
		# Start with left hand side
		lBkg1,_x,_x = bkgFunc_obj(lhs,1,10.0,np.inf)
		lBkg2,_x,_x = bkgFunc_obj(lhs,2,10.0,np.inf)
		lBkgC,_x,_x = bkgFunc_obj(lhs,0,10.0,np.inf)
		
		# Then do right hand side
		rBkg1,_x,_x = bkgFunc_obj(rhs,1,10.0,np.inf)
		rBkg2,_x,_x = bkgFunc_obj(rhs,2,10.0,np.inf)
		rBkgC,_x,_x = bkgFunc_obj(rhs,0,10.0,np.inf)
			
		# 2) Now we're going to initialize a new object
		# which requires checking runs.
		if lhs.run == rhs.run:
			out = bkgStruct(lhs.run)
		else:
			out = bkgStruct(lhs.run)
			out.nruns = lhs.nruns + rhs.nruns # Need to scale things
		out.hDep = lhs.hDep
		out.tDep = lhs.tDep
		# 3) Now we need to scale for different lengths of time
		out.dt = lhs.dt + rhs.dt
		avgTime = (lhs.time * lhs.dt + rhs.time * rhs.dt) / out.dt
		
		# First, scale up sums by number of runs
		lSum1 = lBkg1.val * lhs.dt * lhs.nruns / out.dt
		lSum2 = lBkg2.val * lhs.dt * lhs.nruns / out.dt
		lSumC = lBkgC.val * lhs.dt * lhs.nruns / out.dt
		
		rSum1 = rBkg1.val * rhs.dt * rhs.nruns / out.dt
		rSum2 = rBkg2.val * rhs.dt * rhs.nruns / out.dt
		rSumC = rBkgC.val * rhs.dt * rhs.nruns / out.dt
			
		# 4) Check which of these work, to get averaging.
		if lhs.pmt1 > 0 and rhs.pmt1 > 0:
			out.pmt1 = lSum1 + rSum1
		elif lhs.pmt1 > 0:
			out.pmt1 = lSum1 * out.dt / lhs.dt # Rescale just in case
		elif rhs.pmt1 > 0:
			out.pmt1 = rSum1 * out.dt / rhs.dt
		# Repeat for PMT2 
		if lhs.pmt2 > 0 and rhs.pmt2 > 0:
			out.pmt2 = lSum2 + rSum2
		elif lhs.pmt2 > 0:
			out.pmt2 = lSum2 * out.dt / lhs.dt # Rescale just in case
		elif rhs.pmt2 > 0:
			out.pmt2 = rSum2 * out.dt / rhs.dt
		# And coincidences!
		if lhs.coinc > 0 and rhs.coinc > 0:
			out.coinc = lSumC + rSumC
		elif lhs.coinc > 0:
			out.coinc = lSumC * out.dt / lhs.dt # Rescale just in case
		elif rhs.coinc > 0:
			out.coinc = rSumC * out.dt / rhs.dt
		
		# Now average over the number of runs used to scale:
		out.pmt1  /= out.nruns
		out.pmt2  /= out.nruns
		out.coinc /= out.nruns
		out.dt    /= out.nruns 
		
		out.time  = np.inf # We've scaled back to infinity to do a comparison
		
		# 5) Now we need to scale rates back to the nominal.
		scale1,_x,_x = bkgFunc_obj(out,1,10.0,avgTime)
		scale2,_x,_x = bkgFunc_obj(out,2,10.0,avgTime)
		scaleC,_x,_x = bkgFunc_obj(out,0,10.0,avgTime)
		
		out.pmt1 = scale1.val
		out.pmt2 = scale2.val
		out.coinc = scaleC.val
		out.time = avgTime	
		
		return out		

#def extract_background(cts, pmt1 = True, pmt2 = True):
def extract_background(cts, cfg):#pmt1 = True, pmt2 = True):
	# Extract background parameters from a "cts" object
		
	# Background rates for PMT1, PMT2
	b1 = 0.0 # rate of counts in PMT1
	b2 = 0.0 # rate of counts in PMT2
	bc = 0.0 # rate of counts in coincidence
	bt = 0.0 # Average time of background events
	# Check that the "cts" object has the right parameters:
	dt = float(cts['bkgE'] - cts['bkgS'])
	if (dt > 0):
		if cfg.pmt1:
			b1 = float(cts['bkg1']) / float(dt)
		if cfg.pmt2:
			b2 = float(cts['bkg2']) / float(dt)
		bc = float(cts['bkgC']) / float(dt)
		bt = float(cts['bkgS'] + cts['bkgE']) / 2.0
	else:
		print("No background found for run "+str(cts['run']))
	#print(cts['bkgC'],bc,bt)
	return [b1, b2, bc, bt, dt]
	
def extract_background_obj(cts, cfg):
	# Extract background parameters from a "cts" object

	bkg = bkgStruct(cts['run']) # Trying to do this via OOP	
	try:
		bkg.dt = float(cts['bkgE'] - cts['bkgS'])
	except IndexError:
		if cfg.vb:
			print("Unable to read background on run %d" % cts['run'])
		bkg.dt = -1
		
	if (bkg.dt > 0): # If we don't have an error
		if cfg.pmt1:
			bkg.pmt1 = float(cts['bkg1']) / bkg.dt
		if cfg.pmt2:
			bkg.pmt2 = float(cts['bkg2']) / bkg.dt
		bkg.coinc = float(cts['bkgC']) / bkg.dt
		bkg.time = float(cts['bkgS'] + cts['bkgE']) / 2.0
	else:
		print("No background found for run "+str(cts['run']))
	return bkg


def extract_background_unload(cts, cfg):#pmt1 = True, pmt2 = True, maxUnl = 100, singLT = True):
#def extract_background_unload(cts, pmt1 = True, pmt2 = True, maxUnl = 100, sOrC = True):
	# Here we're going to use the last N seconds of unload as background too.
	# sOrC splits coinc and singles -- True for singles, false for coinc
	
	# Add all counts from the last bit of the last dip
	bkgSum1 = 0.0
	bkgSum2 = 0.0
	bkgSumC = 0.0
	bkgTS   = 0.0
	bkgTE   = bkgTS + 10.0 # Guess
	# Extract last dip (at bottom)
	ctsLastDip = cts[cts['dip']==(max(cts['dip']))]
	if cfg.maxUnl < 100.0: # Don't want to backtrack more than 100s
		maxUnl = 100.0
	else:
		maxUnl = cfg.maxUnl
	for i,row in enumerate(ctsLastDip):
		#if (row['te'] - ctsLastDip[0]['ts'] < maxUnl + 2): # Check max position
		#	bkgTS = row['te'] # Splits exactly on runs
		#	continue # Only take runs after maxTime
		if i < (len(ctsLastDip) - 5):
			continue
		bkgTE = row['te']
		if cfg.sing:
			if cfg.pmt1:
				bkgSum1 += float(row['d1'])# + float(row['d1DT'])
			if cfg.pmt2:
				bkgSum2 += float(row['d2'])# + float(row['d2DT'])
		else:
			bkgSumC += float(row['coinc'])# + float(row['dt'])
	
	dt = (bkgTE - bkgTS)
	bkgTime = (bkgTS + bkgTE) / 2.0
	
	bkgSum1 /= dt
	bkgSum2 /= dt
	bkgSumC /= dt
	
	return bkgSum1, bkgSum2, bkgSumC, bkgTime, dt


def extract_background_unload_obj(cts, cfg):
	# Here we're going to use the last N seconds of unload as background too.
	
	# initialize output object
	bkg = bkgStruct(cts['run'][0]) # This is also being upgraded to OOP
	bkg.hgt = map_height(cfg)[(max(cts['dip']))]
		
	ts = 0.0 # Guess the ending times
	te = ts + 10.0 # Guess
			
	# Extract last dip (at bottom)
	ctsLastDip = cts[cts['dip']==(max(cts['dip']))]
	for i,row in enumerate(ctsLastDip):
		#if (row['te'] - ctsLastDip[0]['ts'] < cfg.maxUnl + 1): # Check max position
		#	ts = row['te'] # Splits exactly on runs
		#	continue # Only take runs after maxTime
		if i < (len(ctsLastDip) - 4):
			ts = row['te']
			continue
		te = row['te']
		
		# Disadvantage here is that we can't load coincidence and singles simultaneously
		if cfg.sing:
			if cfg.pmt1:
				bkg.pmt1 += float(row['d1'])
			if cfg.pmt2:
				bkg.pmt2 += float(row['d2'])
		else:
			bkg.coinc += float(row['coinc'])
	
	bkg.dt   = (te - ts)
	bkg.time = (te + ts) / 2.
	bkg.pmt1  /= bkg.dt
	bkg.pmt2  /= bkg.dt
	bkg.coinc /= bkg.dt
	
	return bkg
	#return bkgSum1, bkgSum2, bkgSumC, bkgTime, dt
		
def extract_average_background(run,cts,mon,hDep,cfg,runList = [],runBreaks = []):
	# This averages the background over some runs.
	# I don't like smoothing out the runs but I think I have to to get 
	# a realistic background average...
	#
	# The reason being that bkg_{coinc, min} = 1/(t_{bkg})*t_{count}
	# 
	# For a 1 neutron precision counting pks 1+2+3 (t_{count} = 160.0), 
	# t_{bkg} = 160.0. But we only realistically have ~90s of usable background
	# which means our background can't be known to a single UCN of precision. (~0.08 s)
	#
	# This is necessary for singles too, because again, a single UCN of precision
	# produces ~30 PEs, or 160.0*30=4800 PE fluctuation over the unload
	#-------------------------------------------------------------------

	# Figure out where the boundaries are
	minB = cfg.bkgWin
	maxB = cfg.bkgWin		
	if len(runBreaks) > 0:
		for i,r in enumerate(runBreaks):
			if i==0:
				continue
			if runBreaks[i-1] <= run and r > run:
				if not ((run-runBreaks[i-1]) >= cfg.bkgWin):
					minB = run-runBreaks[i-1] # Diff between run and lower break
				if not ((r - run) > cfg.bkgWin):
					maxB = r - run # Diff between upper break and run
	if len(runList) == 0:
		runS = range(run-minB,run+maxB+1)
	else:
		runList = np.array(runList)
		runS = runList[(runList >=run-minB)*(runList <= run+maxB)]
		
	bkgSum1 = 0.0
	bkgSum2 = 0.0
	bkgSumC = 0.0
	
	bkgSum1_C = 0.0
	bkgSum2_C = 0.0
	bkgSumC_C = 0.0
	bkgTime = 0.0
	nRuns = 0
	#print (runS)
	for r in runS:
		
		nMonRaw = mon[mon['run']==r]
		ctsRaw  = cts[cts['run']==r]
		
		if len(nMonRaw) > 0 and len(ctsRaw) > 0:
			nRuns += 1
			[bkg1_E,bkg2_E,bkgC_E,bkgT_E,dt_E] = extract_background(nMonRaw,cfg)#.pmt1,cfg.pmt2)			
			if cfg.maxUnl < 140:
				[bkg1_U,bkg2_U,bkgC_U,bkgT_U,dt_U] = extract_background_unload(ctsRaw,cfg)#.pmt1,cfg.pmt2,cfg.maxUnl,cfg.sing)
			else:
				[bkg1_U,bkg2_U,bkgC_U,bkgT_U,dt_U] = [0.0,0.0,0.0,0.0,0.0]
			
			# Since these average backgrounds are taken at different times,
			# need to extrapolate to infinity.
			bkg1_E_C = bkgFunc(bkg1_E,hDep, r, 1, 10.0, 10.0, np.inf, bkgT_E).val
			bkg1_U_C = bkgFunc(bkg1_U,hDep, r, 1, 10.0, 10.0, np.inf, bkgT_U).val
			bkg2_E_C = bkgFunc(bkg2_E,hDep, r, 2, 10.0, 10.0, np.inf, bkgT_E).val
			bkg2_U_C = bkgFunc(bkg2_U,hDep, r, 2, 10.0, 10.0, np.inf, bkgT_U).val
			# I don't do a time dependent coincidence, but if you wanted to put one in here's where it'd go
			bkgC_E_C = bkgFunc(bkgC_E,hDep, r, 0, 10.0, 10.0, np.inf, bkgT_E).val
			bkgC_U_C = bkgFunc(bkgC_U,hDep, r, 0, 10.0, 10.0, np.inf, bkgT_U).val
			
			
			# Since different PMTs might be on at different time, correction required here.
			if bkg1_E > 0 and bkg1_U > 0:
				bkgSum1   += (bkg1_E  *dt_E+bkg1_U  *dt_U)/(dt_E+dt_U)
				bkgSum1_C += (bkg1_E_C*dt_E+bkg1_U_C*dt_U)/(dt_E+dt_U)
			elif bkg1_E > 0:
				bkgSum1   += bkg1_E
				bkgSum1_C += bkg1_E_C
			elif bkg1_U > 0:
				bkgSum1   += bkg1_U
				bkgSum1_C += bkg1_U_C
						
			if bkg2_E > 0 and bkg2_U > 0:
				bkgSum2   += (bkg2_E  *dt_E+bkg2_U  *dt_U)/(dt_E+dt_U)
				bkgSum2_C += (bkg2_E_C*dt_E+bkg2_U_C*dt_U)/(dt_E+dt_U)
			elif bkg2_E > 0:
				bkgSum2   += bkg2_E
				bkgSum2_C += bkg2_E_C
			elif bkg2_U > 0:
				bkgSum2   += bkg2_U
				bkgSum2_C += bkg2_U_C
			
			if bkgC_E > 0 and bkgC_U > 0:
				bkgSumC   += (bkgC_E  *dt_E+bkgC_U  *dt_U)/(dt_E+dt_U)
				bkgSumC_C += (bkgC_E_C*dt_E+bkgC_U_C*dt_U)/(dt_E+dt_U)
			elif bkgC_E > 0:
				bkgSumC   += bkgC_E
				bkgSumC_C += bkgC_E_C
			elif bkgC_U > 0:
				bkgSumC   += bkgC_U
				bkgSumC_C += bkgC_U_C
			# And assume the average time here is going to be the mean.
			# We can just return bkgTime = np.inf if we want to extrapolate.	
			if bkgT_E > 0 and bkgT_U > 0:
				bkgTime += (bkgT_E+bkgT_U)/2.0
			elif bkgT_E > 0:
				bkgTime += bkgT_E
			elif bkgT_U > 0:
				bkgTime += bkgT_U
	
	if nRuns > 0:
		# I'm returning the background at infinity -- should make like-to-like comps better.
		#bkgSum1 /= float(nRuns)
		#bkgSum2 /= float(nRuns)
		#bkgSumC /= float(nRuns)
		if cfg.pmt1:
			bkgSum1_C /= float(nRuns)
		else:
			bkgSum1_C = 0.0
		if cfg.pmt2:
			bkgSum2_C /= float(nRuns)
		else:
			bkgSum2_C = 0.0
		bkgSumC_C /= float(nRuns)
		bkgTime /= float(nRuns)	
		#print(run,bkgSum1-bkgSum1_C, bkgSum2-bkgSum2_C,bkgSum1,bkgSum2)
		#return bkgSum1_C,bkgSum2_C,bkgSumC_C,bkgTime
		return bkgSum1_C,bkgSum2_C,bkgSumC_C,np.inf
	else:
		return 0.,0.,0.,0.


def bkgFunc_obj(bkg,pmt = 0,h_i = 10.0,ti_f = np.inf):
	# This is the generic background function. 
	# Need to fit to alpha, beta, t1, t2.
	#
	# For a given time, have R = f(h)*(a0 + a1*e^(-t/t1) + a2*e^(-t/t2))
	# 
	# End of run thus gives a0 = R_end / f(h_end) - a1*e^(-tf/t1) + a2*e^(-tf/t2)
	#
	# So R(h,t) = f(h)*(R_end / f(h_end)) 
	#              + a1*(f(h)*e^(-t/t1) - f(h_end)*e^(-t/t1)) 
	#              + a2*(f(h)*e^(-t/t2) - f(h_end)*e^(-t/t2))
	#                
	# Requires measured bkg (r_m), run number, and pmt.
	# 
	# ti_f is the time we want to calculate the background at
	# tf_f is the time we measured the background.
	
	# Extract factors for measured point
	a, t1, b, t2 = bkg.tDep.tF(bkg.hDep,pmt)
	#a, t1, b, t2 = time_dependence_run(bkg.run, pmt,bkg.hDep)
	A_i = bkg.hDep.hF(h_i,pmt) # height dependence stored in object
#	A_i = height_factor_run(bkg.run, h_i, pmt)
	ti = measurement(ti_f,0.0) # Convert time to meas.
	
	# Scale background to point
	A_f = bkg.hDep.hF(bkg.hgt, pmt)
	#A_f = height_factor_run(bkg.run, bkg.hgt, pmt)
	tf = measurement(bkg.time,0.0)
	if pmt == 1: 
		r_m = bkg.pmt1
	elif pmt == 2:
		r_m = bkg.pmt2
	elif pmt == 0:
		r_m = bkg.coinc
	else:
		sys.exit("ERROR! You're scaling backgrounds for a non-existent PMT!")
	rate_h = (measurement(r_m,np.sqrt(r_m)) / A_f)
	
	#try:
	if np.isfinite(t1.val) and (t1.val > 0): # numpy doesn't like infinite time constants
		a_val = a*((-ti/t1).exp() - A_f*(-tf/t1).exp()/A_i)
	else:
		a_val = a*(measurement(1.,0.) - A_f/A_i)
	if np.isfinite(t2.val) and (t2.val > 0): # Again, avoid infinite time constants.
		b_val = b*((-ti/t2).exp() - A_f*(-tf/t2).exp()/A_i)
	else:
		b_val = b*(measurement(1.,0.) - A_f/A_i)
	rate_t = a_val + b_val # Time dependence is function of these two
	#else:
	#	rate_t = measurement(0.0,0.0)
	#except RuntimeWarning:
	#	print("Warning! Possible Overflow Error!",str(run))
		# Error here is probably an overflow error
	#	rate_t = measurement(0.0,np.inf)\
	bkgTot = A_i*(rate_h + rate_t)
	bkgPos = A_i / A_f * measurement(r_m,np.sqrt(r_m)) - measurement(r_m,np.sqrt(r_m))
	bkgPos.err = bkgPos.val*np.sqrt(r_m + (A_i/A_f).err*(A_i/A_f).err)
	bkgTime = A_i * rate_t
	bkgTime.err = np.sqrt(bkgTime.err*bkgTime.val) # Fractional time dependence
	
	return bkgTot, bkgPos, bkgTime

def bkgFunc(r_m, hDep, run, pmt,
			hi = 10.0, hf = 10.0, 
			ti_f = np.inf, tf_f = np.inf):
	# This is the generic background function. 
	# Need to fit to alpha, beta, t1, t2.
	#
	# For a given time, have R = f(h)*(a0 + a1*e^(-t/t1) + a2*e^(-t/t2))
	# 
	# End of run thus gives a0 = R_end / f(h_end) - a1*e^(-tf/t1) + a2*e^(-tf/t2)
	#
	# So R(h,t) = f(h)*(R_end / f(h_end)) 
	#              + a1*(f(h)*e^(-t/t1) - f(h_end)*e^(-t/t1)) 
	#              + a2*(f(h)*e^(-t/t2) - f(h_end)*e^(-t/t2))
	#                
	# Requires measured bkg (r_m), run number, and pmt.
	# 
	# ti_f is the time we want to calculate the background at
	# tf_f is the time we measured the background.
	
	#A_i = height_factor_run(run, hi, pmt)
	#A_f = height_factor_run(run, hf, pmt)
	A_i = hDep.hF(hi,pmt)
	A_f = hDep.hF(hf,pmt)
	a, t1, b, t2 = time_dependence_run(run, pmt,hDep)
	ti = measurement(ti_f,0.0)
	tf = measurement(tf_f,0.0)
	if r_m > 0:
		rate_h = (measurement(r_m,np.sqrt(r_m)) / A_f)
	else:		
		rate_h = (measurement(0.0,0.0) / A_f)
	#print(measurement(r_m,np.sqrt(r_m)),rate_h)
	#print(r_m,r_m-(A_i*rate_h).val)
	try:
		rate_t = a*((-ti/t1).exp() - A_f*(-tf/t1).exp()/A_i) + b*((-ti/t2).exp() - A_f*(-tf/t2).exp()/A_i)
		#rate_t = a*(np.exp(-ti/t1) - A_f*np.exp(-tf/t1)/A_i) + b*(np.exp(-ti/t2) - A_f*np.exp(-tf/t2)/A_i)
		
		#print(run)
		#print("   ",(-ti/t1).exp(),ti,t1,ti/t1)		
		#print("   ",np.exp(-tf/t1),np.exp(-tf/t2))
		#print("   ",a, b)
	except RuntimeWarning:
		print(str(ti)+" "+str(t1)+" "+str(tf)+" "+str(t2)+" "+str(run))
		rate_t = a*((-ti/t1).exp() - A_f*(-tf/t1).exp()/A_i) + b*((-ti/t2).exp() - A_f*(-tf/t2).exp()/A_i)
		#rate_t = a*(np.exp(-ti/t1) - A_f*np.exp(-tf/t1)/A_i) + b*(np.exp(-ti/t2) - A_f*np.exp(-tf/t2)/A_i)
	#rate_t = measurement(0.0,0.0)	
	#print(A_i*(rate_h+rate_t))
	return A_i*(rate_h + rate_t)
	
	
def height_factor_run(run, height = 10.0, pmt = 0):
	# This is just a lookup table of height dependence factors.
	# Calculated for PMT1/2. Coinc is "unknown" case [0 = coinc]
	#
	# If in doubt there's no height dependence.
	#if height == 250.0:
	#	height = 490.0
	#elif height == 490.0:
	#	height = 250.0
	if 9.0 < height < 11.0:
		return measurement(1.0,0.0)
	
	factor = []
	if 4200 <= run < 7327: 
		if 489.0 < height < 491.0:
			factor.append(measurement(0.923,0.017)) # Coinc
			factor.append(measurement(1.0221,0.0013)) # PMT 1
			factor.append(measurement(1.0594,0.0020)) # PMT 2
		elif 379.0 < height < 381.0:
			factor.append(measurement(1.007,0.018)) # Coinc
			factor.append(measurement(1.005,0.0012)) # PMT 1
			factor.append(measurement(1.0406,0.0021)) # PMT 2				
		elif 249.0 < height < 251.0:
			factor.append(measurement(1.013,0.0021)) # Coinc
			factor.append(measurement(0.9935,0.0014)) # PMT 1
			factor.append(measurement(1.0196,0.0018)) # PMT 2	
	elif 7327 <= run < 9545: 
		if 489.0 < height < 491.0:
			factor.append(measurement(0.940,0.016)) # Coinc
			factor.append(measurement(1.1536,0.0042)) # PMT 1
			factor.append(measurement(1.1220,0.0028)) # PMT2	
		elif 379.0 < height < 381.0:
			factor.append(measurement(0.997,0.016)) # Coinc
			factor.append(measurement(1.0210,0.0019)) # PMT 1
			factor.append(measurement(1.0691,0.0025)) # PMT 2
		elif 249.0 < height < 251.0:
			factor.append(measurement(0.985,0.017)) # Coinc
			factor.append(measurement(1.0172,0.0016)) # PMT 1
			factor.append(measurement(1.0428,0.0022)) # PMT 2
	elif 9545 <= run < 13309: 
		if 489.0 < height < 491.0:
			factor.append(measurement(0.9353,0.0075)) # Coinc
			factor.append(measurement(1.0180,0.0023)) # PMT 1
			factor.append(measurement(1.0098,0.0011)) # PMT 2
		elif 379.0 < height < 381.0:
			factor.append(measurement(0.9525,0.0081)) # Coinc
			factor.append(measurement(1.0150,0.0024)) # PMT 1
			factor.append(measurement(1.0051,0.0013)) # PMT 2
		elif 249.0 < height < 251.0:
			factor.append(measurement(0.9435,0.0074)) # Coinc
			factor.append(measurement(1.0043,0.0019)) # PMT 1
			factor.append(measurement(0.9833,0.0011)) # PMT 2
	elif 13309 <= run < 15000: 
		if 489.0 < height < 491.0:
			factor.append(measurement(0.9862,0.0148)) # Coinc
			factor.append(measurement(1.0251,0.0019)) # PMT 1
			factor.append(measurement(1.0259,0.0016)) # PMT 2	
		elif 379.0 < height < 381.0:
			factor.append(measurement(0.9983,0.0148)) # Coinc
			factor.append(measurement(1.0203,0.0017)) # PMT 1
			factor.append(measurement(1.0284,0.0016)) # PMT 2	
		elif 249.0 < height < 251.0:
			factor.append(measurement(1.0005,0.0133)) # Coinc
			factor.append(measurement(0.9988,0.0017)) # PMT 1
			factor.append(measurement(1.0002,0.0014)) # PMT 2	
	
	if pmt >= len(factor): # Unknown case assumes at bottom, which is always 1.0.
		return measurement(1.0,0.0)
	else:
		#print (run,height,pmt,factor[pmt])
		#f_test = measurement(1.0,0.0) + (measurement(1.0,0.0) - factor[pmt])*measurement(249.0,0.0)
		#return measurement(1.0,0.0)/factor[pmt]
		return factor[pmt]

def time_dependence_run(run, pmt,hDep):
	# Again this is a lookup table for time dependence runs
	# There are now 4 factors to look up here for each run/PMT.
	#
	# These numbers are "raw" meaning un-height-corrected for the time dep.
	# We height-correct the scaling factor at the end.
	#
	# I'm also assuming there's no time dependence in coincidences (for now.)
	# To turn on coincidences I'd just add an else for PMTs.
	#
	# I should probably re-do the math on these

	a  = measurement(0.0,0.0)
	t1 = measurement(np.inf,0.0)
	b  = measurement(0.0,0.0)
	t2 = measurement(np.inf,0.0)
	
	if 4200 <= run < 7327: 
		if pmt == 1:
			a  = measurement(0.0009514,0.0002624)
			t1 = measurement(35.58,8.35)
			b  = measurement(0.002028,0.000052)
			t2 = measurement(491.2,14.0)
		elif pmt == 2:
			a  = measurement(0.000304,0.000019)
			t1 = measurement(136.0,18.2)
			b  = measurement(0.0007963,0.0000155)
			t2 = measurement(2306.0,101.0)
	elif 7327 <= run < 9545:
		if pmt == 1:
			a  = measurement(0.0001638,0.0000042)
			t1 = measurement(110.2,6.8)
			b  = measurement(0.0006133,0.0000026)
			t2 = measurement(5490,123.7)
		elif pmt == 2:
			a  = measurement(0.0003004,0.000079)
			t1 = measurement(70.54,3.58)
			b  = measurement(0.0007014,0.000031)
			t2 = measurement(3107,46.5)
	elif 9545 <= run < 13309:
		if pmt == 1:
			a  = measurement(0.000611,0.0000035)
			t1 = measurement(31.36,2.54)
			b  = measurement(0.001348,0.000010)
			t2 = measurement(851.8,9.4)
		elif pmt == 2:
			a  = measurement(0.001744,0.000105)
			t1 = measurement(33.23,2.36)
			b  = measurement(0.0027538,0.000046)
			t2 = measurement(364.3,6.4)*3
	elif 13309 <= run < 15000: # These numbers should be re-calculated
		if pmt == 1:
			a  = measurement(0.0006849,0.00000428)
			t1 = measurement(98.27,7.91)
			b  = measurement(0.001191,0.000027)
			t2 = measurement(940.1,26.6)
		elif pmt == 2:
			a  = measurement(0.0006847,0.0000428)
			t1 = measurement(98.26,7.91)
			b  = measurement(0.001191,0.000027)
			t2 = measurement(939.2,26.6)
	
	# Need to correct for height factor here and scale to Hz
	#a = a * height_factor_run(run,490.0,pmt) * measurement(1550.0,0.0)
	#b = b * height_factor_run(run,490.0,pmt) * measurement(1550.0,0.0)
	a = a * hDep.hF(490.0,pmt) * measurement(1550.0,0.0)
	b = b * hDep.hF(490.0,pmt) * measurement(1550.0,0.0)
	
	return a, t1, b, t2
	
# def dBKG1(height, time, rateBkg, run, bkgT = np.inf):
	
	# tScale1 = measurement(0.0,0.0)
	# tc1     = measurement(0.0,0.0)
	# tScale2 = measurement(0.0,0.0)
	# tc2     = measurement(0.0,0.0)
	# if 4200 <= run < 7327:
		# if height == 10.0:
			# factor = measurement(1.0,0.0)
		# elif height == 250.0:
			# factor = measurement(0.9935,0.0014)
		# elif height == 380.0:
			# factor = measurement(1.005,0.0012)
		# elif height == 490.0:
			# factor = measurement(1.0221,0.0013)
		# else:
			# print("Unknown height dependence! on run "+str(run)+" "+str(height))
			# return measurement(0.0,0.0)
	# elif 7327 <= run < 9545:
		# if height == 10.0:
			# factor = measurement(1.0,0.0)
		# elif height == 250.0:
			# factor = measurement(1.0172,0.0016)
		# elif height == 380.0:
			# factor = measurement(1.0210,0.0019)
		# elif height == 490.0:
			# factor = measurement(1.1536,0.0042)
		# else:
			# print("Unknown height dependence! on run "+str(run)+" "+str(height))
			# return measurement(0.0,0.0)
	# elif 9545 <= run < 13309:
		# if height == 10.0:
			# factor = measurement(1.0,0.0)
		# elif height == 250.0:
			# factor = measurement(1.0043,0.0019)
		# elif height == 380.0:
			# factor = measurement(1.0150,0.0024)
		# elif height == 490.0:
			# factor = measurement(1.0180,0.0023)
		# else:
			# print("Unknown height dependence! on run "+str(run)+" "+str(height))
			# return measurement(0.0,0.0)
	# elif 13309 <= run < 15000:
		# if height == 10.0:
			# factor = measurement(1.0,0.0)
		# elif height == 250.0:
			# factor = measurement(0.9988,0.0017)
		# elif height == 380.0:
			# factor = measurement(1.0203,0.0017)
		# elif height == 490.0:
			# factor = measurement(1.0251,0.0019)
		# else:
			# print("Unknown height dependence!")
			# return measurement(0.0,0.0)
	# else: 
		# factor = measurement(1.0,0.0)
	
	# dt = time
	
	# bkg_f    = rateBkg * factor.val
	# #bkgErr_f = np.sqrt(1/rateBkg + (factor.err/factor.val)**2)
	# #return measurement(bkg_f,0.0)
	# bkgMeas  = measurement(rateBkg, sqrt(rateBkg))
	# return bkgMeas*factor
	
# def dBKG2(height, time, rateBkg, run, bkgT = np.inf):
	
	# tScale1 = measurement(0.0,0.0)
	# tc1     = measurement(0.0,0.0)
	# tScale2 = measurement(0.0,0.0)
	# tc2     = measurement(0.0,0.0)
	# if 4200 <= run < 7327:		
		# if height == 10.0:
			# factor = measurement(1.0,0.0)
		# elif height == 250.0:
			# factor = measurement(1.0196,0.0018)
		# elif height == 380.0:
			# factor = measurement(1.0406,0.0021)
		# elif height == 490.0:
			# factor = measurement(1.0594,0.0020)
		# else:
			# print("Unknown height dependence! on run "+str(run)+" "+str(height))
			# return measurement(0.0,0.0)
	# elif 7327 <= run < 9545:
		# if height == 10.0:
			# factor = measurement(1.0,0.0)
		# elif height == 250.0:
			# factor = measurement(1.0428,0.0022)
		# elif height == 380.0:
			# factor = measurement(1.0691,0.0025)
		# elif height == 490.0:
			# factor = measurement(1.1220,0.0028)
		# else:
			# print("Unknown height dependence! on run "+str(run)+" "+str(height))
			# return measurement(0.0,0.0)	
	# elif 9545 <= run < 13309:
		# if height == 10.0:
			# factor = measurement(1.0,0.0)
		# elif height == 250.0:
			# factor = measurement(0.9833,0.0011)
		# elif height == 380.0:
			# factor = measurement(1.0051,0.0013)
		# elif height == 490.0:
			# factor = measurement(1.0098,0.0011)
		# else:
			# print("Unknown height dependence!")
			# return measurement(0.0,0.0)
	# elif 13309 <= run < 15000:
		# if height == 10.0:
			# factor = measurement(1.0,0.0)
		# elif height == 250.0:
			# factor = measurement(1.0002,0.0014)
		# elif height == 380.0:
			# factor = measurement(1.0284,0.0016)
		# elif height == 490.0:
			# factor = measurement(1.0259,0.0016)
		# else:
			# print("Unknown height dependence!")
			# return measurement(0.0,0.0)
	# else:
		# factor = measurement(1.0,0.0)
	
	# dt = time
	# bkg_f    = rateBkg * factor.val
	# #bkgErr_f = np.sqrt(((rateBkg * factor.err)/factor.val)**2)
	# #return measurement(bkg_f, 0.0)
	# bkgMeas  = measurement(rateBkg, sqrt(rateBkg))
	# return bkgMeas*factor
	# #return measurement(bkg_f, bkgErr_f)
	
# def dBKGc(height, time, rateBkg, run, bkgT = np.inf):
	# # No time dependence in coincidences
	# if 4200 <= run < 7327:
		# if height == 10.0:
			# factor = measurement(1.0,0.0)
		# elif height == 250.0:
			# factor = measurement(1.013,0.0021)
		# elif height == 380.0:
			# factor = measurement(1.007,0.018)
		# elif height == 490.0:
			# factor = measurement(0.923,0.017)
		# else:
			# print("Unknown height dependence! on run "+str(run)+" "+str(height))
			# return measurement(0.0,0.0)
	# elif 7327 <= run < 9545:
		# if height == 10.0:
			# factor = measurement(1.0,0.0)
		# elif height == 250.0:
			# factor = measurement(0.985,0.017)
		# elif height == 380.0:
			# factor = measurement(0.997,0.016)
		# elif height == 490.0:
			# factor = measurement(0.940,0.016)
		# else:
			# print("Unknown height dependence! on run "+str(run)+" "+str(height))
			# return measurement(0.0,0.0)	
	# elif 9545 <= run < 13309:
		# if height == 10.0:
			# factor = measurement(1.0,0.0)
		# elif height == 250.0:
			# factor = measurement(0.9435,0.0074)
		# elif height == 380.0:
			# factor = measurement(0.9525,0.0081)
		# elif height == 490.0:
			# factor = measurement(0.9353,0.0075)
		# else:
			# print("Unknown height dependence!")
			# return measurement(0.0,0.0)
	# elif 13309 <= run < 15000:
		# if height == 10.0:
			# factor = measurement(1.0,0.0)
		# elif height == 250.0:
			# factor = measurement(1.0005,0.0133)
		# elif height == 380.0:
			# factor = measurement(0.9983,0.0148)
		# elif height == 490.0:
			# factor = measurement(0.9862,0.0148)
		# else:
			# print("Unknown height dependence!")
			# return measurement(0.0,0.0).val
	# else:
		# factor = measurement(1.0,0.0)
	
	# dt = time
	# # bkg_f    = rateBkg * factor.val
	# # bkgErr_f = np.sqrt(((rateBkg * factor.err)/factor.val)**2)
	# bkgMeas  = measurement(rateBkg, sqrt(rateBkg))
	# return bkgMeas * factor
	# #return measurement(bkg_f,0.0)
	# #return measurement(bkg_f, 0.0)
	# #return measurement(bkg_f, bkgErr_f)

# def extract_average_background(run,cts,mon,runBreaks = [],win=2,maxUnl=100,pmt1=True,pmt2=True,sing=True,runList=[]):
	# # This averages the background over some runs.
	# # I don't like smoothing out the runs but I think I have to to get 
	# # a realistic background average...
	# #
	# # The reason being that bkg_{coinc, min} = 1/(t_{bkg})*t_{count}
	# # 
	# # For a 1 neutron precision counting pks 1+2+3 (t_{count} = 160.0), 
	# # t_{bkg} = 160.0. But we only realistically have ~90s of usable background
	# # which means our background can't be known to a single UCN of precision. (~0.08 s)
	# #
	# # This is necessary for singles too, because again, a single UCN of precision
	# # produces ~30 PEs, or 160.0*30=4800 PE fluctuation over the unload
	# #-------------------------------------------------------------------

	# # Figure out where the boundaries are
	# minB = win
	# maxB = win
	# if len(runBreaks) > 0:
		# for i,r in enumerate(runBreaks):
			# if i==0:
				# continue
			# if runBreaks[i-1] <= run and r > run:
				# if not ((run-runBreaks[i-1]) >= win):
					# minB = run-runBreaks[i-1] # Diff between run and lower break
				# if not ((r - run) > win):
					# maxB = r - run # Diff between upper break and run
	# if len(runList) == 0:
		# runS = range(run-minB,run+maxB+1)
	# else:
		# runList = np.array(runList)
		# runS = runList[(runList >=run-minB)*(runList <= run+maxB)]
		
	# bkgSum1 = 0.0
	# bkgSum2 = 0.0
	# bkgSumC = 0.0
	
	# bkgSum1_C = 0.0
	# bkgSum2_C = 0.0
	# bkgSumC_C = 0.0
	# bkgTime = 0.0
	# nRuns = 0
	# #print (runS)
	# for r in runS:
		
		# nMonRaw = mon[mon['run']==r]
		# ctsRaw  = cts[cts['run']==r]
		
		# if len(nMonRaw) > 0 and len(ctsRaw) > 0:
			# nRuns += 1
			# [bkg1_E,bkg2_E,bkgC_E,bkgT_E,dt_E] = extract_background(nMonRaw,pmt1,pmt2)			
			# if maxUnl < 140:
				# [bkg1_U,bkg2_U,bkgC_U,bkgT_U,dt_U] = extract_background_unload(ctsRaw,pmt1,pmt2,maxUnl,sing)
			# else:
				# [bkg1_U,bkg2_U,bkgC_U,bkgT_U,dt_U] = [0.0,0.0,0.0,0.0,0.0]
			
			# # Since these average backgrounds are taken at different times,
			# # need to extrapolate to infinity.
			# bkg1_E_C = bkgFunc(bkg1_E, r, 1, 10.0, 10.0, np.inf, bkgT_E).val
			# bkg1_U_C = bkgFunc(bkg1_U, r, 1, 10.0, 10.0, np.inf, bkgT_U).val
			# bkg2_E_C = bkgFunc(bkg2_E, r, 2, 10.0, 10.0, np.inf, bkgT_E).val
			# bkg2_U_C = bkgFunc(bkg2_U, r, 2, 10.0, 10.0, np.inf, bkgT_U).val
			# # I don't do a time dependent coincidence, but if you wanted to put one in here's where it'd go
			# bkgC_E_C = bkgFunc(bkgC_E, r, 0, 10.0, 10.0, np.inf, bkgT_E).val
			# bkgC_U_C = bkgFunc(bkgC_U, r, 0, 10.0, 10.0, np.inf, bkgT_U).val
			
			
			# # Since different PMTs might be on at different time, correction required here.
			# if bkg1_E > 0 and bkg1_U > 0:
				# bkgSum1   += (bkg1_E  *dt_E+bkg1_U  *dt_U)/(dt_E+dt_U)
				# bkgSum1_C += (bkg1_E_C*dt_E+bkg1_U_C*dt_U)/(dt_E+dt_U)
			# elif bkg1_E > 0:
				# bkgSum1   += bkg1_E
				# bkgSum1_C += bkg1_E_C
			# elif bkg1_U > 0:
				# bkgSum1   += bkg1_U
				# bkgSum1_C += bkg1_U_C
						
			# if bkg2_E > 0 and bkg2_U > 0:
				# bkgSum2   += (bkg2_E  *dt_E+bkg2_U  *dt_U)/(dt_E+dt_U)
				# bkgSum2_C += (bkg2_E_C*dt_E+bkg2_U_C*dt_U)/(dt_E+dt_U)
			# elif bkg2_E > 0:
				# bkgSum2   += bkg2_E
				# bkgSum2_C += bkg2_E_C
			# elif bkg2_U > 0:
				# bkgSum2   += bkg2_U
				# bkgSum2_C += bkg2_U_C
			
			# if bkgC_E > 0 and bkgC_U > 0:
				# bkgSumC   += (bkgC_E  *dt_E+bkgC_U  *dt_U)/(dt_E+dt_U)
				# bkgSumC_C += (bkgC_E_C*dt_E+bkgC_U_C*dt_U)/(dt_E+dt_U)
			# elif bkgC_E > 0:
				# bkgSumC   += bkgC_E
				# bkgSumC_C += bkgC_E_C
			# elif bkgC_U > 0:
				# bkgSumC   += bkgC_U
				# bkgSumC_C += bkgC_U_C
			# # And assume the average time here is going to be the mean.
			# # We can just return bkgTime = np.inf if we want to extrapolate.	
			# if bkgT_E > 0 and bkgT_U > 0:
				# bkgTime += (bkgT_E+bkgT_U)/2.0
			# elif bkgT_E > 0:
				# bkgTime += bkgT_E
			# elif bkgT_U > 0:
				# bkgTime += bkgT_U
	
	# if nRuns > 0:
		# # I'm returning the background at infinity -- should make like-to-like comps better.
		# #bkgSum1 /= float(nRuns)
		# #bkgSum2 /= float(nRuns)
		# #bkgSumC /= float(nRuns)
		# if pmt1:
			# bkgSum1_C /= float(nRuns)
		# else:
			# bkgSum1_C = 0.0
		# if pmt2:
			# bkgSum2_C /= float(nRuns)
		# else:
			# bkgSum2_C = 0.0
		# bkgSumC_C /= float(nRuns)
		# bkgTime /= float(nRuns)	
		# #print(run,bkgSum1-bkgSum1_C, bkgSum2-bkgSum2_C,bkgSum1,bkgSum2)
		# #return bkgSum1_C,bkgSum2_C,bkgSumC_C,bkgTime
		# return bkgSum1_C,bkgSum2_C,bkgSumC_C,np.inf
	# else:
		# return 0.,0.,0.,0.
