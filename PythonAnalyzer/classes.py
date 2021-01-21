#!/usr/local/bin/python3
#import sys
#import pdb
#import csv
#from math import *
import numpy as np
#from statsmodels.stats.weightstats import DescrStatsW
#from scipy import stats, special
#from scipy.odr import *
#from scipy.optimize import curve_fit, nnls
#from datetime import datetime
#import matplotlib.pyplot as plt

# Two conversion functions for lists
def convert_float_to_meas(floV,floE = []):
	# Convert float lists (with or without errors) to measurements
	measL = []
	if len(floE) == len(floV):
		for i, m in enumerate(floV):
			measL.append(m,floE[i])
	else:
		for m in floV:
			measL.append(m,0.0)
	return measL
	
def convert_meas_to_float(measL):
	# Convert measurement lists to floats	
	lVal = []
	lErr = []

	for m in measL:
		lVal.append(m.val)
		lErr.append(m.err)
	
	return lVal, lErr


class measurement: # Definition of Measurement Class
	# Create the measurement class -- it has error propagation routines
	# Any basic operation (add, subtract, multiply, divide) with a non-measurement
	# value will be treated as a measurement with zero uncertainty
	
	def __init__(self, val, err):
		self.val = val
		self.err = err
	
	def __float__(self):
		return np.float(self.val)
	
	def __str__(self):
		return "{"+str(self.val)+" +/- "+str(self.err)+"}"
	
	def __repr__(self):
		return str(self)
		
	def __neg__(self):
		return measurement(-self.val,self.err)
		
	def __setitem__ (self, rhs):
		# For assigning lists
		self.val = rhs.val
		self.err = rhs.err
		
	def __round__ (self):
		return np.round(self.val)
	
	#-------------------------------------------------------------------
	# Basic Math Functions
	#-------------------------------------------------------------------
	def __add__(lhs, rhs):	
		try: # Modified so that typeerrors automatically create new measurements with error 0
			return measurement(lhs.val+rhs.val, 
								np.sqrt(lhs.err*lhs.err+rhs.err*rhs.err))
		except AttributeError:
			return measurement(lhs.val + rhs, lhs.err)	
	
	def __sub__(lhs, rhs):
		try:
			return measurement(lhs.val-rhs.val, 
								np.sqrt(lhs.err*lhs.err+rhs.err*rhs.err))
		except AttributeError:
			return measurement(lhs.val-rhs, lhs.err*lhs.err)
			
	def __mul__(lhs, rhs):	
		if -1E-12 < lhs.val < 1E-12:  # Errors need to be relative, but if a value
			rel = 0.0				  # is zero we can't do that.
		else:
			rel = np.power(lhs.err/lhs.val,2)
		try: # Multiplication by other measurement
			if -1E-12 < rhs.val < 1E-12:
				rer = 0.0
			else:
				rer = np.power(rhs.err/rhs.val,2)
			return measurement(lhs.val*rhs.val, 
							   lhs.val*rhs.val*np.sqrt(rel + rer))
		except AttributeError: # Multiplication by constant value
			return measurement(lhs.val*rhs, lhs.val*rhs*np.sqrt(rel))
	
	def __truediv__(lhs, rhs): # Change to __div__ for python 2.7 installations
		
		if -1E-12 < lhs.val < 1E-12:  # Errors need to be relative, but if a value 
			rel = 0.0				  # is zero we can't do that.
		elif np.isfinite(lhs.val):
			rel = np.power(lhs.err/lhs.val,2)
		else:
			rel = np.inf
		try: # Division by other measurement
			if -1E-12 < rhs.val < 1E-12:  # Dividing by zero is undefined, so 
				rer = 0.0				  # setting to zero!
				return measurement(0.0,np.inf) # Break!
			elif np.isfinite(rhs.val):
				rer = np.power(rhs.err/rhs.val,2)
			else:
				rer = np.inf
			if np.isfinite(rel) and np.isfinite(rer):
				
				return measurement(lhs.val/rhs.val, 
							   lhs.val/rhs.val*np.sqrt(rel + rer))
			else:
				return measurement(lhs.val/rhs.val,np.inf)
		except AttributeError: # Division by constant value
			if -1E-12 < rhs < 1E-12: # Dividing by zero is undefined!
				return measurement(np.inf,np.inf) # Break!
			else:
				return measurement(lhs.val/rhs, 
								   lhs.val/rhs*np.sqrt(rel))
	def __pow__(lhs, rhs):
		# Error of x = p^y is y*x*(s_p/p) assuming fixed y.
		# A little more work is needed to get unc. in y
		if -1E-12 < lhs.val < 1E-12:
			rel = 0.0
		else: 
			rel = lhs.err/lhs.val
		eV = lhs.val ** rhs.val
		return measurement(eV, rhs.val*eV *rel)

	def exp(self): # Natural exponent
		if np.isfinite(self.val): # Numpy doesn't like if I pass inf.
			return measurement(np.exp(self.val), 
							   np.exp(self.val)*np.abs(self.err))
		elif self.val > 0: # These are the +/- limits of e^(+/- inf)
			return measurement(np.inf,np.inf)
		else:
			return measurement(0.0,0.0)
		
	def log(self): # Natural log
		if self.val <= 0: # Can't take the log of a negative number
			return measurement(self.val,np.inf)
		return measurement(np.log(self.val), self.err/self.val)
		
class reduced_run: # Definition of reduced run information
	# Generate a reduced run with all values zero (except run)
	def __init__(self,run):
		self.run    = run
			
		# "Type of Data" info (Used to generate plots):
		self.thresh  = True # True for High, False for Low
		self.sing  = True
		self.pmt1  = True
		self.pmt2  = True
		
		# Timing info:
		self.hold   = 0.0
		self.mat    = measurement(0.0,0.0)
		
		# Counting info:
		self.ctsSum  = measurement(0.0,0.0)
		self.dtSum   = measurement(0.0,0.0)
		self.bkgSum  = measurement(0.0,0.0)
		self.bkgHSum = measurement(0.0,0.0)
		self.tCSum   = measurement(0.0,0.0)
		
		# Efficiency info
		self.eff    = [measurement(0.5,0.0),measurement(0.5,0.0)]
		self.frac1  = 0.5
		self.frac2  = 0.5
		self.len   = 210.0 # Length of unload period
		
		# Normalization info:
		self.mon    = [0.0,0.0] # 2 non-dagger monitors (at least)
		self.alpha  = 0.0 # Y = a*m1 + b*m2/m1
		self.alphaE = 0.0
		self.beta  = 0.0 
		self.betaE = 0.0
		self.cov   = 0.0
		
		self.norm   = 0.0
		self.pcts   = [0.0,0.0,0.0]
	
		self.cts    = 0.0 # cts is the summed counts buffer
		self.nCts   = 0.0
		
		self.nTime = 20
		
	def __str__(self): # Print as debug
		return "\nRUN: "+str(self.run) + " HOLD: "+str(int(self.hold))
	def __repr__(self):
		return str(self)

	def __sub__(lhs,rhs): # Subtract two reduced runs (for comparing thresholds)
		if lhs.run != rhs.run:
			print("UNABLE TO SUBTRACT, NOT THE SAME RUN")
			return lhs
		elif lhs.sing != rhs.sing:
			print("UNABLE TO SUBTRACT, NEED TO BE THE SAME TYPE")
			return lhs
			
		self = reduced_run(lhs.run)
		self.thresh  = lhs.thresh # True for High, False for Low
		self.sing  = lhs.thresh
		self.pmt1  = lhs.pmt1
		self.pmt2  = lhs.pmt2
		
		# Timing info:
		self.hold   = lhs.hold
		self.mat    = lhs.mat - rhs.mat
		
		# Counting info:
		self.ctsSum  = measurement((lhs.ctsSum - rhs.ctsSum).val,lhs.ctsSum.err)
		self.dtSum   = measurement((lhs.dtSum  - rhs.dtSum).val, lhs.dtSum.err)
		self.bkgSum  = measurement((lhs.bkgSum - rhs.bkgSum).val,lhs.bkgSum.err)
		self.bkgHSum = measurement((lhs.bkgHSum- rhs.bkgHSum).val,lhs.bkgHSum.err)
		self.tCSum   = measurement((lhs.tCSum  - rhs.tCSum).val, lhs.tCSum.err)
		
		# Efficiency info
		self.eff[0]  = lhs.eff[0] - rhs.eff[0] # Might want more here?
		self.eff[1]  = lhs.eff[1] - rhs.eff[1]
		self.frac1  = 0.5
		self.frac2  = 0.5
		
		# Normalization info:
		self.mon    = lhs.mon   # Here we're copying the lhs --- monitors are the smae
		self.alpha  = lhs.alpha - rhs.alpha
		self.alphaE = lhs.alphaE - rhs.alphaE
		self.beta  = lhs.beta - rhs.beta
		self.betaE = lhs.betaE - rhs.betaE
		self.cov   = lhs.cov - rhs.cov
				
		self.norm   = lhs.norm - rhs.norm
		self.pcts = np.zeros(len(lhs.pcts))
		for i in range(len(lhs.pcts)):
			self.pcts[i] = lhs.pcts[i] - rhs.pcts[i]
		
		self.cts    = measurement((lhs.cts - rhs.cts).val,lhs.cts.err) # cts is the summed counts buffer
		self.nCts   = measurement((lhs.normalize_cts() - rhs.normalize_cts()).val,lhs.normalize_cts().err)
		
		return self # As a note, be careful not to re-run a function since it'll probably break things.
		
		
	# Various unload counts
	def total_cts(self,cfg): # Counts
		#return self.ctsSum+self.dtSum-self.bkgSum
		cts = self.ctsSum
		if cfg.useDTCorr:
			cts += self.dtSum
		if cfg.useBkgs:
			cts -= self.bkgSum
		return cts
		
	def eff_cts(self,cfg): # efficiency scaled counts
		return self.total_cts(cfg) / (self.eff[0]+self.eff[1])
	def t_ind_bkg(self): # Background
		return self.bkgSum - self.tCSum
	def pct_cts(self,cfg): # This is the main thing we actually use
		#if (len(cfg.dips) != cfg.ndips):
		cts = measurement(0.0,0.0)
		for d in cfg.dips: # Note that percentage uncertainty shouldn't be double-counted here
				cts += self.total_cts(cfg) * self.pcts[d]
		#else:
		#	cts = self.total_cts(cfg)
		self.cts = cts # Make this callable later
		return cts
	def pct_raw_cts(self,cfg):
		# Only dealing with ctsSum
		#if (len(cfg.dips) != cfg.ndips-1):
		cts = measurement(0.0,0.0)
		for d in cfg.dips:
			cts += self.ctsSum * self.pcts[d]
		#else:
		#	cts = self.ctsSum
		return cts
	def pct_eff_cts(self,cfg): # This is the main thing we actually use
		#if (len(cfg.dips) != cfg.ndips-1):
		cts = measurement(0.0,0.0)
		for d in cfg.dips: # Note that percentage uncertainty shouldn't be double-counted here
			cts += self.eff_cts(cfg) * self.pcts[d]
		#else:
		#	cts = self.eff_cts(cfg)
		self.cts = cts # Make this callable later
		return cts
		
	# Normalization:
	def norm_unl(self,cfg):
		cts  = self.mon[0] # Define counts and spectral correction terms
		#spec = self.mon[1]*self.mon[1]/self.mon[0]
		spec = self.mon[0]*self.mon[0]/self.mon[1]
		unl = cts*self.alpha + spec*self.beta
		
		# Need to incorporate uncertainty in alpha, beta, and cov
		unlE = np.sqrt(unl.val # Poisson
					   + (unl.err*unl.err) # Monitor uncertainty
					   + ((cts *cts ).val * self.alphaE * self.alphaE) # Covariance uncertainty
					   + ((spec*spec).val * self.betaE  * self.betaE )
					   + (2*(cts*spec).val* self.cov))
		
		scale = 0. # Don't double-count percentage uncertainty
		#scale = measurement(0.,0.) # Don't double-count percentage uncertainty
		for d in cfg.normDips:
			scale += self.pcts[d]
		self.norm = measurement(unl.val,unlE)/scale # Make this callable later
		return self.norm
		
	def cts_by_dip(self,cfg): # Number of counts in each dip
		unld = []
		cts = self.total_cts(cfg)
		for p in self.pcts:
			unld.append(cts*p)
		return unld
	
	def normalize_cts(self):
		if self.norm == 0.0: # Require norm value to not be zero
			print("Normalization factor is zero!")
			return measurement(np.inf,np.inf)
					
		return self.cts / self.norm		
	
	def normalize_guess(self,cfg):
		# Guess for normalization is just low monitor
		if self.mon[0] != 0:
			return self.total_cts(cfg) / self.mon[0]
		else:
			return self.total_cts(cfg)
	#def load_pcts(self,p1,p2,p3):
	#	self.pcts   = [p1,p2,p3]
	
	#def get_pct_by_dip(self,dip):
	#	return self.cts*self.pcts[dip]
	
	#def bkg_subtract(self):
	#	self.cts -= self.bkg
	
	#def dt_add(self):
	#	self.cts += self.dt
		
	

#-----------------------------------------------------------------------
# Configuration structs. 
# Most of these are booleans but there's some exceptions
class analyzer_cfg:
	# This is the configuration structure that we use to reduce our runs
	# It's just one object that stores the config file 
	def __init__(self):
		# I'm putting in the defaults here.
		self.vb       = True
		
		# Normalization Parameters
		self.hold     = 20     # What holding time to use
		self.w        = 9999   # How big is our window
		self.bkgWin   = 0
		self.det17    = [3,5]  # Which detectors are we using?
		self.det18    = [8,4]
		self.year     = 2017   # Year to call data from
		self.expoNorm = True   # Deprecated
		self.geomNorm = False  # Deprecated
				
		# Unload Parameters
		self.thresh = True # Threshold, to be passed into reduced_run
		self.pmt1  = True  # Turning on and off PMTs
		self.pmt2  = True
		self.sing  = False # If sing is false, it should go to coinc.
		self.maxUnl = 100.0
		self.ndips  = 3
		self.dips   = [0,1,2]
		self.normDips = [0,1,2]
				
		# Systematics
		self.useMeanArr = True
		self.useDTCorr  = True
		self.useBkgs    = True
		self.usePosBkgs = True
		self.useTimeBkgs= True
		self.useLong    = True
		self.useMoving  = True
		self.scaleSing  = True
	def thresh_scan(self,t):
		if t == 0: 				# Even are low thresholds.
			self.thresh = False # This means that you should always put
		elif t == 1:			# the first unload file as low thresh.
			self.thresh = True
		return t
		
	def pmt_scan(self,p):
		if p == 0:			
			self.pmt1 = True
			self.pmt2 = True
		elif p == 1: # p == 1 for pmt1
			self.pmt1 = True
			self.pmt2 = False
		elif p == 2: # p == 2 for pmt2
			self.pmt1 = False
			self.pmt2 = True
		return p
	def load_format(self,r_red):
		self.thresh = r_red.thresh
		self.sing   = r_red.sing
		self.pmt1   = r_red.pmt1
		self.pmt2   = r_red.pmt2
		return self
			
		
class loading_cfg:
	# This contains the runs we want to load
	def __init__(self):
		self.loadBreaks = False # Deprecated
		
		# Hardcoded run windows
		self.minRun     = 0     
		self.maxRun     = 99999
		
		# Preset run sections
		self.preBlock   = True
		self.alBlock    = True
		self.postBlock  = True
		self.rhc        = True
		self.mid2018    = True
		self.badDag     = True

		# Individual bad runs:
		self.badBackground = False
		self.lightLeaks    = False
		self.badTiming     = False
		self.notProduction = False
		
class output_cfg:
	# This structure contains output info
	def __init__(self):
		self.plotBreaks = True # Do we plot runbreaks?
		
		# Individual plotting 
		self.plotRaw    = True
		self.plotNCts   = True
		self.plotNHists = True
		self.plotBSub   = False
		self.plotNorm   = False
		self.plotPSE    = False
		self.plotSig    = False
		
		# Lifetime plottings
		self.plotLTPair = True
		self.plotLTExp  = True
		
		# Writeout runs
		self.writeAllRuns = True
		self.writeLTPairs = True
		self.writeLongY   = True
