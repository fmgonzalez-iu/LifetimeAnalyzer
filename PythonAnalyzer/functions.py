#!/usr/local/bin/python3

#import threeML
from PythonAnalyzer.classes import measurement
from math import *
#from statsmodels.stats.weightstats import DescrStatsW
#from scipy.optimize import curve_fit, nnls, leastsq, lsq_linear
#from scipy.odr import *
#from itertools import starmap

import numpy as np
import datetime
#import sys
#import pdb
#import csv
#from scipy import stats, special
#from datetime import datetime
#import matplotlib.pyplot as plt
#from multiprocessing import Pool  #for parallelizing
#from numpy.random import normal   #just a normal thing to do in any code
#import emcee 					#MCMC module

#-----------------------------------------------------------------------
#
def map_height(cfg):
	# Hardcoding the dagger heights we can guess.
	hMap = {1:[10.0], \
			3:[380.0, 250.0, 10.0], \
			4:[250.0,380.0,250.0,10.0], \
			9:[380.0, 250.0, 180.0, 140.0, 110.0, 80.0, 60.0, 40.0, 10.0]}
	if ((cfg.ndips == 1) or (cfg.ndips == 3) or (cfg.ndips == 4) or (cfg.ndips == 9)): 
		return hMap[cfg.ndips]
	else:
		sys.exit("ERROR! No height dependence for this number of dips!")


#-----------------------------------------------------------------------
# Functions below this line might not be needed anymore
#-----------------------------------------------------------------------
# Model is an equation which relates the GV counts, standpipe counts, and storage time
# to the dagger counts: 
#
# y = alpha*(m1 + beta*m2) * exp(-dt / tau)
#-----------------------------------------------------------------------
def d_cts(tau,alpha,beta,dt,m1,m2): 
	# Expected yields in the dagger
	# Initial counts should be estimated through alpha*(m1+beta*m2), then through time
	
	return alpha*(m1+beta*m2)*np.exp(-dt/tau)

def mon1_cts(tau,alpha,beta,dt,y,m2):
	# Inverse relationship for mon1 detector in case we need it
	# TODO: Incorporate uncertainty into D,G,S(?)
	
	return y / alpha*np.exp(dt/tau)-beta*m2

def mon2_cts(tau,alpha,beta,dt,y,m1):
	# Inverse relationship for mon2 detector in case
		
	return (y / alpha*np.exp(dt/tau) - m1) / beta

#-----------------------------------------------------------------------
# Likelihoods
#-----------------------------------------------------------------------
def ln_pdf(tau,alpha,beta,dt,y,m1,m1E,m2,m2E):
	#The log of the PDF for each data point. Just a chi2 for now
	
	Y_0  = d_cts(tau,alpha,beta,dt,m1,m2)
	M1_0 = mon1_cts(tau,alpha,beta,dt,y,m2)
	return (y-Y_0)**2 / y # Assuming Gaussian

def ln_multid(x, tau,a,b):
	
	out = []
	for y in x:
		
		t_0, m1, m2 = y
		out.append(a * (m1 + b*m2) * np.exp(-t_0 / tau))
	return out

# The positive logarithm of the likelihood function
# This is maximized by emcee, as a function of any model parameters theta
def lnL(theta,data_arr): 
	# input the data and find the number of breaks:
	breaks = (len(theta) - 1) / 2 # 3 parameters in theta: tau, N, beta. Have 2*breaks n and betas
	
	tau = theta[0] # lifetime, number of init. counts, and "temperature correction" factor beta
	
	if (any([N < 0.0 for n in theta[1:breaks+1]]) \
		or tau < 0.0): # Physical requirement that counts are positive			
		return -np.inf, np.inf
	
	# Sum the log pdf for each data point
	#L_ = sum(map(lambda x : ln_pdf(tau,N,beta,*x[1:]),data_arr)) # Return ln_pdf cast from data as (tau,N,beta, filled_data)
	L_ = sum( \
			starmap(ln_pdf, \
					 [ \
						(tau,n,b,*d[1:]) \
						for (n,b,data) in zip(theta[1:breaks+1],theta[breaks+1:2*breaks+1],data_arr) \
						for d in data \
					 ] \
					) \
			)
	# return the likelihood to be maximize, and the chi2 for reference later
	return -0.5*L_, L_
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# Lifetimes
#-----------------------------------------------------------------------
def lt(sCts, lCts, sDD, lDD): # Lifetime, no mean arr error
	# Lifetime pair measurement calculation -- need short/long counts and short/long times
	tau = measurement(0.0, np.inf)
	if sCts.val <= 0 or lCts.val <= 0:
		return tau
	tau.val = (lDD-sDD)/(log(sCts.val/lCts.val))
	tau.err = sqrt(pow(sCts.err/sCts.val,2)+pow(lCts.err/lCts.val,2))*(lDD-sDD)/pow((log(sCts.val/lCts.val)),2)
	return tau
	
def ltMeas(sCts, lCts, sDD, lDD): # Lifetime, propagate mean arr. error
	# Lifetime pair measurement calculation including uncertainty in MAT
	
	# Error lifetime is -1 with infinite uncertainty
	tau = measurement(-1.0, np.inf)
	
	if (sCts.val <= 0.0) or (lCts.val <= 0.0) or (sDD.val <=0) or (lDD.val <= 0): # Check that we got data into all four values
		return tau
	if (sCts.val <= lCts.val) or (sDD.val >= lDD.val): # Check that short/long counts are actually short/long
		return tau
	
	dt = lDD - sDD # Difference in lifetime
	try:
		sCts = measurement(float(sCts.val),float(sCts.err))
		lCts = measurement(float(lCts.val),float(lCts.err))
	except AttributeError:
		sCts = measurement(float(sCts),0.0)
		lCts = measurement(float(lCts),0.0)
	ctsR = sCts/lCts # Ratio of counts
		
	tau = dt / ctsR.log()
		
	#print tau
	return tau

def ltMeas_corr(sCts, lCts, sDD, lDD): # Lifetime, propagate mean arr. error
	# Lifetime pair measurement calculation including uncertainty in MAT
	
	# Error lifetime is -1 with infinite uncertainty
	tau = measurement(-1.0, np.inf)
	
	if (sCts.val <= 0.0) or (lCts.val <= 0.0) or (sDD.val <=0) or (lDD.val <= 0): # Check that we got data into all four values
		return tau
	if (sCts.val <= lCts.val) or (sDD.val >= lDD.val): # Check that short/long counts are actually short/long
		return tau
	
	dt = lDD - sDD # Difference in lifetime
	try:
		sCts = measurement(float(sCts.val),float(sCts.err))
		lCts = measurement(float(lCts.val),float(lCts.err))
	except AttributeError:
		sCts = measurement(float(sCts),0.0)
		lCts = measurement(float(lCts),0.0)
	ctsR = (sCts/lCts)
	
	corr1 = measurement((sCts/lCts).val * (lCts.err/lCts.val)*(lCts.err/lCts.val),0.)
	#corr3 = (lCts.err/(lCts.val*lCts.val))*((sCts.val/lCts.val)*lCts.err + sCts.err)
	ctsR -=	corr1
	
	#ctsR.err = np.sqrt(( (sCts.err/lCts.err)*(1 - (lCts.err/lCts.val)**2))**2 \
				#+ ((sCts.val*lCts.err)/(lCts.val*lCts.val)*(1 - 3*(lCts.err/lCts.val)**2))**2)
	#print(corr1,  corr3)
	#ctsR = sCts/lCts + (lCts.err/(lCts.val*lCts.val))*((sCts.val/lCts.val)*lCts.err + sCts.err)
	#ctsR = sCts/lCts # Ratio of counts
		
	tau = dt / ctsR.log()# * (1 - (1 - 0.5*ctsR.log().val)*((ctsR.err)/(ctsR.val*ctsR.log().val))**2)
	#corr = 0
	#corr = -0.5 * (dt *(ctsR.log() + 2) / (ctsR.log()*ctsR.log()*ctsR.log()) ).val * (ctsR.err*ctsR.err)/(ctsR.val*ctsR.val)
	
	#tau.err = dt.val/((ctsR.val**3)*((ctsR.log()).val**4))*((ctsR.val**2 - ctsR.err**2)*(ctsR.log().val**2) \
	#			 - 3*ctsR.err*ctsR.err*(1 + ctsR.log().val) )*ctsR.err
	corr = -0.5 * (dt *(ctsR.log() + 2) / (ctsR.log()*ctsR.log()*ctsR.log()) ).val * \
					((sCts.err/sCts.val)*(sCts.err/sCts.val) \
					-(lCts.err/lCts.val)*(lCts.err/lCts.val) \
					- 2*(sCts.err*lCts.err)/(sCts.val*lCts.val*(ctsR.log().val+2)))
					
					
					
	tau = measurement(float(tau.val+corr),float(tau.err))
	
	#print(tau)
	return tau

def explt(x,a,b): # Formatted for curve_fit
	# Calculate lifetime by fitting an exponential
	return a*np.exp(-x/b)
	
def explt_fix(x,a):
	
	func = np.exp(-20/a)
	return np.exp(-x/a) / func
	
	
def lnlt(p,x): # Formatted for ODR
	# Same as above but formatted for ODR
	N, tau = p
	return N*np.exp(-x/tau)
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# Spectral Normalizations
#-----------------------------------------------------------------------	
def spectral_norm(x,a,b):
	
	m1,m2 = x
	
	#val = a*m1+b*m2
	#print val
	
	return np.array(a*m1+b*m2)

def spectral_norm_meas(x,a,b):
	
	m1,m2 = x
	aM = measurement(a,0.0)
	bM = measurement(b,0.0)
	
	#arr = aM*m1+bM*m2
	arrS = []
	wSum = 0	
	for i,m in enumerate(m1):
		#arrS.append(arr.val)
	#	if isinstance(m1[i],measurement):
			#arrS.append(float(m1[i]*aM + m2[i]*aM))
	#		weight = 1.0 / ((m1[i].err/m1[i].val)**2 + (m2[i].err/m2[i].val)**2)
	#	else:
		arrS.append(m1[i]*aM.val+m2[i]*bM.val)
	#		weight = 1.0
		#arrS.append(aM.val*(float(m1[i])+bM.val*float(m2[i])))
		#wSum += weight
		
	#val = a*m1+b*m2
	#print val
	
	return arrS

def spectral_norm_meas_inv(x,a,b):
	#Change this later
	m1,m2 = x
	aM = measurement(a,0.0)
	bM = measurement(b,0.0)
	
	arrS = []
	wSum = 0
		
	for i,m in enumerate(m1):
		# Find "main detector"
		#if m1[i] > m2[i]:
		arrS.append(m1[i]*aM.val+ (m2[i]/m1[i])*bM.val)
		#else:
		#	arrS.append(m2[i]*aM.val+(m1[i]/m2[i])*bM.val)

	return arrS

def spectral_norm_meas_inv2(x,a,b):
	m1,m2 = x
	aM = measurement(a,0.0)
	bM = measurement(b,0.0)
	arrS = []
	wSum = 0
	for i,m in enumerate(m1):
		#arrS.append(m1[i]*aM.val + (m2[i]*m2[i]/m1[i])*bM.val)
		arrS.append(m1[i]*aM.val + (m1[i]*m1[i]/m2[i])*bM.val)
	return arrS

def linear(x,a,b):
	return a + b*x

	
def bkg_cts(bkgVar,S_mu,b):
	# "Expected" background likelihood
	# This assumes that backgrounds can be modeled as a Poisson process (with a relevant scaling factor)
	# This isn't optimized, since it used gammaln
		
	return (b/S_mu)*np.log(bkgVar/S_mu)-(bkgVar/S_mu) - gammaln(b/S_mu+1)#((b/S_mu)*np.log(b/S_mu) - (b/S_mu)) # Poisson background (with stirling)

def gaus_N(x,x0):
	# Gaussian is declared like 50 times so i'm pulling it out
    return 1/np.sqrt(2*np.pi*x0)*np.exp(-(x-x0)**2/(2*x0))

def gaus(x,a,x0,sigma):
	# gaussian function, parameter mu, sigma are the fit parameter
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
	
def quad(x,a,b,c):
    return a + b*(x-c)**2
