#!/usr/local/bin/python3
import numpy as np
from scipy.special import gammaln
#import sys
#import math
import matplotlib.pyplot as plt

#from scipy import stats, special
#from scipy.optimize import curve_fit
#from scipy.special import factorial, gammaln
#from datetime import datetime
from schwimmbad import MPIPool


from PythonAnalyzer.extract import  extract_reduced_runs,\
									extract_reduced_runs_all_mon
#from PythonAnalyzer.plotting import *
from PythonAnalyzer.functions import linear, gaus, gaus_N, quad
from PythonAnalyzer.plotting_monitors import color_run_breaks

totally_sick_blinding_factor = 4.2069

#-----------------------------------------------------------------------
# Need to integrate this into plotting, functions, and other 
# method things
#-----------------------------------------------------------------------

# Here are the functions to use for optimization
def exp_cts(tau,alpha,beta,dt,m1,m2):
	# Expected yields in the dagger
	# Initial counts should be estimated through alpha*m1+beta*m2/m1, then through time
	
	#return (alpha*m1+beta*(m2/m1))*np.exp(-(dt-20.)/tau) # Setting 20s as scaling time
	return (alpha*m1+beta*(m1*m1/m2))*np.exp(-(dt-20.)/tau) # Setting 20s as scaling time

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
	return (b/S_mu)*np.log(bkgVar/S_mu)-(bkgVar/S_mu) - scipy.special.gammaln(b/S_mu+1)#((b/S_mu)*np.log(b/S_mu) - (b/S_mu)) # Poisson background (with stirling)

def poisson_like(l,m):
	# This is the probability of a measurement m from an expected value l
	# numpy factorial gives an overflow error at 171! so I'll use Stirling's approx
	# above that
	m = np.array(m)
	like = m*np.log(l) - l - scipy.special.gammaln(m+1)
	return like	

def unl_cts(U,B,Y,S):
	# Poisson with Stirling's Approx
	
	return (U/S)*np.log(Y+B/S) - (Y+B/S) - ((U/S)*np.log(U/S)-(U/S))
	
def ln_pdf_full(tau,alpha,beta,S_mu,B_mu,
				dt,u,b,s,m1,m1E,m2,m2E):
	
	# This is the probability distribution
	Y_mu = exp_cts(tau,alpha,beta,dt,m1,m2) # Expected UCN in unload
	if Y_mu < 0:
		return -np.inf # Force at least 1 neutron in trap
		
	# For Coincidences, S_mu must be 1.
	unlVar = unl_cts_summed(u,B_mu,Y_mu,S_mu) # Variation in unload counts
	bkgVar = bkg_cts(B_mu,S_mu,b) # Expected background counts
	sVar   = eff_scale(S_mu,u-b, s) # Expected scaling factor -- how many "events" are a UCN
	
	#print(unlVar,bkgVar,sVar)
	#LL = unlVar
	#LL += bkgVar + sVar # Incorporate nuisance factor scaling
	return unlVar+bkgVar+sVar
		
def lnL(param,data_arr):
	# param here is the list of parameters we're tuning.
				
	# The positive logarithm of the likelihood function
	# This is maximized by emcee, as a function of any model parameters theta
	
	# input the data and find the number of breaks:
	breaks = int((len(param) - 1) / 4) # 5 parameters in param: tau, N, beta,S,B_mu. Have 2*breaks n and betas
	tau = param[0] # lifetime, number of init. counts, and "temperature correction" factor beta
		
	# Physical requirement that counts are positive and lifetime is real, and also background/efficiency positive
	if (tau < 0.0  \
		or any([n <= 0.0 for n in param[1:breaks+1]]) \
		or any([s <= 0.0 for s in param[2*breaks+1:3*breaks+1]]) \
		or any([b <= 0.0 for b in param[3*breaks+1:4*breaks+1]])): 
		return -np.inf, np.inf
		
	#print(len(param),np.size(param/len(param)))
	# Sum the log pdf for each data point
	L_ = sum(
		starmap(
			ln_pdf_full,
					 [(tau,n,b,sm,bm,*d[1:])
						for (n,b,sm,bm,data) in zip(param[1:(breaks+1)],param[(breaks+1):(2*breaks+1)],param[(2*breaks+1):(3*breaks+1)],param[(3*breaks+1):(4*breaks+1)],data_arr)
						for d in data
					 ]
					)
			)
	# return the likelihood to be maximized, and the chi2 for reference later
	return -0.5*L_, L_

global data_arr_global
def lnL_global(param): # For faster parallelization, need to define a global data_arr
	# input the data and find the number of breaks:
	breaks = int((len(param) - 1) / 4) # 5 parameters in param: tau, N, beta,S,B_mu. Have 2*breaks n and betas
	tau = param[0] # lifetime, number of init. counts, and "temperature correction" factor beta
		
	# Physical requirement that counts are positive and lifetime is real, 
	# and also background/efficiency positive. 
	# Efficiency is 0.9 so that I can pre-calculate the factorials.
	if (tau < 0.0  \
		or any([n <= 0.0 for n in param[1:breaks+1]]) \
		or any([s <= 0.9 for s in param[2*breaks+1:3*breaks+1]]) \
		or any([b <= 0.0 for b in param[3*breaks+1:4*breaks+1]])):  
		return -np.inf, np.inf
		
	#print(len(param),np.size(param/len(param)))
	# Sum the log pdf for each data point
	L_ = sum(
		starmap(
			ln_pdf_full,
					 [(tau,n,b,sm,bm,*d[1:])
						for (n,b,sm,bm,data) in zip(param[1:(breaks+1)],param[(breaks+1):(2*breaks+1)],param[(2*breaks+1):(3*breaks+1)],param[(3*breaks+1):(4*breaks+1)],data_arr_global)
						for d in data
					 ]
					)
			)
	print(L_)
	# return the likelihood to be maximized, and the chi2 for reference later
	return -0.5*L_, L_




def write_latex_table(r1,r2,name='test.tex'):
	outfile = open(name,'a')
	outfile.write("%d & %d \\\\ \n" % (r1,r2))
	outfile.close()
	return 1
	
def curve_fit_unloads(runNum,tStore,ctsSum,mSum1,mSum2,holdT = 20, runBreaks = []):
	# Use curve_fit on unload counts with two monitor
	
	parameters=np.ones((3,2))
	parList = []
	covList = []
	
	for i, r in enumerate(runBreaks):
		if i==len(runBreaks)-1:
			continue	
		if holdT > 0:
			condition=(tStore==holdT)*(r<=runNum)*(runNum<runBreaks[i+1])
			lbl = ('%d <= run_no < %d' % (r,runBreaks[i+1]))
			ttl = ("%d s" % holdT)
		else:
			condition=(r<=runNum)*(runNum<runBreaks[i+1])
			lbl = ('%d <= run_no < %d' % (r,runBreaks[i+1]))
			ttl = ("all times")
		
		testMon = np.transpose(np.array([mSum1[condition],mSum2[condition]]))
		if len(testMon) == 0: # check that we loaded counts
			parameters = np.array([0.,0.]) # Put in zeros if we didn't
			pov_matrix = np.array([1.,0.,],[0.,1.])
			parList.append(parameters)
			covList.append(pov_matrix)
			continue
		testCts = ctsSum[condition]#/scale[condition] # Prior to this, subtract bkg
		fmt = color_run_breaks(i,runBreaks)		
		guess=[testCts[0]/testMon[0,0],0.]
			
		parameters,pov_matrix = curve_fit(linear_inv_2det,np.float64(testMon),np.float64(testCts),p0=guess,bounds=([0.,-np.inf],[np.inf,np.inf]))
	
		parList.append(parameters)
		covList.append(pov_matrix)
	
	return parList, covList

def linear_2det(x,a,b):
	try:
		x1=np.array(x[:,0]) # Must convert to two numbers
		x2=np.array(x[:,1])
	except IndexError:
		x1=x[0]
		x2=x[1]
	except ValueError:
		print(x)
		sys.exit()
	return a*x1+b*x2

def linear_inv_2det(x,a,b):
	try:
		x1=np.array(x[:,0]) # Must convert to two numbers
		x2=np.array(x[:,1])
	except IndexError:
		x1=x[0]
		x2=x[1]
	except ValueError:
		print(x)
		sys.exit()
	return a*x1+b*x2/x1

def calc_one_norm_sigma(par,cov,mon1):
	# Calculate sigma for normalization, with only one monitor
	sigma2 = (
				(par[0]*par[0]*mon1*mon1)
				*(cov[0,0]/(par[0]*par[0])+(1./mon1)) # Parameter 1
				+(cov[1,1]) # Parameter 2 -- note that the par[1] terms cancel
				+(2*cov[0,1] # And introduce correlation term [since a change in error bars in 1 should reduce error bars in 2 and v.v.]
					*(par[0]*mon1)*np.sqrt(cov[0,0]/(par[0]*par[0])+(1./mon1)) # x1 Sigma x1
					*(par[1])*np.sqrt(cov[1,1]/(par[1]*par[1])) # x2 Sigma x2
				)
			)
	return sigma2
	
def calc_linear_norm_sigma(par,cov,mon1,mon2):
	# Calculate sigma for normalization, for y=a * x1 + b * x2
	sigma2 = (
				par[0]*mon1 # Yield statistics
				+ par[1]*mon2 
				
				#+(mon1*mon1*cov[0,0]) # Covariance
				#+(mon2*mon2*cov[1,1])
				#+2*cov[0,1]*mon1*mon2
				
				#+par[0]*par[0]*mon1 # Monitor statistics
				#+par[1]*par[1]*mon2
				
				#-------------------------------------------------------
				#+(par[1]*par[1]*(mon2*mon2)/(mon1*mon1))*(1./mon1+1./mon2)) # Parameter 2
				#+(par[0]*par[0]*mon1) # Parameter 1
				#mon1*mon1*cov[0,0]
				#+(mon2*mon2)/(mon1*mon1)*cov[1,1]
				#+(2*cov[0,1] # And introduce correlation term [since a change in error bars in 1 should reduce error bars in 2 and v.v.]
					#*(par[0]*mon1)*np.sqrt(cov[0,0]/(par[0]*par[0])+(1./mon1)) # x1 Sigma x1
					#*(par[1]*(mon2/mon1))*np.sqrt(cov[1,1]/(par[1]*par[1])+(1./mon1+1./mon2)) # x2 Sigma x2
					
					
					#*(par[1]*(mon2/mon1))*np.sqrt(cov[1,1]/(par[1]*par[1])+(1./mon2)) # x2 Sigma x2
					#*np.sqrt(cov[0,0]/(par[0]*par[0])+ cov[1,1]/(par[1]*par[1]) +(1./mon1+1./mon2))
				#)
				#-------------------------------------------------------
				#(par[0]*par[0]*mon1*mon1)
				#*(cov[0,0]/(par[0]*par[0]) + (1./mon1)) # Parameter 1
				#*((1./mon1)) # Parameter 1
				#+(par[1]*par[1]*mon2*mon2)
				#*(cov[1,1]/(par[1]*par[1]) + (1./mon2)) # Parameter 2
				#*((1./mon2)) # Parameter 2
				#+(2*cov[0,1] # And introduce correlation term [since a change in error bars in 1 should reduce error bars in 2 and v.v.]
				#	*(par[0]*mon1*np.sqrt(cov[0,0]/(par[0]*par[0]) + 1./mon1)) # x1 Sigma x1
				#	*(par[1]*mon2*np.sqrt(cov[1,1]/(par[1]*par[1]) + 1./mon2)) # x2 Sigma x2
				#)
			)
			
	parM = par.reshape(-1,1)
	parMT = parM.reshape(1,-1)
	#print(parMT)
	#print(cov.dot(parM))
	
	sigma1 = parMT.dot(cov.dot(parM))
	print("SIGMA: ",np.sqrt(sigma1))
	
	#sigma2 = (
	#			(par[0]*par[0]*mon1*mon1)
	#			*(cov[0,0]/(par[0]*par[0])+(1./mon1)) # Parameter 1
	#			+(par[1]*par[1]*mon2*mon2)
	#			*(cov[1,1]/(par[1]*par[1])+(1./mon2)) # Parameter 2
	#			+(2*cov[0,1] # And introduce correlation term [since a change in error bars in 1 should reduce error bars in 2 and v.v.]
	#				*(par[0]*mon1)*np.sqrt(cov[0,0]/(par[0]*par[0])+(1./mon1)) # x1 Sigma x1
	#				*(par[1]*mon2)*np.sqrt(cov[1,1]/(par[1]*par[1])+(1./mon2)) # x2 Sigma x2
	#			)
	#		)
	#print(np.sqrt(par[0]*par[0]*mon1*mon1)*(cov[0,0]/(par[0]*par[0])+(1./mon1)))
	#print(np.sqrt(par[1]*par[1]*mon2*mon2)*(cov[1,1]/(par[1]*par[1])+(1./mon2)))
	#print((2*cov[0,1]*(par[0]*mon1)*np.sqrt(cov[0,0]/(par[0]*par[0])+(1./mon1))*(par[1]*mon2)*np.sqrt(cov[1,1]/(par[1]*par[1])+(1./mon2))))
	return sigma2
	
def calc_inv_norm_sigma(par, cov, mon1,mon2):
	# Calculate sigma for normalization, for y=a * x1 + b * (x2/x1)
	
	#sigma2 = (
	#			par[0]*mon1 # Yield statistics
	#			+ par[1]*mon2/mon1	
	#			+mon1*mon1*cov[0,0] # Covariance 
	#			+(mon2*mon2)/(mon1*mon1)*cov[1,1]
	#			+2*mon2*cov[0,1]
	#			+(par[0]*par[0]*mon1 # Monitor uncertainty
	#			+par[1]*par[1]*(mon2/mon1)*(1. / mon1 + 1. / mon2))
	#		)
			
	sigma2 = (
				(par[0]*par[0]*mon1*mon1)
				*(cov[0,0]/(par[0]*par[0])+(1./mon1)) # Parameter 1
				+(par[1]*par[1]*(mon2*mon2)/(mon1*mon1))
				*(cov[1,1]/(par[1]*par[1])+(1./mon1+1./mon2)) # Parameter 2
				#*(cov[1,1]/(par[1]*par[1])+(1./mon2)) # Parameter 2
				+(2*cov[0,1] # And introduce correlation term [since a change in error bars in 1 should reduce error bars in 2 and v.v.]
					*(par[0]*mon1)*np.sqrt(cov[0,0]/(par[0]*par[0])+(1./mon1)) # x1 Sigma x1
					#*(mon1)#*np.sqrt((1./mon1)) # x1 Sigma x1
					*(par[1]*(mon2/mon1))*np.sqrt(cov[1,1]/(par[1]*par[1])+(1./mon1+1./mon2)) # x2 Sigma x2
					#*((mon2/mon1)*(mon2/mon1))*np.sqrt((1./mon1+1./mon2)) # x2 Sigma x2
					#*(par[1]*(mon2/mon1))*np.sqrt(cov[1,1]/(par[1]*par[1])+(1./mon2)) # x2 Sigma x2
					#*np.sqrt(cov[0,0]/(par[0]*par[0])+ cov[1,1]/(par[1]*par[1]) +(1./mon1+1./mon2))
				)
			)
	
	parM = par.reshape(-1,1)
	parMT = parM.reshape(1,-1)
	
	sigma1 = parMT.dot(cov.dot(parM))
	print("SIGMA: ",sigma1)
	return sigma2

def calc_correlation_coeff(arr1,arr2):
	# The Pearson correlation coefficient is a measure of correlations of
	# two arrays.
	arr1 = np.array(arr1) # If not already arrays.
	arr2 = np.array(arr2)
	
	muX = np.mean(arr1) # Calculate means
	muY = np.mean(arr2)
	
	r = np.sum((arr1 - muX)*(arr2 - muY)) \
		/ np.sqrt(np.sum((arr1 - muX)*(arr1 - muX))*np.sum((arr2 - muY)*(arr2 - muY)))
	return r # Should go from -1 to 1

class plotting_config:
	def __init__(self):
		
		self.plotPoisson     = True
		self.plotRawMonitors = True
		self.plotYield       = True
		self.plotMonVsCts    = True
		self.plotMonComps    = True
		self.plotNormalizedYields = True
		self.plotVsExp    = True

# I fucked up by making the plotting_object and wasting time.		
class plotting_object:
	# I'm making an object-oriented form of plotting.
	# Maybe this'll make things easier for the next analyzer...
	
	def __init__ (self,rRedL,cfg = []):
		
		self.plot_fig = 1
		# I use a reducedRun object for actual analysis.
		# For plotting some things I did it with numpy vectors. Thus,
		# I'm going to write this object with numpy arrays.
		#
		# Start by initializing some numpy arrays.
		# Declare important plotting things
		self.run_no  = np.zeros(len(rRedL),dtype='i4') # run
		self.hold    = np.zeros(len(rRedL),dtype='i4') # hold
		self.fore    = np.zeros(len(rRedL),dtype='f8') # cts + dt
		self.back    = np.zeros(len(rRedL),dtype='f8') # bkg
		self.bTDep   = np.zeros(len(rRedL),dtype='f8') 
		self.mat     = np.zeros(len(rRedL),dtype='f8')
		self.eff     = np.zeros(len(rRedL),dtype='f8')
		self.gv_sum  = np.zeros(len(rRedL),dtype='f8')
		self.rh_sum  = np.zeros(len(rRedL),dtype='f8')
		self.sp_sum  = np.zeros(len(rRedL),dtype='f8')
		self.rhac_sum= np.zeros(len(rRedL),dtype='f8')
		self.ds_sum  = np.zeros(len(rRedL),dtype='f8')
		
		# If there's no config object loaded, just set all to 0
		try:
			cfg.useDTCorr # Should have a safer method than this...
		except:
			cfg = analyzer_cfg()
			
		if len(rRedL[0].mon) == 2:
			self.just_two_monitors = True
		else:
			self.just_two_monitors = False
		self.scale_to_GV = False
		# Loop through reduced runs and load these into vector/array form.
		for i,r in enumerate(rRedL):
			self.run_no[i]  = r.run
			self.hold[i] = int(round(r.hold))       # Hold, nearest int
			self.fore[i]    = (r.ctsSum).val
			
			if cfg.useDTCorr:
				self.fore[i] += r.dtSum.val
			self.back[i]    = (r.bkgSum - r.tCSum).val # Time Independent Component
			self.bTDep[i]   = (r.tCSum).val            # Time Dependent Component
			if cfg.useMeanArr:
				self.mat[i] = r.mat.val
			else:
				self.mat[i] = r.hold
			if r.sing:
				self.eff[i] = (r.eff[0]+r.eff[1]).val
			else:
				self.eff[i] =  1.
			if self.just_two_monitors: # Assume GV and SP if we pulled from extract_reduced_runs
				self.gv_sum[i] = r.mon[0].val
				self.sp_sum[i] = r.mon[1].val
			else:
				self.gv_sum[i]   = r.mon[2].val
				self.rhac_sum[i] = r.mon[3].val # 2017 bare
				self.sp_sum[i]   = r.mon[4].val
				self.ds_sum[i]   = r.mon[6].val
				self.rh_sum[i]   = r.mon[7].val # 2017 foil
			
			# Holding time trickery
			self.hold_fmt = [('hold','i4'),('fmt','U4'),('label','U16')]
			self.hold_list = np.zeros(6,dtype=self.hold_fmt)
			
			self.hold_list[0] = (20,   'r.','20s Hold')
			self.hold_list[1] = (50,   'y.','50s Hold')
			self.hold_list[2] = (100,  'g.','100s Hold')
			self.hold_list[3] = (200,  'b.','200s Hold')
			self.hold_list[4] = (1550, 'c.','1550s Hold')
			self.hold_list[5] = (0    ,'k.','Other Hold')
	
	def poisson_hist_comparison(self,run_breaks = [],req_yield='Background',hold = 20):
		# For checking whether or not a given monitor is a Poisson distribution
		# Spoiler alert: They're not.
		plt.figure(self.plot_fig)
		self.plot_fig +=1 
		
		# This thing will break super hardcore if you do a bunch of runbreaks, but w/e
		if len(run_breaks) < 2:
			if min(run_breaks) > np.min(self.run_no):
				run_breaks.append(np.min(self.run_no))
			if max(run_breaks) < np.max(self.run_no):
				run_breaks.append(np.max(self.run_no))
			run_breaks.sort()
		
		# Figure out what yields we intend to plot, do some separations.
		if req_yield == 'Background':
			yld = self.back
			name = "Background Counts"
			combine_holds = True
		elif req_yield == 'Foreground':
			yld = self.fore
			name = "Foreground Counts"
			combine_holds = False
		else: 
			return 
		
		if not combine_holds:
			condT = (self.hold == hold)
		else:
			condT = np.ones(len(self.hold),dtype=bool)
		
		# Find max, min, mean for plotting
		sMu  = np.mean(yld[condT])
		#sMin = int(np.min(yld[condT]) - (sMu/4)) # No poisson < 0
		sMin = int(np.min(yld[condT]))
		if sMin < 0:
			sMin = 0
		#sMax = int(np.max(yld[condT]) + (sMu/4))
		sMax = int(np.max(yld[condT]))
		
		if sMax - sMin < 500: # Roughly singles vs coincidence counts
			nBins = int(((sMax - sMin) + 1)/2) # 2 events/bin
		else:
			nBins = int((sMax - sMin)/30 + 1) # Assume 30 PE/UCN
		
		if nBins > 50:
			nBins = 50
		
		bins=np.linspace(sMin,sMax,nBins)
		# Now let's generate some histogram bins
		parList = np.empty(len(run_breaks)-1) # Outputting parameters
		covList = np.empty(len(run_breaks)-1) # And Covariance Matrices
		for i in range(0,len(run_breaks)-1):
			
			color = color_run_breaks(i,run_breaks) # color for plotting
			cond = (run_breaks[i] <= self.run_no)*(self.run_no < run_breaks[i+1])
			if not combine_holds:
				cond *= (self.hold == hold)	
			parse_yield = yld[cond]
			guess=np.mean(parse_yield)

			lbl = ('%d <= run < %d: ' % (run_breaks[i],run_breaks[i+1]))
			
			nRuns = len(parse_yield)
			if nRuns == 0:
				continue
			print("guess= ",guess)
			print("sMin,sMax,nBins",sMin,sMax,nBins)		
			# Now make a histogram
			if sMin >= 0 and sMax > 0:
				entries, bin_edges, patches = plt.hist(parse_yield,bins,density=True,ec=color,fc='none')
			else:
				entries, bin_edges, patches = plt.hist(parse_yield,bins=nBins,density=True,ec=color,fc='none')
			bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1]) # For chi2 calc
			
			# Fit the data
			entries = np.array(entries)
			from scipy.optimize import curve_fit # Import optizimation locally
			from scipy.special import factorial,gammaln
			def poisson(mu,k): # Define a poisson distribution. 
				logP = k*np.log(mu) - mu - gammaln(k+1) 
				return np.exp(logP)#(mu**k/factorial(k))*np.exp(-mu)
				#return nRuns*np.exp(logP)#(mu**k/factorial(k))*np.exp(-mu)
		
			try:
				parameters, cov_matrix = curve_fit(poisson, bin_middles, entries, p0=[guess]) 
			
				print(parameters)
				plt.plot(bin_middles, poisson(bin_middles, *parameters), 'k-', lw=5)
				plt.plot(bin_middles, poisson(bin_middles, *parameters), color=color,ls='--',lw=3)
				#plt.plot(bin_middles, poisson(bin_middles, guess), 'k-', lw=5)
				#plt.plot(bin_middles, poisson(bin_middles, guess), color=color,ls='--',lw=3)
			
				#chisq = np.sum((entries-poisson(bin_middles, *parameters))**2/(poisson(bin_middles,*parameters)))
				chisq = np.sum((entries-poisson(bin_middles, *parameters))**2/(entries))
				#chisq = np.sum((entries-poisson(bin_middles, guess))**2/(poisson(bin_middles,guess)))
				dof = float(len(bin_middles)-1)
				if 0.01 < chisq/dof < 999:
					plt.plot([],[],color=color,label=(r'$ %d \leq $run$ < %d; \chi^2 / $NDF$ = %0.2f$' %  (run_breaks[i],run_breaks[i+1],chisq/dof)))
				else:
					plt.plot([],[],color=color,label=(r'$ %d \leq $run$ < %d; \chi^2 / $NDF$ = %0.2e$' %  (run_breaks[i],run_breaks[i+1],chisq/dof)))
				plt.title(name)
				plt.ylabel("Density")
				plt.xlabel("Counts")
				#plt.yscale('log')
				print("Stats for %d <= run_no < %d" % (run_breaks[i], run_breaks[i+1]))
				print('no. runs = ', nRuns)
				print("mu = ", parameters, cov_matrix)
				print("delta_mu = ", parameters*np.sqrt(cov_matrix))
				print("1/sqrtN = ", 1/np.sqrt(nRuns))
				
				print('chisq = ',chisq,'; dof =', dof, '; chisq/dof =', chisq/dof,"\n")
			except RuntimeError: # Can't find curve_fit result!
				plt.plot(bin_middles, poisson(bin_middles, guess), 'k-', lw=5)
				plt.plot(bin_middles, poisson(bin_middles, guess), color=color,ls='--',lw=3)
			
				#chisq = np.sum((entries-poisson(bin_middles, sMu))**2/(poisson(bin_middles,sMu)))
				chisq = np.sum((entries-poisson(bin_middles, guess))**2/(poisson(bin_middles, guess)))
				chisq = np.sum((entries-poisson(bin_middles, guess))**2/(entries))
				dof = float(len(bin_middles)-1)
			
				if 0.01 < chisq/dof < 999:
					plt.plot([],[],color=color,label=(r'$ %d \leq $run$ < %d; \chi^2 / $NDF$ = %0.2f$' %  (run_breaks[i],run_breaks[i+1],chisq/dof)))
				else:
					plt.plot([],[],color=color,label=(r'$ %d \leq $run$ < %d; \chi^2 / $NDF$ = %0.2e$' %  (run_breaks[i],run_breaks[i+1],chisq/dof)))
				
				plt.title(name)
				plt.ylabel("Density")
				plt.xlabel("Counts")
				#plt.yscale('log')
				parameters = guess
				cov_matrix = np.inf
				print("Stats for %d <= run_no < %d" % (run_breaks[i], run_breaks[i+1]))
				print('no. runs = ', nRuns)
				print("mu = ", parameters, cov_matrix)
				print("delta_mu = ", parameters*np.sqrt(cov_matrix))
				print("1/sqrtN = ", 1/np.sqrt(nRuns))
				
				print('chisq = ',chisq,'; dof =', dof, '; chisq/dof =', chisq/dof,"\n")
				
			plt.legend(loc='upper right')
			parList[i] = parameters
			covList[i] = cov_matrix
					
		return parList, covList
							
	def monitor_by_run(self):
		# This is plotting the monitor counts on a run-by-run basis	
		plt.figure(self.plot_fig)
		self.plot_fig += 1
		if not self.scale_to_GV:
			if self.just_two_monitors: # Low and High
				plt.plot(self.run_no,self.gv_sum,marker='+',color='b',linestyle='solid',  label="Low Monitor")
				plt.plot(self.run_no,self.sp_sum,marker='.',color='r',linestyle='dashdot',label="High Monitor")
				plt.plot(self.run_no,self.fore,  marker='<',color='k',linestyle='dashed', label="Dagger Unloads")
			else: # All monitors
				#plt.plot(self.run_no,self.gv_sum,  marker='+',color='b',linestyle='solid',          label="GV")
				#plt.plot(self.run_no,self.sp_sum,  marker='*',color='y',linestyle='dotted',         label="SP")
				#plt.plot(self.run_no,self.rh_sum,  marker='<',color='c',linestyle=(0,(5,1)),        label="Foil/RH")
				#plt.plot(self.run_no,self.rhac_sum,marker='.',color='r',linestyle='dashdot',        label="Bare/RHAC")
				#plt.plot(self.run_no,self.ds_sum,  marker='x',color='g',linestyle=(0,(3,5,1,5,1,5)),label="Downstream")
				plt.plot(self.run_no,self.gv_sum,  color='b',linestyle='solid',          label="GV")
				plt.plot(self.run_no,self.sp_sum,  color='y',linestyle='dotted',         label="SP")
				plt.plot(self.run_no,self.rh_sum,  color='c',linestyle=(0,(5,1)),        label="Foil/RH")			
				plt.plot(self.run_no,self.rhac_sum,color='r',linestyle='dashdot',        label="Bare/RHAC")
				plt.plot(self.run_no,self.ds_sum,  color='g',linestyle=(0,(3,5,1,5,1,5)),label="Downstream")
			plt.title("Monitor Detector Summed Counts")
			plt.xlabel("Run Number")
			plt.ylabel("Counts")
		else:
			if self.just_two_monitors: # Low and High
				#plt.plot(self.run_no,self.gv_sum/self.gv_sum,color='b',linestyle='solid', label="Low Monitor")
				plt.plot(self.run_no,self.sp_sum/self.gv_sum,color='r',linestyle='dashdot',label="High Monitor")
				plt.plot(self.run_no,self.fore/self.gv_sum,  color='k',linestyle='dashed', label="Dagger Unloads")
			else: # All monitors
				#plt.plot(self.run_no,self.gv_sum/self.gv_sum,  marker='+',color='b',linestyle='solid',          label="GV")
				plt.plot(self.run_no,self.sp_sum/self.gv_sum,  marker='*',color='y',linestyle='dotted',         label="SP")
				plt.plot(self.run_no,self.rh_sum/self.gv_sum,  marker='<',color='c',linestyle=(0,(5,1)),        label="Foil/RH")
				plt.plot(self.run_no,self.rhac_sum/self.gv_sum,marker='.',color='r',linestyle='dashdot',     label="Bare/RHAC")
				plt.plot(self.run_no,self.ds_sum/self.gv_sum,  marker='x',color='g',linestyle=(0,(3,5,1,5,1,5)),label="Downstream")
			plt.title("Monitor Detector Counts, Scaled to GV")
			plt.xlabel("Run Number")
			plt.ylabel("Counts Ratio (arb.)")
		plt.legend(loc='upper right')
		plt.yscale('log')
		plt.grid(True)
		#plt.xaxis(scilimits=(0,0))
		#plt.yaxis(scilimits=(0,0))
		
		return self.plot_fig
		
	def raw_yields(self):
		# The reducedRun version of this works too and is more flexible.
		plt.figure(self.plot_fig) # Raw Yield
		self.plot_fig += 1
		for t in self.hold_list:
			if t['hold'] > 0: # Find the condition  for holding times
				cond = (t['hold'] - 1 < self.hold)*(self.hold < t['hold'] + 1)
			else: # Do "Other holds" last...
				cond = np.ones(len(self.run_no),dtype=int)
				for ti in range(len(self.hold_list)-1): 
					cond -= (self.hold_list[ti]['hold'] - 1 < self.hold)*(self.hold < self.hold_list[ti]['hold'] + 1)
			plt.plot(self.run_no[cond],self.fore[cond],color=t['fmt'],label=t['label'])
		plt.title("Raw Counts")
		plt.xlabel("Run Number")
		plt.ylabel("Counts")
		plt.legend(loc='upper right')
		
		return self.plot_fig
	
	def check_monitor(self,mon):
		# Check which monitor we're using
		if mon=='GV':
			monOut = self.gv_sum
		elif mon=='RH':
			monOut = self.rh_sum
		elif mon=='RHAC':
			monOut = self.rhac_sum
		elif mon=='SP':
			monOut = self.sp_sum
		elif mon=='DS':
			monOut = self.ds_sum
		else:
			monOut = self.fore
	
		return monOut
	
	def counts_scatter(self,mon1='GV',mon2 = 'Dagger', run_breaks=[],hold = 20):
		# This plots a scatter between two different monitors
		# or between monitors and unload counts
		
		if min(run_breaks) > np.min(self.run_no):
			run_breaks.append(np.min(self.run_no))
		if max(run_breaks) < np.max(self.run_no):
			run_breaks.append(np.max(self.run_no))
		run_breaks.sort()
		
		if mon1 == 'Dagger' or mon2 == 'Dagger':
			# For dagger, need to compare to just one hold.
			one_hold = True
		else:
			# For monitors it doesn't matter.
			one_hold = False
		
		parList = np.empty(len(run_breaks)-1) # Outputting parameters
		covList = np.empty(len(run_breaks)-1) # And Covariance Matrices
		chiList = np.empty(len(run_breaks)-1) # Also the goodness of fit of each run
		for i in range(len(run_breaks)-1):
			
			# Conditions!
			cond = (run_breaks[i] <= self.run_no)*(self.run_no < run_breaks[i+1])
			if one_hold:
				cond *= (self.hold==hold)
			
			# Need to extract the two items we're plotting against
			# First monitor 1:
			mon1Cts = self.check_monitor(mon1)[cond]
			mon2Cts = self.check_monitor(mon2)[cond]
			
			if len(mon1Cts) == 0:
				continue
			
			color = color_run_breaks(i,run_breaks)
			plt.plot(mon1Cts,mon2Cts,'.',color=color)
			from PythonAnalyzer.functions import linear
			from scipy.optimize import curve_fit
			
			guess = [np.mean(mon2Cts)/np.mean(mon1Cts),0.] # Assume no background, means are the same
			params, cov = curve_fit(linear, mon1Cts, mon2Cts, p0=[guess])
			chi = (mon2Cts-linear(mon1Cts,*params))**2/linear(mon1Cts,*params)
			chisq = np.sum(chi)
			dof = len(mon1Cts)-2
			
			bins = np.linspace(np.min(mon1Cts),np.max(mon1Cts),100)
			plt.plot(bins,linear(bins, *params),'-',color='k',lw=3)
			plt.plot(bins,linear(bins, *params),'--',color=color,lw=2)
			plt.plot([],[],'.',color=color,label=(r'$%d <= $run$ < %d; \chi^2/$NDF$ = %0.2f$' % (run_breaks[i],run_breaks[i+1],chisq/dof)))
			
			print("Stats for %d <= run_no < %d" % (run_breaks[i], run_breaks[i+1]))
			print('chisq = ',chisq,'; dof =', dof, '; chisq/dof =', chisq/dof,"\n")
			print('linear = ',params[0],'; +=',params[1])
			print('delta = ',np.sqrt(cov[0,0]),'; +=',np.sqrt(cov[1,1]))
			
			
			#plt.xlabel(mon1+' Counts')
			#plt.ylabel(mon2+' Counts')
			#plt.legend(loc='upper left')
		
			#parList[i] = np.array(params)
			#covList[i] = np.array(cov)
			#chiList[i] = chi
		
		return parList,covList,chiList
	def counts_correlation(self,mon1='GV',mon2 = 'Dagger', run_breaks=[],hold = 20):
		# This plots Correlation Coefficients between two monitors.
		
		if min(run_breaks) > np.min(self.run_no):
			run_breaks.append(np.min(self.run_no))
		if max(run_breaks) < np.max(self.run_no):
			run_breaks.append(np.max(self.run_no))
		run_breaks.sort()
		
		if mon1 == 'Dagger' or mon2 == 'Dagger':
			# For dagger, need to compare to just one hold.
			one_hold = True
		else:
			# For monitors it doesn't matter.
			one_hold = False
		
		parList = np.empty(len(run_breaks)-1) # Outputting parameters
		covList = np.empty(len(run_breaks)-1) # And Covariance Matrices
		chiList = np.empty(len(run_breaks)-1) # Also the goodness of fit of each run
		for i in range(len(run_breaks)-1):
			
			# Conditions!
			cond = (run_breaks[i] <= self.run_no)*(self.run_no < run_breaks[i+1])
			if one_hold:
				cond *= (self.hold==hold)
			
			# Need to extract the two items we're plotting against
			# First monitor 1:
			mon1Cts = self.check_monitor(mon1)[cond]
			mon2Cts = self.check_monitor(mon2)[cond]
			
			if len(mon1Cts) == 0:
				continue
			
			color = color_run_breaks(i,run_breaks)
			plt.plot(mon1Cts,mon2Cts,'.',color=color)
			from PythonAnalyzer.functions import linear
			from scipy.optimize import curve_fit
			
			guess = [np.mean(mon2Cts)/np.mean(mon1Cts),0.] # Assume no background, means are the same
			params, cov = curve_fit(linear, mon1Cts, mon2Cts, p0=[guess])
			chi = (mon2Cts-linear(mon1Cts,*params))**2/linear(mon1Cts,*params)
			chisq = np.sum(chi)
			dof = len(mon1Cts)-2
			
			bins = np.linspace(np.min(mon1Cts),np.max(mon1Cts),100)
			plt.plot(bins,linear(bins, *params),'-',color='k',lw=3)
			plt.plot(bins,linear(bins, *params),'--',color=color,lw=2)
			plt.plot([],[],'.',color=color,label=(r'$%d <= $run$ < %d; \chi^2/$NDF$ = %0.2f$' % (run_breaks[i],run_breaks[i+1],chisq/dof)))
			
			print("Stats for %d <= run_no < %d" % (run_breaks[i], run_breaks[i+1]))
			print('chisq = ',chisq,'; dof =', dof, '; chisq/dof =', chisq/dof,"\n")
			print('linear = ',params[0],'; +=',params[1])
			print('delta = ',np.sqrt(cov[0,0]),'; +=',np.sqrt(cov[1,1]))
			
			
			#plt.xlabel(mon1+' Counts')
			#plt.ylabel(mon2+' Counts')
			#plt.legend(loc='upper left')
		
			#parList[i] = np.array(params)
			#covList[i] = np.array(cov)
			#chiList[i] = chi
		
		return parList,covList,chiList
	def monitor_corner(self,run_breaks=[],hold=0):
	
		if min(run_breaks) > np.min(self.run_no):
			run_breaks.append(np.min(self.run_no))
		if max(run_breaks) < np.max(self.run_no):
			run_breaks.append(np.max(self.run_no))
		run_breaks.sort()
				
		plt.figure(self.plot_fig)
		self.plot_fig += 1
		
		mons=['GV','SP','RH','RHAC','DS']
		# Loop through to generate monitor cornering counts
		mx = len(mons)
		ctr = 1
		for m1 in range(mx):
			for m2 in range(mx):
				plt.subplot(mx,mx,ctr)
				self.counts_scatter(mons[m1],mons[m2],run_breaks)
				ctr += 1
		plt.show()
			
def monitor_plotting_likelihoods(runList,cts,nMon,bkgsH,bkgsT,cfg,runBreaks = []):
	# Making a new plotting thingy here
	from PythonAnalyzer.plotting_monitors import histogram_poisson_function,\
												 plot_monitor_by_run,\
												 make_errorbar_by_hold,\
												 plot_counts_vs_mon,\
												 plot_run_vs_exp

	plotter = plotting_config()
		
	if len(runBreaks) == 0:
		runBreaks=[4200,9600,14520] # Just years.
	
	# Extract reduced runs from our counts lists.	
	reducedRunL = extract_reduced_runs_all_mon(runList, cts, nMon, bkgsH, bkgsT, cfg)
	
	# Here we convert reducedRunL to numpy arrays since I'm parsing 2 different coding regimes
	# Declare important plotting things
	run_no  = np.zeros(len(reducedRunL),dtype='i4') # run
	t_store = np.zeros(len(reducedRunL),dtype='i4') # hold
	fore    = np.zeros(len(reducedRunL),dtype='f8') # cts + dt
	back    = np.zeros(len(reducedRunL),dtype='f8') # bkg
	bTDep   = np.zeros(len(reducedRunL),dtype='f8') 
	mat     = np.zeros(len(reducedRunL),dtype='f8')
	eff     = np.zeros(len(reducedRunL),dtype='f8')
	gv_sum  = np.zeros(len(reducedRunL),dtype='f8')
	rh_sum  = np.zeros(len(reducedRunL),dtype='f8')
	sp_sum  = np.zeros(len(reducedRunL),dtype='f8')
	rhac_sum= np.zeros(len(reducedRunL),dtype='f8')
	ds_sum  = np.zeros(len(reducedRunL),dtype='f8')
	# Loop through reduced runs and load these into vector/array form.
	for i,r in enumerate(reducedRunL):
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
		gv_sum[i]   = r.mon[2].val
		rhac_sum[i] = r.mon[3].val # 2017 bare
		sp_sum[i]   = r.mon[4].val
		ds_sum[i]   = r.mon[6].val
		rh_sum[i]   = r.mon[7].val # 2017 foil
	
	run_breaks = []
	for i in range(0,len(runBreaks)-1): # reduce the number of runBreaks
		if runBreaks[i]<=run_no[0] < runBreaks[i+1]: 
			run_breaks.append(runBreaks[i]) # find first
			for j in range(i+1,len(runBreaks)):
				if runBreaks[j] <= run_no[len(run_no)-1]:
					run_breaks.append(runBreaks[j]) # append through the last break
				else:
					break
			break
	if run_breaks[-1] < max(run_no): # And possibly add the last run.
		run_breaks.append(max(run_no))
	print("Using Run Breaks:", run_breaks)	
	
	pltFig = 0
	
	# Now we need to do normalization spreads to get things working.			
	# Plot the raw monitor counts
	pltObj = plotting_object(reducedRunL,cfg)
	plotter.plotPoisson=False
	plotter.plotRawMonitors=True
	plotter.plotYield=False
	plotter.plotMonVsCts = False
	plotter.plotVsExp = True
	# First plot histograms between foreground/background and just Poisson.
	if plotter.plotPoisson:
		pltObj.poisson_hist_comparison(run_breaks,'Background')
		pltObj.poisson_hist_comparison(run_breaks,'Foreground')
		plt.show()
	# Next, plot the raw monitor counts
	if plotter.plotRawMonitors:
		pltObj.monitor_by_run()
		pltObj.scale_to_GV = True
		pltObj.monitor_by_run()
		pltObj.scale_to_GV = False
		plt.show()	
	# And this is the raw yields (though the other plotter is probably better.
	if plotter.plotYield:
		pltObj.raw_yields()
		plt.show()
	# Plot the comparison between Monitors and Unloads
	if plotter.plotMonVsCts:
		pltObj.counts_scatter('GV','Dagger',   run_breaks,20)
		pltObj.counts_scatter('SP','Dagger',   run_breaks,20)
		pltObj.counts_scatter('RH','Dagger',   run_breaks,20)
		pltObj.counts_scatter('RHAC','Dagger', run_breaks,20)
		pltObj.counts_scatter('DS','Dagger',   run_breaks,20)
		plt.show()
	# Plot correlations between monitors
	plotter.plotMonComps = False
	if plotter.plotMonComps:
		pltObj.monitor_corner(run_breaks) # TODO: Make work
		plt.show()
	plotter.plotNormalizedYields = False # TODO: Make work
	if plotter.plotNormalizedYields: 
		# TODO: This compared background normalized to other things?
		# Possibly do a single-monitor unload (?) 
		
		# plt.figure(pltFig) # "Normalized yield
		# pltFig+=1
		# for i, t in enumerate(t_store):
			# make_errorbar_by_hold(t,run_no[i],(fore[i]-back[i])/GV_sum[i])
		# plt.title("Normalized to GV")
		# plt.xlabel("Run Number")
		# plt.ylabel("Normalized Counts (arb.)")
				
		# plt.figure(pltFig)
		# pltFig+=1
		# for i, t in enumerate(t_store):
			# make_errorbar_by_hold(t,run_no[i],(fore[i]-back[i])/SP_ave[i])
		# plt.title("Normalized to SP")
		# plt.xlabel("Run Number")
		# plt.ylabel("Normalized Counts (arb.)")
				
		# plt.figure(pltFig) # "Normalized yield
		# pltFig+=1
		# for i, t in enumerate(t_store):
			# make_errorbar_by_hold(t,run_no[i],(fore[i]-back[i])/RH_sum[i])
		# plt.title("Normalized to RH")
		# plt.xlabel("Run Number")
		# plt.ylabel("Normalized Counts (arb.)")
				
		# plt.figure(pltFig)
		# pltFig+=1
		# for i,t in enumerate(t_store):
			# make_errorbar_by_hold(t,run_no[i],(fore[i]-back[i])/RHAC_sum[i])
		# plt.title("Normalized to RHAC")	
		# plt.xlabel("Run Number")
		# plt.ylabel("Normalized Counts (arb.)")	
		
		# plt.figure(pltFig)
		# pltFig+=1
		# for i, t in enumerate(t_store):
			# make_errorbar_by_hold(t,run_no[i],fore[i]/back[i])
		# plt.title("Signal to Background")
		# plt.xlabel("Run Number")
		# plt.ylabel("Normalized Counts (arb.)")
				
		# plt.figure(pltFig)
		# pltFig+=1
		# for i,t in enumerate(t_store):
			# make_errorbar_by_hold(t,run_no[i],(RH_sum[i]/GV_sum[i]))
		# plt.title("RH/GV")	
		# plt.xlabel("Run Number")
		# plt.ylabel("Normalized Monitor Ratio (arb.)")
		
		# plt.figure(pltFig)
		# pltFig+=1
		# for i,t in enumerate(t_store):
			# make_errorbar_by_hold(t,run_no[i],(SP_ave[i]/GV_sum[i]))
		# plt.title("SP/GV")	
		# plt.xlabel("Run Number")
		# plt.ylabel("Normalized Monitor Ratio (arb.)")
		
		
		#plt.figure(pltFig)
		#pltFig +=1
		#for i,t in enumerate(t_store):
		#	make_errorbar_by_hold(t,run_no[i],back[i])
		#plt.title("Background (singles)")	
		#plt.xlabel("Run Number")
		#plt.ylabel("Normalized Monitor Ratio (arb.)")
			
		#plt.figure(pltFig)
		#pltFig += 1
		#rB_temp = [4230,9768,11217,14503]
		#rB_temp = [9767,9960,10936,10988,11085,11669,12516,14509]
		#rB_temp = [9768,11217,14503]
		#_f,_f = histogram_poisson_function(run_no,GV_sum/RH_sum,t_store,rB_temp, 'GV/DS', 0,20)
		
		#pltFig+=1
		#_f,_f = histogram_gaussian_function(run_no,(fore-back)/RH_sum,t_store,rB_temp, 'Counts/DS', 0,20)
		#pltFig += 1
		#rB_temp = [4230,9768,11217,14503]
		#rB_temp = [9768,11217,14503]
		#_f,_f = histogram_gaussian_function(run_no,GV_sum,t_store,rB_temp, 'GV', 0,20)
		#pltFig += 1
		#rB_temp = [4230,9768,11217,14503]
		#rB_temp = [9768,11217,14503]
		#_f,_f = histogram_gaussian_function(run_no,DS_sum,t_store,rB_temp, 'DS', 0,20)
		plt.show()
	
	plotter.plotVsExp = False	
	if plotter.plotVsExp: # This is just unl / mon_i on a run-by-run basis
		hold_list = [20,50,100,200,1550]
		for t in hold_list:
			pltFig = plot_run_vs_exp(run_no, t_store, fore, gv_sum, t, run_breaks, pltFig)
			pltFig = plot_run_vs_exp(run_no, t_store, fore, sp_sum, t, run_breaks, pltFig)
			pltFig = plot_run_vs_exp(run_no, t_store, fore, rh_sum, t, run_breaks, pltFig)
			pltFig = plot_run_vs_exp(run_no, t_store, fore, rhac_sum, t, run_breaks, pltFig)
		plt.show()
	import sys
	sys.exit()
	
	# Below here we did multi-monitor fits
	# Combine multiple monitors, starting with low
	nDet1 = 4
	nDet2 = 8
	if nDet1 == 3 or nDet2 == 3: # Forcing low to go first (stability reasons)
		low_mon = GV_sum
	elif nDet1 == 8 or nDet2 == 8:
		low_mon = RH_sum
	else:
		low_mon = GV_sum # you're forced to use GV normalization anyways
	
	if nDet2 == 5 or nDet1 == 5: # Now high monitor
		hi_mon  = SP_ave
	elif nDet2 == 4 or nDet1 == 4:
		hi_mon  = RHAC_sum
	else:
		hi_mon  = np.zeros(len(SP_ave)) # Default to no spectral correction
		
	monS = [] # Combine monitors
	for i in range(0,len(run_no)):
		monS.append([low_mon[i],hi_mon[i]])
	monS = np.array(monS)
	
	# Set fitting format here
	#fitFmt = 'linear'
	#fitFmt = 'linear_2det'
	fitFmt = 'linear_inv_2det'
	
	parameters = np.zeros(len(run_breaks))
	pov_matrix = np.zeros(len(run_breaks))
	t_fit = 0 # time to use for normalization
	#parameters, pov_matrix,pltFig =  plot_and_fit_unloads(run_no,t_store,(fore-back)/eff,monS,t_fit, run_breaks,fitFmt,eff,pltFig)
	rB_temp = [9767,9960,10936,10988,11085,11669,12516,14509]
	parameters, pov_matrix,pltFig =  plot_and_fit_unloads(run_no,t_store,SP_ave,GV_sum,t_fit, rB_temp,'linear',eff,pltFig)
	plotUnloads = True
	if plotUnloads:
		plt.legend()
		#plt.ylim(0,max(fore/eff))
		#plt.xlabel('Low Monitor Counts')
		#plt.ylabel('fore-back (20 s store)')
		plt.ylim(0,max(SP_ave))
		plt.xlabel('GV Monitor Counts')
		plt.ylabel('SP Monitor Counts')
		#plt.show()
	else:
		plt.close()
	
	parameters, pov_matrix,pltFig =  plot_and_fit_unloads(run_no,t_store,DS_sum,GV_sum,t_fit, rB_temp,'linear',eff,pltFig)
	if plotUnloads:
		plt.legend()
		#plt.ylim(0,max(fore/eff))
		#plt.xlabel('Low Monitor Counts')
		#plt.ylabel('fore-back (20 s store)')
		plt.ylim(0,max(DS_sum))
		plt.xlabel('GV Monitor Counts')
		plt.ylabel('DS Monitor Counts')
		plt.show()
	else:
		plt.close()
	
	_f,_f = histogram_gaussian_function(runNum,ctsList,tStore,runBreaks = [], leg = [], holdT = 0, nBins=0)
	sys.exit()
	#------------------------------------------------------------------
	# In finding the correlation between the GV monitor counts and the foreground counts, 
	# I have attempted the following two functions:
	#
	#   1. a+b⋅x
	#   2. a⋅x^b
	#
	# Not clear which form is better. I have tried both and it does not 
	# have a significant impact on the resulting neutron lifetime.
	#
	# Note that
	#
	#   - only 20-s runs are used to construct the correlation function.
	#   - three segments has different fitted parameters: a and b
	#   - later studies show that it is important to remove the background 
	#	  counts from the foreground in building the correct normalization function. 
	#     Without the removal of background, the derived neutron lifetime 
	#     is 3 second shorter.
	#
	# These functions are used to make a normalization function:
	#-------------------------------------------------------------------
	
	if fitFmt == 'linear':
		def fit_function(x,a,b):
			return linear(x,a,b)
	elif fitFmt == 'linear_2det':
		def fit_function(x,a,b):
			return linear_2det(x,a,b)			
	elif fitFmt == 'linear_inv_2det':
		def fit_function(x,a,b):
			return linear_inv_2det(x,a,b)	
	else: sys.exit(" Unknown fitting function!")
	
	# normalization function derived from the 20 s data
	# Calculate Expanded Uncetainty (due to covariance)
	norm_unl=np.zeros(len(run_no))
	sigma = np.zeros(len(run_no)) # Calculated from covariance
	sigma_stat = np.zeros(len(run_no)) # Calculated as pure statistical
	for i,r in enumerate(run_breaks):
		if i == len(run_breaks)-1:
			continue
		if parameters[i].all() == 0:
			continue
		if i == len(run_breaks) - 2:
			cond = (run_no>=run_breaks[i])*(run_no<=run_breaks[i+1])
		else:
			cond = (run_no>=run_breaks[i])*(run_no<run_breaks[i+1])
		
		norm_unl+=fit_function(monS,*parameters[i])*cond
		if fitFmt == 'linear_inv_2det':
			sigma_stat+=fit_function(monS,*parameters[i])*cond
			sigma+=calc_inv_norm_sigma(parameters[i],pov_matrix[i],monS[:,0],monS[:,1])*cond
		elif fitFmt == 'linear_2det':
			sigma_stat+=fit_function(monS,*parameters[i])*cond
			sigma+=calc_linear_norm_sigma(parameters[i],pov_matrix[i],monS[:,0],monS[:,1])*cond
		elif fitFmt == 'linear':
			sigma_stat+=fit_function(monS[:,0],*parameters[i])*cond
			sigma+=calc_one_norm_sigma(parameters[i],pov_matrix[i],monS[:,0])*cond
	
	plotSigmas = False
	if plotSigmas:
		plt.figure(pltFig)
		pltFig += 1
		for i in range(0,len(run_breaks)-1):
			fmt = color_run_breaks(i,run_breaks)
			condition = (run_no>=run_breaks[i])*(run_no<run_breaks[i+1])
			plt.plot(sigma2[condition],np.sqrt(sigma[condition])/sigma2[condition], '.',color=fmt)
		plt.show()
	
def chen_yu_likelihood_fit(runList,cts,nMon,dips=range(0,3),runBreaks=[]):
	#-------------------------------------------------------------------
	# This is chen-yu's likelihood fit for coincidences. I'm going 
	# to copy this blindly/exactly then do stuff with it later.
	#
	# Chen-Yu did all the work on this (apart from the extracted values)
	# 
	# Comments are mostly Chen-Yu's JuPyter notebook
	#-------------------------------------------------------------------
	
	
	
	# Start here with extracting our values
	pltFig = 0
	if len(runBreaks) == 0:
		#runBreaks=[4230,4672,5713,6753,6960,7326,9768,9960,10940,10988,11085,11217,12514,13209,14508] # Added 13307 
		#runBreaks=[4230,4672,5713,5955,6125,6429,6753,6960,7326,7490,9768,9960,10940,10988,11085,11217,12514,13209,13307,14508]
		runBreaks=[4223,4672,5713,5955,6125,6429,6753,6960,7326,7490,9767,9960,10936,10988,11085,11669,12516,13209,14509]
		#6367,6390,
	goodR, timeL, _x, _x, monL, sBoth, ctsF, bSub, MAT,bCorr = extract_values(runList, cts, nMon, 3, 8, 0, True, True, dips,runBreaks)
	_x,    _x,    _x, _x, monU, _x,    _x,   _x ,  _x, _x    = extract_values(runList, cts, nMon, 4, 5, 0, True, True, dips,runBreaks)
	_x,    _x,    _x, _x, monX, _x,    _x,   _x ,  _x, _x    = extract_values(runList, cts, nMon, 7, 6, 0, True, True, dips,runBreaks)
	
	run_no  = np.array(goodR)
	t_store = np.zeros(len(timeL))
	for i in range(len(timeL)):
		t_store[i] = int(round(float(timeL[i]))) # Have to round holding times
	
	# TODO: Also might need to extract arrival times.
	fore    = np.array(ctsF)
	back    = np.array(bSub)
	bTDep   = np.array(bCorr)
	back += bTDep
	mat     = np.array(MAT)
	eff = np.zeros(len(run_no))
	for i in range(len(sBoth)):
		eff[i] = sBoth[i][0]+sBoth[i][1]
		
	gvSum = []
	rhSum = []
	for gv,rh in monL:
		gvSum.append(float(gv))
		rhSum.append(float(rh))
	GV_sum = np.array(gvSum)
	RH_sum = np.array(rhSum)
	
	rhacSum = []
	spSum   = []
	for ac,sp in monU:
		rhacSum.append(float(ac))
		spSum.append(float(sp))
	RHAC_sum = np.array(rhacSum)
	SP_ave   = np.array(spSum)
	
	dsSum = []
	for ds,ac in monX:
		dsSum.append(float(ds))
	DS_sum = np.array(dsSum)
	# And some printout functions
	print('run numbers: \n', run_no[0], '----', run_no[len(run_no)-1])
	run_breaks = []
	for i in range(0,len(runBreaks)-1): # reduce the number of runBreaks
		if runBreaks[i]<=run_no[0] < runBreaks[i+1]: 
			run_breaks.append(runBreaks[i]) # find first
			for j in range(i+1,len(runBreaks)):
				if runBreaks[j] <= run_no[len(run_no)-1]:
					run_breaks.append(runBreaks[j]) # append through the last break
				else:
					break
			break
	if run_breaks[-1] < max(run_no): # And possibly add the last run.
		run_breaks.append(max(run_no))
	print(run_breaks)	
	#-------------------------------------------------------------------
	# "Poisson"-ness checks
	#-------------------------------------------------------------------
	# Above is the range of run number I will analyze and the time duration
	# when the data were collected.
	# Let's first take a look at the statistical distribution of the background counts:
	#
	# I am interested to know whether the recorded background counts follows 
	# a Poisson distribution:
	#
	# P(\mu,x)= \frac{\mu^x}{x!} e^{−\mu}
	#
	# where x is the integer count number, and \mu is the averaged number of counts.
	#
	# The dagger detector was damaged, but repaired for run number > 13209.
	# So we will look into run segments separately.
	# For run number >= 13209
	#-------------------------------------------------------------------
	
	plotPoisson = False
	if plotPoisson:
		# Loop through and plot the backgrounds inside our run_breaks
		for i in range(0,len(run_breaks)-1):
			if len(run_no[(run_no >= run_breaks[i])*(run_no < run_breaks[i+1])]) > 0:
				plt.figure(pltFig)
				pltFig+=1
				rB_Med = [run_breaks[i],run_breaks[i+1]]
				p, c_m = histogram_poisson_function(run_no,back,t_store,rB_Med, "Background",0)
		# and same thing for foregrounds
		for i in range(0,len(run_breaks)-1):
			if len(run_no[(run_no >= run_breaks[i])*(run_no < run_breaks[i+1])]) > 0:
				plt.figure(pltFig)
				pltFig+=1
				rB_Med = [run_breaks[i],run_breaks[i+1]]
				p, c_m = histogram_poisson_function(run_no,fore,t_store,rB_Med, "Foreground",20,25)
		plt.show()

	#-------------------------------------------------------------------
	# As shown above, the fit to a gaussian function with sigma=sqrt(N) does not work very well. -> The foreground counts from the 20 s storage runs have a spread much larger than the statistical fluctuation of a mean counts of 11099.
	# We will need to implement the normalization corrections.
	#
	# There are several monitors:
	#
	#   - Gatevalve monitor (GV)
	#   - Roundhouse monitor (RH)
	#   - Roundhouse active cleaner (RHAC)
	#   - Standpipe detector (SP)
	#   - Downstream monitor (not analyzed): this detector monitors the leakage through the cat door. It could be important to refine the normalization.
	#   - Active cleaner (only implemented in 2019 and beyond)
	#   - Round house counts in the end of the fill (not yet implemented)
	#
	# Chris also implemented a "fermi-function" weighting scheme to integrate
	# the counts measured by the RHAC. He divides the rate during the filling
	# time into 12 zones. Integrate the numbers in the zones with the following weighting:
	#
	# [0, 0, 0, 0, 0, 0, 0, 0.0752, 0.15862, 0.2402, 0.2848, 0.2411]
	#
	# FMG can't do this weighting scheme for now, everything's already
	# weighted. Use a different ./normNByDip-det.txt file 
	#-------------------------------------------------------------------
		
	plotRawMonitors = False
	if plotRawMonitors:
		plt.figure(pltFig) # Raw Counts
		pltFig += 1
		plot_monitor_by_run(run_no,RHAC_sum,"RH","Bare/RHAC")
		plot_monitor_by_run(run_no,GV_sum,"GV","GV")
		plot_monitor_by_run(run_no,RH_sum,"DS","Downstream")
		plot_monitor_by_run(run_no,SP_ave,"SP","SP")
		plt.yscale('log')
		#plt.figure(pltFig) # Scaled for comparison
		#pltFig += 1
		#plot_monitor_by_run(run_no,RHAC_sum,"RH","RHAC_sum")
		#plot_monitor_by_run(run_no,GV_sum,"GV","GV_sum")
		#plot_monitor_by_run(run_no,SP_ave*10,"SP","SP_ave x10")
		#plot_monitor_by_run(run_no,RH_sum*3,"RH","RH_sum x3")
		
		#plt.show()
		
	#-------------------------------------------------------------------
	# The bottom plots are the monitor counts rescaled to compare the general behaviors. 
	# It seems that Chris's weighted sum (x 5) follows closely to the RHAC_sum. 
	# This is not surprising as the weighted sum is calculated with the RHAC counts.
	#
	# On the other hands, the GV monitor, the RH detector(x3) and the SP monitor(x50) have similar behaviors 
	# --- after the installation of the cleaner inside the round house. 
	# Note that the long-term linear drift observed in the RHAC detector is less procouned in the other detector.
	#
	# Also note that the sawtooth behaviors shows the degradation of the solid D2 UCN source, 
	# and the improved UCN output after each source regeneration (melt & refreeze)
	# We will now attempt to normalize the foreground counts to the minotor counts:
	#-------------------------------------------------------------------
	
	plotYield = False
	if plotYield:
		plt.figure(pltFig) # Raw Yield
		pltFig+=1
		for i, t in enumerate(t_store):
			make_errorbar_by_hold(t,run_no[i],fore[i])
		plt.title("Raw Counts")
		plt.show()
		
	#-------------------------------------------------------------------
	# The bottom plot is the renormalized foregounds using GV 
	#
	# 1. The sawtooth features are removed, but
	# 2. The data is segmented with discrete jumps.
	# 3. There are long-term drifts in each data segments.
	#
	# This prompted Chris to implement the rolling average to remove these long-term drifts.
	#
	# Let's see how other detectors are correlated with the foreground counts:
	#-------------------------------------------------------------------

	#run_breaks = [11500,12514,13209,15000]
	#run_breaks = [11500,15000]
	
	plotMonVsCts = False
	if plotMonVsCts:
		pltFig = plot_counts_vs_mon(run_no, t_store, fore, GV_sum, 20, run_breaks, pltFig,'GV')
		pltFig = plot_counts_vs_mon(run_no, t_store, fore, SP_ave, 20, run_breaks, pltFig,'SP')
		pltFig = plot_counts_vs_mon(run_no, t_store, fore, RH_sum, 20, run_breaks, pltFig,'RH')
		pltFig = plot_counts_vs_mon(run_no, t_store, fore, RHAC_sum, 20, run_breaks, pltFig,'RHAC')
		plt.show()
	
	plotMonComps = False
	if plotMonComps:
		#fore_scale = (fore -back)* np.exp(t_store / 880.) # This might not work
		#pltFig = plot_counts_vs_mon(run_no, t_store, fore_scale, RHAC_sum/GV_sum, 0, run_breaks, pltFig,'RHAC/GV')
		#pltFig = plot_counts_vs_mon(run_no, t_store, fore_scale, SP_ave/GV_sum, 0, run_breaks, pltFig,'SP/GV')
		#pltFig = plot_counts_vs_mon(run_no, t_store, fore_scale, RH_sum/GV_sum, 0, run_breaks, pltFig,'DS/GV')
		pltFig = plot_counts_vs_mon(run_no, t_store, SP_ave, GV_sum, 0, run_breaks, pltFig,'SP vs. GV')
		pltFig = plot_counts_vs_mon(run_no, t_store, RH_sum, SP_ave, 0, run_breaks, pltFig,'DS vs. SP')
		#pltFig = plot_counts_vs_mon(run_no, t_store, RHAC_sum, GV_sum, 0, run_breaks, pltFig,'RHAC vs. GV')
		pltFig = plot_counts_vs_mon(run_no, t_store, RH_sum, GV_sum, 0, run_breaks, pltFig,'DS vs. GV')
		plt.show()
	
	#-------------------------------------------------------------------
	# Note that the foreground counts (measured by the dagger detector
	# in the end of each storage) has the strongest and cleanest correlation 
	# with the GV and RH monitor counts.
	#
	# The late 2018 data can be grouped into three run segments:
	#
	# 1. run number: 11105--11500 --> then, install the cleaner inside the round house.
	# 2. run number: 11501--13219 --> then, dagger detector damaged & repaired.
	# 3. run number: 13220--14503
	#
	# Let's normalize the foreground counts to the GV monitor:
	# 
	#-------------------------------------------------------------------
	
	plotNormalizedYields = False
	if plotNormalizedYields:
		# plt.figure(pltFig) # "Normalized yield
		# pltFig+=1
		# for i, t in enumerate(t_store):
			# make_errorbar_by_hold(t,run_no[i],(fore[i]-back[i])/GV_sum[i])
		# plt.title("Normalized to GV")
		# plt.xlabel("Run Number")
		# plt.ylabel("Normalized Counts (arb.)")
				
		# plt.figure(pltFig)
		# pltFig+=1
		# for i, t in enumerate(t_store):
			# make_errorbar_by_hold(t,run_no[i],(fore[i]-back[i])/SP_ave[i])
		# plt.title("Normalized to SP")
		# plt.xlabel("Run Number")
		# plt.ylabel("Normalized Counts (arb.)")
				
		# plt.figure(pltFig) # "Normalized yield
		# pltFig+=1
		# for i, t in enumerate(t_store):
			# make_errorbar_by_hold(t,run_no[i],(fore[i]-back[i])/RH_sum[i])
		# plt.title("Normalized to RH")
		# plt.xlabel("Run Number")
		# plt.ylabel("Normalized Counts (arb.)")
				
		# plt.figure(pltFig)
		# pltFig+=1
		# for i,t in enumerate(t_store):
			# make_errorbar_by_hold(t,run_no[i],(fore[i]-back[i])/RHAC_sum[i])
		# plt.title("Normalized to RHAC")	
		# plt.xlabel("Run Number")
		# plt.ylabel("Normalized Counts (arb.)")	
		
		# plt.figure(pltFig)
		# pltFig+=1
		# for i, t in enumerate(t_store):
			# make_errorbar_by_hold(t,run_no[i],fore[i]/back[i])
		# plt.title("Signal to Background")
		# plt.xlabel("Run Number")
		# plt.ylabel("Normalized Counts (arb.)")
				
		# plt.figure(pltFig)
		# pltFig+=1
		# for i,t in enumerate(t_store):
			# make_errorbar_by_hold(t,run_no[i],(RH_sum[i]/GV_sum[i]))
		# plt.title("RH/GV")	
		# plt.xlabel("Run Number")
		# plt.ylabel("Normalized Monitor Ratio (arb.)")
		
		# plt.figure(pltFig)
		# pltFig+=1
		# for i,t in enumerate(t_store):
			# make_errorbar_by_hold(t,run_no[i],(SP_ave[i]/GV_sum[i]))
		# plt.title("SP/GV")	
		# plt.xlabel("Run Number")
		# plt.ylabel("Normalized Monitor Ratio (arb.)")
		
		
		#plt.figure(pltFig)
		#pltFig +=1
		#for i,t in enumerate(t_store):
		#	make_errorbar_by_hold(t,run_no[i],back[i])
		#plt.title("Background (singles)")	
		#plt.xlabel("Run Number")
		#plt.ylabel("Normalized Monitor Ratio (arb.)")
		
		
		
		#plt.figure(pltFig)
		#pltFig += 1
		#rB_temp = [4230,9768,11217,14503]
		rB_temp = [9767,9960,10936,10988,11085,11669,12516,14509]
		#rB_temp = [9768,11217,14503]
		#_f,_f = histogram_poisson_function(run_no,GV_sum/RH_sum,t_store,rB_temp, 'GV/DS', 0,20)
		
		#pltFig+=1
		#_f,_f = histogram_gaussian_function(run_no,(fore-back)/RH_sum,t_store,rB_temp, 'Counts/DS', 0,20)
		#pltFig += 1
		#rB_temp = [4230,9768,11217,14503]
		#rB_temp = [9768,11217,14503]
		#_f,_f = histogram_gaussian_function(run_no,GV_sum,t_store,rB_temp, 'GV', 0,20)
		
		pltFig += 1
		#rB_temp = [4230,9768,11217,14503]
		#rB_temp = [9768,11217,14503]
		_f,_f = histogram_gaussian_function(run_no,DS_sum,t_store,rB_temp, 'DS', 0,20)
		
		plt.show()
	#sys.exit()	
	#-------------------------------------------------------------------
	# Voila!
	#
	#  - All sawtooth features are gone.
	#  - The dominant long-term drift is removed --> Rolling average could be omitted?
	#  - Need to investigate how the spectral variation lead to the sub-dominant drift.
	#    
	# We will segment the data sets into three segments,
	# as each segment has known changes of configurations of the dagger detector (run_no=13220) 
	# and the RH cleaner (run_no=11500).
	#
	# I also decide to use 20 s data to construct the normalization function, 
	# based solely on the counts of GV monitor.
	#
	#-------------------------------------------------------------------
	
	plotVsExp = False
	if plotVsExp:
		hold_list = [20,50,100,200,1550]
		for t in hold_list:
			pltFig = plot_run_vs_exp(run_no, t_store, fore, GV_sum, t, run_breaks, pltFig)
			pltFig = plot_run_vs_exp(run_no, t_store, fore, SP_ave, t, run_breaks, pltFig)
			pltFig = plot_run_vs_exp(run_no, t_store, fore, RH_sum, t, run_breaks, pltFig)
			pltFig = plot_run_vs_exp(run_no, t_store, fore, RHAC_sum, t, run_breaks, pltFig)
			plt.show()

	#-------------------------------------------------------------------
	# The above plot shows a fit to the normalized 20-s yield. 
	# The spread of data points agrees pretty well with the expected statistical fluctuation!
	#
	# The normalized 20-s yield clustered into three groups of mean counts (8700, 11200, 12200) 
	# for the three segments of data. 
	# This is an indication that the efficiency of UCN trapping and detection inside the trap changed.
	#
	# The objective of the normalization is to construct a function to predict/infer the N0 counts (at 20 s) using the recorded counts in the GV monitor in every single run (beyond the 20 s stores).
	#
	# Next, we will find the efficiency for each segment of data:
	#
	#-------------------------------------------------------------------
	
	# Combine multiple monitors, starting with low
	nDet1 = 4
	nDet2 = 8
	if nDet1 == 3 or nDet2 == 3: # Forcing low to go first (stability reasons)
		low_mon = GV_sum
	elif nDet1 == 8 or nDet2 == 8:
		low_mon = RH_sum
	else:
		low_mon = GV_sum # you're forced to use GV normalization anyways
	
	if nDet2 == 5 or nDet1 == 5: # Now high monitor
		hi_mon  = SP_ave
	elif nDet2 == 4 or nDet1 == 4:
		hi_mon  = RHAC_sum
	else:
		hi_mon  = np.zeros(len(SP_ave)) # Default to no spectral correction
		
	monS = [] # Combine monitors
	for i in range(0,len(run_no)):
		monS.append([low_mon[i],hi_mon[i]])
	monS = np.array(monS)
	
	# Set fitting format here
	#fitFmt = 'linear'
	#fitFmt = 'linear_2det'
	fitFmt = 'linear_inv_2det'
	
	parameters = np.zeros(len(run_breaks))
	pov_matrix = np.zeros(len(run_breaks))
	t_fit = 0 # time to use for normalization
	#parameters, pov_matrix,pltFig =  plot_and_fit_unloads(run_no,t_store,(fore-back)/eff,monS,t_fit, run_breaks,fitFmt,eff,pltFig)
	rB_temp = [9767,9960,10936,10988,11085,11669,12516,14509]
	parameters, pov_matrix,pltFig =  plot_and_fit_unloads(run_no,t_store,SP_ave,GV_sum,t_fit, rB_temp,'linear',eff,pltFig)
	plotUnloads = True
	if plotUnloads:
		plt.legend()
		#plt.ylim(0,max(fore/eff))
		#plt.xlabel('Low Monitor Counts')
		#plt.ylabel('fore-back (20 s store)')
		plt.ylim(0,max(SP_ave))
		plt.xlabel('GV Monitor Counts')
		plt.ylabel('SP Monitor Counts')
		#plt.show()
	else:
		plt.close()
	
	parameters, pov_matrix,pltFig =  plot_and_fit_unloads(run_no,t_store,DS_sum,GV_sum,t_fit, rB_temp,'linear',eff,pltFig)
	if plotUnloads:
		plt.legend()
		#plt.ylim(0,max(fore/eff))
		#plt.xlabel('Low Monitor Counts')
		#plt.ylabel('fore-back (20 s store)')
		plt.ylim(0,max(DS_sum))
		plt.xlabel('GV Monitor Counts')
		plt.ylabel('DS Monitor Counts')
		plt.show()
	else:
		plt.close()
	
	
	
	
	_f,_f = histogram_gaussian_function(runNum,ctsList,tStore,runBreaks = [], leg = [], holdT = 0, nBins=0)
	sys.exit()
	#------------------------------------------------------------------
	# In finding the correlation between the GV monitor counts and the foreground counts, 
	# I have attempted the following two functions:
	#
	#   1. a+b⋅x
	#   2. a⋅x^b
	#
	# Not clear which form is better. I have tried both and it does not 
	# have a significant impact on the resulting neutron lifetime.
	#
	# Note that
	#
	#   - only 20-s runs are used to construct the correlation function.
	#   - three segments has different fitted parameters: a and b
	#   - later studies show that it is important to remove the background 
	#	  counts from the foreground in building the correct normalization function. 
	#     Without the removal of background, the derived neutron lifetime 
	#     is 3 second shorter.
	#
	# These functions are used to make a normalization function:
	#-------------------------------------------------------------------
	
	if fitFmt == 'linear':
		def fit_function(x,a,b):
			return linear(x,a,b)
	elif fitFmt == 'linear_2det':
		def fit_function(x,a,b):
			return linear_2det(x,a,b)			
	elif fitFmt == 'linear_inv_2det':
		def fit_function(x,a,b):
			return linear_inv_2det(x,a,b)	
	else: sys.exit(" Unknown fitting function!")
	
	# normalization function derived from the 20 s data
	# Calculate Expanded Uncetainty (due to covariance)
	norm_unl=np.zeros(len(run_no))
	sigma = np.zeros(len(run_no)) # Calculated from covariance
	sigma_stat = np.zeros(len(run_no)) # Calculated as pure statistical
	for i,r in enumerate(run_breaks):
		if i == len(run_breaks)-1:
			continue
		if parameters[i].all() == 0:
			continue
		if i == len(run_breaks) - 2:
			cond = (run_no>=run_breaks[i])*(run_no<=run_breaks[i+1])
		else:
			cond = (run_no>=run_breaks[i])*(run_no<run_breaks[i+1])
		
		norm_unl+=fit_function(monS,*parameters[i])*cond
		if fitFmt == 'linear_inv_2det':
			sigma_stat+=fit_function(monS,*parameters[i])*cond
			sigma+=calc_inv_norm_sigma(parameters[i],pov_matrix[i],monS[:,0],monS[:,1])*cond
		elif fitFmt == 'linear_2det':
			sigma_stat+=fit_function(monS,*parameters[i])*cond
			sigma+=calc_linear_norm_sigma(parameters[i],pov_matrix[i],monS[:,0],monS[:,1])*cond
		elif fitFmt == 'linear':
			sigma_stat+=fit_function(monS[:,0],*parameters[i])*cond
			sigma+=calc_one_norm_sigma(parameters[i],pov_matrix[i],monS[:,0])*cond
	
	plotSigmas = False
	if plotSigmas:
		plt.figure(pltFig)
		pltFig += 1
		for i in range(0,len(run_breaks)-1):
			fmt = color_run_breaks(i,run_breaks)
			condition = (run_no>=run_breaks[i])*(run_no<run_breaks[i+1])
			plt.plot(sigma2[condition],np.sqrt(sigma[condition])/sigma2[condition], '.',color=fmt)
		plt.show()
	
	#-------------------------------------------------------------------
	# Want just this part!
	#-------------------------------------------------------------------
	# Maximum Likelihood Analysis:
	#
	# The likelihood function: 
	#   L(\tau)=\product_i{\frac{1}{N_i!}(N_{0i}e^{−t_i/\tau})^{N_i}e^{−N_{0i}e^{−ti/τ}}} 
	#	describes the probability of measuring our whole data sets the way it is.
	#
	# We calculate the log of the likelihood function: 
	#   M(\tau)=\ln{(L(\tau)}=\sum_i{−\ln{N_i!}+N_i \ln{N_{0i}} − N_i t_i/\tau − N_{0i} e^{−t_i/\tau}}.
	#
	# Here we need to input the data:
	#
	#   - N_i is the signal. We will start with N_i=fore
	#   - N_{0i} is the predicted 20 s counts based on the GV monitor counts.
	#   - t_i is t_store
	#
	# You then numerically calculate M by varying \tau. 
	# It seems that M(\tau) can be fit quite well by a quadratic function of a+b*(x−c)^2.
	#
	# Then the fitted parameter c gives the value of the lifetime, which maximize M, 
	# and thus maximize the likelihood function L.
	#
	# The fitted parameter b then gives the 1-sigma uncertainty in the lifetime: 
	# 	\sqrt{−1/(2∗b)}, which caused \Delta M = \Delta \ln(L)=−1/2
	#
	# Next, to properly take into account of the increased uncertainty in 
	# the N0 determination, we will add extra probability function in the 
	# likelihood function:
	#
	# L(\tau,N_{0i})=\product{i[1Ni!(N0ie−ti/τ+BG)Nie−(N0ie−ti/τ+BG)][BGBiBi!e−BG] [12π⋅2.7√N0ie−(N0i−N0GVi)2/(2.72N0i)] describes the probability of measuring our whole data sets the way it is.
	#
	# We calculate the log of the likelihood function: M(τ)=ln(L(τ))=∑i{−ln(Ni!)+Niln(N0ie−ti/τ+BG)−N0ieti/τ−BG−ln(Bi!)+Bi⋅ln(BG)−BG}.
	#
	# Here we need to input the measured data:
	#
	#   # Ni is the foreground.
	#   # Bi is the background.
	#   # ti is t_store
	#   # N0GVi is the predicted 20 s counts based on the GV monitor counts.
	#   # N0i is the normalized counts, properly distributed over the enlarge fluctuation of 2.7σN.
	#   # BG is the mean background counts, determined previously by fitting the background data points by a Poisson function (see the very beginning of the analysis). Because of the change of the dagger detector, we have two values of BG.
	#
	#-------------------------------------------------------------------
	# Poisson Likelihood function (with BG probability function)
	
	# calculate the M function for each data set ---------------------------
	from scipy.special import factorial
	
	tau0 = 887
	tau = np.zeros(50)
	M = np.zeros(50)
	
	# Debug here
	N = np.zeros(50)
	O = np.zeros(50)
	P = np.zeros(50)
	
	# Background correction -- probably need to incorporate uncertainty here.
	BG_correction = np.ones(len(run_no)) # position dependent background correction
	
	muB = np.zeros(len(run_no)) # mean background
	muN = np.zeros(len(run_no)) # mean normalization value
	muS = np.zeros(len(run_no)) # mean efficiency value
	alpha = np.zeros(len(run_no))
	beta  = np.zeros(len(run_no))
	# Calculate means of: Background, Normalization
	U = fore
	B = back#-BG_correction # add additional position dependent background.
	S = eff 
	for i, r in enumerate(run_breaks): # Only average inside run_breaks
		if i == len(run_breaks) - 1: # skip last break
			continue
		elif i== len(run_breaks)-2:
			condition = (run_no>=run_breaks[i])*(run_no<=run_breaks[i+1])
		else:
			condition = (run_no>=run_breaks[i])*(run_no<run_breaks[i+1])
		b_avg = np.mean(B[condition])
		b_avg = np.mean(B[condition])
		n_avg = np.mean(abs(norm_unl[condition]))
		s_avg = np.mean(eff[condition])
		#s_avg = np.mean(eff[condition]/(U-B)[condition]) * np.sum((U-B)[condition]) # Attempt at weighting
		#if i == len(run_breaks) - 2:
		muB += b_avg*condition
		#muB = B
		muN += n_avg*condition
		muS += s_avg*condition
		alpha += parameters[i][0]*condition
		beta  += parameters[i][1]*condition
		#else:
		#	muB += b_avg*(run_no>=run_breaks[i])*(run_no<run_breaks[i+1])
		#	muN += n_avg*(run_no>=run_breaks[i])*(run_no<run_breaks[i+1])
		#	alpha += parameters[i][0]*(run_no>=run_breaks[i])*(run_no<run_breaks[i+1])
		#	beta  += parameters[i][1]*(run_no>=run_breaks[i])*(run_no<run_breaks[i+1])
		
	#print(alpha,beta,muS)	
	# signal: fore[j]; 
	# N0: norm_GV[j]; 
	# background: back[j] 
	# t_i = t_store
	
	
	
	def exp_cts(tau,alpha,beta,dt,m1,m2):
		# Expected yields in the dagger
		# Initial counts should be estimated through alpha*m1+beta*m2/m1, then through time
		
		return (alpha*m1+beta*(m2/m1))*np.exp(-(dt-20.)/tau) # Setting 20s as scaled to 1.
		
	Y_vec = exp_cts(887.7,alpha,beta,t_store,low_mon,hi_mon)
	plt.figure(40)
	for i in range(0,len(Y_vec)):
		make_errorbar_by_hold(t_store[i],run_no[i],(Y_vec[i]-(U[i]-muB[i])/S[i])/np.sqrt(2*Y_vec[i]))
	plt.figure(41)
	for i in range(0,len(B)):
		#print(B[i],muB[i])
		make_errorbar_by_hold(t_store[i],run_no[i],B[i] - muB[i])
	plt.figure(42)
	for i in range(0,len(S)):
		#print(B[i],muB[i])
		make_errorbar_by_hold(t_store[i],run_no[i],S[i] - muS[i])
	
	plt.show()
	
	def eff_scale(sVar,cts,s):
		# Efficiency factor likelihood
		# cts should be scaled somehow since it's not a real Gaussian
		
		return -(s-sVar)**2 / s # Assuming Gaussian
	
	def bkg_cts(bkgVar,S_mu,b):
		# "Expected" background likelihood
		# This assumes that backgrounds can be modeled as a Poisson process (with a relevant scaling factor)
			
		return (b/S_mu)*np.log(bkgVar/S_mu)-(bkgVar/S_mu) - ((b/S_mu)*np.log(b/S_mu) - (b/S_mu)) # Poisson background (with stirling)
	
	#def binom_like(l,m,p):
		# This is the probability of a binomial distribution 
		# Or more accurately the log likelihood of that.
	#	ln(l!) - ln(m!) - ln((l-m)!) + l ln(p) (l-m) ln (1-p)
		
	
	def poisson_like(l,m):
		# This is the probability of a measurement m from an expected value l
		# numpy factorial gives an overflow error at 171! so I'll use Stirling's approx
		# above that
		m = np.array(m)
		like = m*np.log(l) - l - gammaln(m+1)
		#try:
		#	like  = np.zeros(len(m))
		
		#	like = m*np.log(l) - l - gammaln(m+1)
			#likeL[np.isnan(likeL)] = 0
			#like = likeH+likeL
		#except TypeError:
			#if m > 170:
				#like = m*np.log(l) - l - (m*np.log(m) - m)
		#	like = m*np.log(l) - l - gammaln(m+1)#(m*np.log(m) - m)
			#else:
			#	like = m*np.log(l) - l - np.log(factorial(m))
		#print(like)
		return like	
		#if m > 170:
		#	return m*np.log(l) - (m*np.log(m) - m) 
		#else:
		#	return m*np.log(l) - np.log(factorial(m))
	
	# Probably should code Gaussian/Other likelihood functions but w/e
	def unl_cts_summed(U,B,Y,S):
		# Poisson, summing across all possible background counts
		# For unload counts I'm using Stirling's approx for poisson counts
		
		#LL = -(Y + B/S) # Yield and (scaled) background.
		
		lnL = 0.0 # Start second part at 0
		#print(int(U/S))
		b1min = 0 #if B-5*(U-B) < 0 else B-5*(U-B)
		b1max = U #if B+5*(U-B) > U else B+5*(U-B)
		#print(b1max,B+5*(U-B))
		#b1 = np.linspace(0,int(U/S),int(U/S)+1) # Find the possible backgrounds to sum over
		b1 = np.linspace(b1min,b1max) # Find the possible backgrounds to sum over
		#print(Y-(U-B)/S)
		lnLArr = poisson_like(Y+b1/S,U/S) + poisson_like(B/S,b1/S)
		
		#lnLArr0 = ((U/S)*np.log(Y) - (Y) - ((U/S)*np.log(U/S) - (U/S)) # Poisson yield
		#			- (B/S)) # Background Yield (Stirling's approx fails for 0
				
		#lnLArr = ((U/S)*np.log(Y+b1) - (Y+b1) - ((U/S)*np.log(U/S) - (U/S)) # Poisson yield
		#			+ b1*np.log(B/S) - (B/S)  - ((b1) *np.log(b1 ) - (b1))) # Background Yield
		# Python factorial gives an overflow error at 171! (I tested on my laptop)
		# Above this we need to do Stirling's approx. (AKA all unload counts
		
		#			+ b1*np.log(B/S) - (B/S)  - (np.log(factorial(b1))))# *np.log(b1 ) - (b1))) # Background Yield
		# Now I'm going to be sneaky -- I have to exponentiate this
		# BUT 
		# It'll probably end up with an overflow error!
		#print(lnLArr)
		off = lnLArr.max() # This is the maximum lnL exponent
		#print(off)
		lnL = np.sum(np.exp(lnLArr)) # So we can offset this
		
		return np.log(lnL)# + off
		
	def unl_cts(U,B,Y,S):
		# Poisson with Stirling's Approx
		
		return (U/S)*np.log(Y+B/S) - (Y+B/S) - ((U/S)*np.log(U/S)-(U/S))
	
		#LL  = (u/S_mu)*np.log(Y_mu + B_mu/S_mu) - (Y_mu+B_mu/S_mu) # Poisson for counts
		#LL -= (u/S_mu)*np.log(u/S_mu)-u/S_mu # Stirling's approx for ln(U!)
			
	def ln_pdf_full(tau,alpha,beta,S_mu,B_mu,
					dt,u,b,s,m1,m1E,m2,m2E):
	
		# This is the probability distribution
		Y_mu = exp_cts(tau,alpha,beta,dt,m1,m2) # Expected UCN in unload
		
		# For Coincidences, S_mu must be 1.
		#s = 20.0 # Hardcoding for coinc
		#S_mu = 1.0
		#diff = (Y_mu - (u-b)/s) / np.sqrt(Y_mu)
		#if abs(diff) > 5:
		#	print(diff, dt, Y_mu-(u-b)/(s),m1E)
		bkgVar = bkg_cts(B_mu,S_mu,b) # Expected background counts
		sVar   = eff_scale(S_mu,u-b, s) # Expected scaling factor -- how many "events" are a UCN
		#unlVar = unl_cts_summed(u,B_mu,Y_mu,S_mu) # Variation in unload counts
		unlVar = unl_cts_summed(u,B_mu,Y_mu,s) # Variation in unload counts
		#unlVar2 = unl_cts(u,B_mu,Y_mu,S_mu)
		#LL  = (u/S_mu)*np.log(Y_mu + B_mu/S_mu) - (Y_mu+B_mu/S_mu) # Poisson for counts
		#LL -= (u/S_mu)*np.log(u/S_mu)-u/S_mu # Stirling's approx for ln(U!)		
		LL = unlVar
		LL += bkgVar + sVar # Incorporate nuisance factor scaling
		#if not np.isfinite(unlVar):
		#if unlVar > 0:
		#	print(dt,u,unlVar,bkgVar,sVar)
		#print(LL)
		return LL
		
	def lnL(param,data_arr):
		# param here is the list of parameters we're tuning.
				
		# The positive logarithm of the likelihood function
		# This is maximized by emcee, as a function of any model parameters theta
		
		# input the data and find the number of breaks:
		breaks = (len(param) - 1) / 2 # 3 parameters in param: tau, N, beta. Have 2*breaks n and betas
		
		tau = param[0] # lifetime, number of init. counts, and "temperature correction" factor beta
		
		# Physical requirement that counts are positive and lifetime is real
		if (any([N < 0.0 for n in param[1:breaks+1]]) or tau < 0.0): 
			return -np.inf, np.inf
		
		# Sum the log pdf for each data point
		#L_ = sum(map(lambda x : ln_pdf(tau,N,beta,*x[1:]),data_arr)) # Return ln_pdf cast from data as (tau,N,beta, filled_data)
		L_ = sum(
				starmap(ln_pdf,
						 [(tau,n,b,*d[1:])
							for (n,b,data) in zip(param[1:breaks+1],param[breaks+1:2*breaks+1],data_arr)
							for d in data
						 ]
						)
				)
		# return the likelihood to be maximized, and the chi2 for reference later
		return -0.5*L_, L_
	#-----------------------------------------------------------------------
	
	# Essentially copying this from Dan's code:
	raw_data = np.empty((len(run_no),7)) # 7 parameters 
	print(np.size(raw_data))
	for r in range(0,len(run_no)):
		#print(run_no[r],t_store[r],U[r],low_mon[r],hi_mon[r])
		raw_data[r] = np.array([(run_no[r],
								t_store[r],
								U[r], # Eventually add in B[r] and S[r]
								low_mon[r],0.0, # For now assume no uncertainty from monitor counts.
								hi_mon[r],0.0)])
	#raw_data = genfromtxt(argv[1])
	#runlist = list(genfromtxt(argv[2])) # List of runs -- don't need 
	#runbreaks = list(genfromtxt(argv[3])) # List of data 
	
	input_list = [] #Separate the numpy data arrays into a list for each run break
	for rb1,rb2 in zip(run_breaks,run_breaks[1:]+[1e6]): 
		input_list.append(np.array([x for x in raw_data if rb1<=x[0]<rb2]))
		#input_list.append(array([x for x in raw_data if x[0] in runlist and rb1<=x[0]<rb2]))
		
	#Start N values near the average 
	N_guesses = [np.average([x[2]/x[3] for x in data if 19.0<x[1]<21.0])*exp(-20.0/880.0) for data in input_list]
	# print(N_guesses)
	# Now on to emcee. ndim is number of model parameters + the value of lnf
	# breaks = (len(param) - 1) / 2 # 3 parameters in param: tau, N, beta. Have 2*breaks n and betas
	ndim, nwalkers = 1+2*len(run_breaks), 64

	# generate ensemble sampler, providing likelihood function, additional arguments, and any other options.
	# Each carbonate node has 24 cores, so I can run this for an hour in the debug queue.
	pool = Pool(24)
	sampler = emcee.EnsembleSampler(nwalkers,ndim,lnL,args=(input_list,),pool=pool)

	# generate initial points in parameter space for the walkers.
	# I start the walkers as spread normally about my best guess
	walker_init = [
					[np.random.normal(880.0,10.0)] \
					+[np.random.normal(N,sqrt(N)) for N in N_guesses] \
					+[np.random.normal(0.0,10.0) for _ in range(len(run_breaks))]
					for j in range(nwalkers)
					]

	then = datetime.now()
	print("Start Time:",then)

	# Run our markov chain monte carlo
	#sampler.run_mcmc(walker_init,40000)

	print("Elapsed Time:",(datetime.now()-then).total_seconds())
	pool.close()

	# Results here
	try:
		# Print the autocorr time -- this gives a measure
		# of how well we have converged -- if the autocorr times are too 
		# long compared to the total run time, we might have to run longer
		print(sampler.get_autocorr_time())
		
		# flatten out all the walkers into one list of walker points for 
		# each parameter discarding the first 500 for "burn-in" and keeping
		# only every so often, since points too nearby are correlated
		samples = sampler.get_chain(discard=500,thin=100,flat=True)
		
		# here "blob" is any additional return argument from my lnL 
		# function, which in my case is just a chi2 goodness of fit
		blobs   = sampler.get_blobs(discard=500,thin=100,flat=True)
		
		# Save output
		savetxt(argv[4],column_stack((samples,blobs)))	
	except Exception as e:
		print(e)
		
	def likelihood_scan(tStore):
		# Single Dimensional Likelihood Scan
		
		tau = np.zeros(50)
		M = np.zeros(50)
	
		# Debug here
		N = np.zeros(50)
		O = np.zeros(50)
		P = np.zeros(50)
			
		for i in range(0, 50): #loop over different tau
			tau[i]=tau0+(i-25)*0.1;
			t=tau[i]
			index = 0	
			
			for j in range(0, len(run_no)): #loop over the run number
				
				#parameters[i][0],parameters[i][1]
				
				tt = t_store[j]-20 # Don't bother subtracting t_store every time
				#if norm_unl[j] <= 0 or sigma[j] <= 0: # Should fix this 
				#	norm_unl[j] = abs(norm_unl[j])+1 # temp fix
				#	sigma[j] = abs(sigma[j])+1
					#print(run_no[j], monS[j])
				#	continue
				if (tt > -1) and (tt < 2000): # not 20 s runs and for singles can't do super long holds
					#ln_pdf_full(tau,alpha,beta,S_mu,B_mu,
					#	dt,u,b,s,m1,m1E,m2,m2E)
					#print(run_no[j])
					M[i] += ln_pdf_full(t,alpha[j],beta[j],muS[j],muB[j],
						t_store[j],U[j],B[j],S[j],low_mon[j],run_no[j],hi_mon[j],0.0)
					#print(norm_unl[j]*np.exp(-tt/t) - (U[j]-B[j])/S[j])
					#M[i] = M[i] + U[j]*np.log(norm_unl[j]*np.exp(-tt/t)+muB[j])-norm_unl[j]*np.exp(-tt/t)-muB[j]
					#M[i] = M[i] + U[j]*np.log(norm_unl[j]*np.exp(-tt/t)+B[j])-norm_unl[j]*np.exp(-tt/t)-B[j]
					#M[i] = M[i] - (U[j]*np.log(U[j]) - U[j]) # Stirling's approximation for ln(U!)
									
					#M[i] = M[i]  -(B[j] - muB[j])**2/(2.*muB[j])-np.log(np.sqrt(2.*np.pi*muB[j])) # Gaussian backgrounds
					
					
					P[i] = P[i]  -(B[j] - muB[j])**2/(2.*B[j])-np.log(np.sqrt(2.*np.pi*B[j])) # note that sigma is really sigma2 here.
					#M[i] = M[i] + B[j]*np.log(muB[j])-muB[j]
					#M[i] = M[i] - (np.log(factorial(B[j])))
					#M[i] = M[i] - (B[j]*np.log(B[j]) - B[j]) # Stirling's approximation for ln(B!)
					#M[i] = M[i]  -(norm_unl[j] - muN[j])**2/(sigma_stat[j])
					#M[i] = M[i] -np.log(np.sqrt(2.*np.pi*sigma[j])) # note that sigma is really sigma2 here.
					N[i] = N[i] -np.log(np.sqrt(2.*np.pi*sigma_stat[j])) # note that sigma is really sigma2 here.
					N[i] = N[i]  -(norm_unl[j] - muN[j])**2/(sigma_stat[j])-np.log(np.sqrt(2.*np.pi*sigma_stat[j])) # note that sigma is really sigma2 here.
					#O[i] = O[i]  -(norm_unl[j] - muN[j])**2/(sigma[j])-np.log(np.sqrt(2.*np.pi*sigma[j])) # note that sigma is really sigma2 here.
					#M[i] = M[i]  -(norm_unl[j] - muN[j])**2/(sigma[j])-np.log(np.sqrt(2.*np.pi*sigma[j])) # note that sigma is really sigma2 here.
					
					index = index + 1
			
				
		M = 0
		return M
	
	for i in range(0, 50): #loop over different tau
		tau[i]=tau0+(i-25)*0.1;
		t=tau[i]
		index = 0	
		
		for j in range(0, len(run_no)): #loop over the run number
			
			#parameters[i][0],parameters[i][1]
			
			tt = t_store[j]-20 # Don't bother subtracting t_store every time
			#if norm_unl[j] <= 0 or sigma[j] <= 0: # Should fix this 
			#	norm_unl[j] = abs(norm_unl[j])+1 # temp fix
			#	sigma[j] = abs(sigma[j])+1
				#print(run_no[j], monS[j])
			#	continue
			if (tt > 0) and (tt < 2000): # not 20 s runs
				#ln_pdf_full(tau,alpha,beta,S_mu,B_mu,
				#	dt,u,b,s,m1,m1E,m2,m2E)
				#print(run_no[j])
				M[i] += ln_pdf_full(t,alpha[j],beta[j],muS[j],muB[j],
					t_store[j],U[j],B[j],S[j],low_mon[j],run_no[j],hi_mon[j],0.0)
				#print(norm_unl[j]*np.exp(-tt/t) - (U[j]-B[j])/S[j])
				#M[i] = M[i] + U[j]*np.log(norm_unl[j]*np.exp(-tt/t)+muB[j])-norm_unl[j]*np.exp(-tt/t)-muB[j]
				#M[i] = M[i] + U[j]*np.log(norm_unl[j]*np.exp(-tt/t)+B[j])-norm_unl[j]*np.exp(-tt/t)-B[j]
				#M[i] = M[i] - (U[j]*np.log(U[j]) - U[j]) # Stirling's approximation for ln(U!)
								
				#M[i] = M[i]  -(B[j] - muB[j])**2/(2.*muB[j])-np.log(np.sqrt(2.*np.pi*muB[j])) # Gaussian backgrounds
				
				
				P[i] = P[i]  -(B[j] - muB[j])**2/(2.*B[j])-np.log(np.sqrt(2.*np.pi*B[j])) # note that sigma is really sigma2 here.
				#M[i] = M[i] + B[j]*np.log(muB[j])-muB[j]
				#M[i] = M[i] - (np.log(factorial(B[j])))
				#M[i] = M[i] - (B[j]*np.log(B[j]) - B[j]) # Stirling's approximation for ln(B!)
				#M[i] = M[i]  -(norm_unl[j] - muN[j])**2/(sigma_stat[j])
				#M[i] = M[i] -np.log(np.sqrt(2.*np.pi*sigma[j])) # note that sigma is really sigma2 here.
				N[i] = N[i] -np.log(np.sqrt(2.*np.pi*sigma_stat[j])) # note that sigma is really sigma2 here.
				N[i] = N[i]  -(norm_unl[j] - muN[j])**2/(sigma_stat[j])-np.log(np.sqrt(2.*np.pi*sigma_stat[j])) # note that sigma is really sigma2 here.
				#O[i] = O[i]  -(norm_unl[j] - muN[j])**2/(sigma[j])-np.log(np.sqrt(2.*np.pi*sigma[j])) # note that sigma is really sigma2 here.
				#M[i] = M[i]  -(norm_unl[j] - muN[j])**2/(sigma[j])-np.log(np.sqrt(2.*np.pi*sigma[j])) # note that sigma is really sigma2 here.
				
				index = index + 1
	
	
	
	n_runs= index
	print('Total number of runs=', len(run_no))
	print('Total 20 s (normalization) =', len(run_no)-index)
	print('Total runs used for lifetime analysis =', index)
	
	
	plt.plot(tau+totally_sick_blinding_factor,M,'bx',label='ln(L)')
	#plt.plot(tau+totally_sick_blinding_factor,N,'rx',label='ln(L)')
	#plt.plot(tau+totally_sick_blinding_factor,O,'gx',label='ln(L)')
	#plt.plot(tau+totally_sick_blinding_factor,P,'cx',label='ln(L)')
	MM=M
	popt2,pcov2 = curve_fit(quad,tau,MM,p0=[-1000,-4,tau0])
	
	plt.plot(tau+totally_sick_blinding_factor,quad(tau,*popt2),'r:',label='fit to quadratic function')
	plt.ylabel('M=log(L)')
	plt.xlabel('tau (s)')
	plt.legend()
	#plt.show()
	
	print("fit parameters = ", popt2)
	print('\033[31m' + 'Success!' + '\x1b[0m')
	print('\x1b[36m' + '*** lifetime (likelihood) = ' + str(popt2[2]) +  ' +/- ' + str(np.sqrt(-1/(2*popt2[1]))) + '\x1b[0m')
	#print('\x1b[0;31;47m' + '*** lifetime (likelihood) = ' + str(popt2[2]) +  ' +/- ' + str(np.sqrt(-1/(2*popt2[1]))) + '\x1b[0m')
	
	L=np.exp(quad(tau,*popt2)-popt2[0])
	plt.figure(69)
	plt.plot(tau+totally_sick_blinding_factor, L/np.max(L),'x',label='fit')
	plt.ylabel('Relative Likelihood Function')
	plt.xlabel('tau (s)')
	title_string = "No of runs =" + str(n_runs)
	plt.title(title_string)
		
	popt3,pcov3 = curve_fit(gaus,tau,L/np.max(L),p0=[1,tau0,1])
	print("\n fit parameters = ", popt3, pcov3)
	print("\n")
	print("*** lifetime (fitted) = ", popt3[1])
		
	plt.plot(tau+totally_sick_blinding_factor,gaus(tau,*popt3),'ro:',label='fit')
		
	# Make the shaded region
	popt3[1] += totally_sick_blinding_factor # Blinding
	a=popt3[1]-popt3[2]#+totally_sick_blinding_factor
	b=popt3[1]+popt3[2]#+totally_sick_blinding_factor
	ix = np.linspace(a, b)
	iy = gaus(ix,*popt3)
	plt.fill_between(ix,iy, facecolor='0.9', edgecolor='0.5', label='1 sigma band')
	plt.legend()
	plt.show()
	
	return [popt2[2],np.sqrt(-1/(2*popt2[1]))]
		
	#-------------------------------------------------------------------
	# Questions:
	#
	#   - How do we access the goodness of fit using the likelihood function? 
	#		What if the probability function (Poisson in this case) does not correctly 
	#		describe the true probability density function?
	#   - How do we perform the maximum likelihood analysis work for the singles data
	#		 (or current mode measurement)?
	#   - What to do about the hidden variable on the normalization?
	#
	# Discussons & Answers:
	#
	#   - Use Gaussian function with separate N and sigma (two variables) to allow for 
	#		larger fluctuations beyond the statistical variation.
	#   - Need to include the probability distribution of the N_{0i} in the
	#		 likelihood function.
	#   - Use Monte-Carlo data to construct a pior (or posterior) probability 
	#		density function.
	#   - To analyze the singles data (using the Poisson function), we might 
	#		need to allow a \mu factor (nuisance parameter and then marginalize it).
	#		Chris said that it is about 30 to 40 photons per UCN event.
	#   - The a and b parameters also need to have a probability density 
	#		functions that get multiplied in the likelihood function. 
	#		Perhaps the Markov Chain Monte-Carlo simulation could help to 
	#		establish this information.
	#
	# Remaining questions:
	#
	#   - The distribution of the background events is fitted to the Poisson function, 
	#		but the \chi^2 is bad!
	#   - No spectral correction implemented yet in the normalization function. 
	#		---> extra work is needed to reduced the \chi^2 from 2.7 to 1.
	#   - Need to make the Allen deviation plot.
	#-------------------------------------------------------------------



#-------------------------------------------------------------------

	# Above histogram shows the fore-back normalized by the GV monitor with adjusted efficiencies. Though the reduced \chi^2 is not good, the spread of the data is comparable to the statistical fluctuation.
	#
	# This is to be expected because the source output strength (and thus the UCN yields) varies with time.
	#
	# At least, we could try to compare the measured fore-back counts (20 s store) and the predicted yields (20 s) based on the GV monitor.
	#
	# Fit these two data sets by the function y=ax, and look at the distribution of the residuals:
	#--------------------------------------------------------------------
	    	
	# plt.figure(pltFig)
	# pltFig+=1
	# condition=(t_store==20)
	# irun=run_no[condition]
	# ix=norm_unl[condition]
	# #ix=monS[condition]
	# iy=fore[condition]-back[condition] # subtract the background
	# plt.plot(ix,iy,'x')
	# plt.xlabel('predicted fore w/ monitors')
	# plt.ylabel('fore (20s)')
	
	# # param,pov_matrix = curve_fit(fit_function, np.float64(ix), np.float64(iy), p0=[1])
	# # print("PREDICTED FORE W/ MONITORS")
	# # print("fit a, b = ", param, pov_matrix)
	# # print("delta_a,b = ", param*pov_matrix)
	
	# # # plot poisson-deviation with fitted parameter
	# # x_plot = np.linspace(0, max(ix), 1000)
	
	# # #plt.plot(x_plot, poisson(x_plot, *parameters), 'r-', lw=2)
	# # plt.plot(x_plot, fit_function(x_plot, *param), 'r-', lw=2)
		
	# # residual = iy-fit_function(ix, *param)
	# # chisq=sum(residual**2/iy) # Actually need to do chi2 with sigma, not sqrt(cts)
	# # dof=len(iy)-2
	# # plt.figure(pltFig)
	# # pltFig+=1
	# # print('dof=', dof,'; chisq/dof=', chisq/dof)
	# # plt.hist(residual/np.sqrt(np.float64(iy)))
	# # plt.xlabel('(y-y_mean)/sigma')
	
	# #--------------------------------------------------------------------
	# # We can calculate the \chi^2 assuming the uncertainty of each data point is purely statistical: \sqrt{N}.
	# #
	# # The reduced \chi^2 = 2.7 indicates a hidden variable causing the excessive fluctuations. 
	# # This is most likely due to the spectral variation of the UCN output.
	# #
	# # I would like to refine the normalization model by using the monitor counts, however, 
	# # due to the frequent change of the roundhouse active cleaner, 
	# # I am not sure the RHAC data is useful to tease out the spectral dependence. 
	# # Perhaps the SP monitor could be useful--> more work is needed.
	# #-------------------------------------------------------------------
	
	# plt.figure(pltFig)
	# pltFig+=1
	# # plot the normalized yield
	# plt.plot(irun, iy/ix,'x',label='20 s data')
	# plt.xlabel('Run no')
	# plt.ylabel('Normalized yield at 20 s')
	# plt.legend()
	
	# plt.figure(pltFig)
	# pltFig+=1
	# plt.plot(run_no, fore/norm_unl,'x', label='all data')
	# plt.plot(irun, iy/ix,'rx')
	# plt.xlabel('Run no')
	# plt.ylabel('Normalized yields')
	# plt.legend()
	# plt.show()
	
	# #-------------------------------------------------------------------
	# # As shown in the top plot, the normalized yield (at 20 s) varies by about 2% to 3%.
	# #
	# # The expected variation is 1/\sqrt{N}=1/\sqrt{11,000}−−−−−−√ = 0.9 %. 
	# # This is consistent with our earlier conclusion that the spread of the data is about 2.7 of the statistical fluctuation.
	# # We will look at the distribution of the normalized yield of the other storage times:
	# #-------------------------------------------------------------------
	
	# for i in range (0, 5):
	    # if i==0: 
	        # condition = (t_store==20)
	        # title_string = '20 s'
	        # init_N = 12000
	    # if i==1:
	        # condition = (t_store==50)
	        # title_string = '50 s'
	        # init_N = 11600
	    # if i==2:
	        # condition = (t_store==100)
	        # title_string = '100 s'
	        # init_N = 11000
	    # if i==3:
	        # condition = (t_store==200)
	        # title_string = '200 s'
	        # init_N = 9800
	    # if i==4:
	        # condition = (t_store==1550)
	        # title_string = '1550 s'
	        # init_N = 2200
	       
	    # plt.figure(pltFig)
	    # pltFig+=1
	    # yieldn = (fore-back)/norm_unl*(12000)
	    # entries, bin_edges, patches=plt.hist(yieldn[condition], bins=20)#, normed=True)
	    # plt.xlabel('Normalized yields')
	
	    # bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
	
	    # #def gaus(x,x0):
	     # #   return 1/np.sqrt(2*np.pi*x0)*np.exp(-(x-x0)**2/(2*x0))
	
	    # parG,coG = curve_fit(gaus_N,bin_middles,entries,p0=[init_N])
	
	    # # plot poisson-deviation with fitted parameter
	    # #x_plot = np.linspace(0, 100, 1000)
	    # x_plot = bin_middles
	    # #plt.plot(x_plot, poisson(x_plot, *parG), 'r-', lw=2)
	    # plt.plot(x_plot, gaus_N(x_plot, *parG), 'r-', lw=2)
	    # plt.title(title_string)
	    	    
	    # print("lamb = ", parG, coG)
	    # #print("delta_lamb = ", parG*coG)
	    # #print("1/sqrtN = ", 1/math.sqrt(len(test_array)))
	
	    # #for i in range (0, len(x_plot)):
	    # chisq = np.sum((entries-gaus_N(x_plot, *parG))**2/(gaus_N(x_plot,*parG))) # Assuming norm is gaussian, which it's not.
	    # dof = len(bin_middles)-1
	    # print('dof =', dof, '; chisq/dof =', chisq/dof)
	   
	# temperature correction
	# Ignore these plots -- I made better ones further up
	# plt.figure(pltFig)
	# pltFig+=1
	# condition = (t_store==20)*(run_no<13209)*(run_no>11500)
	# irun = run_no[condition]
	# temperature_RH=RHAC_sum/(6.5*RH_sum)
	# temperature_SP=SP_ave/GV_sum
	# itemperature_RH = temperature_RH[condition]
	# itemperature_SP = temperature_SP[condition]
	# plt.plot(irun, itemperature_RH,'x', label='Round House')
	# plt.plot(irun, itemperature_SP*60,'rx', label='Stand Pipe x 60')
	# plt.close()
	# plt.legend()
		
	# signal = (fore[condition]-back[condition])/(GV_sum[condition]*(1))
	# irun = run_no[condition]
	# plt.figure(pltFig)
	# pltFig+=1
	# plt.plot(irun, signal,'x')
	# plt.xlabel('Run Number')
	# plt.ylabel('Stand Pipe')
		
	# signal = (fore[condition]-back[condition])/(GV_sum[condition]*(1-0.06*itemperature_RH))
	# plt.figure(pltFig)
	# pltFig+=1
	# plt.plot(irun, signal,'x')
	# plt.xlabel('Round House')
	# plt.ylabel('Run Number')
# # Poisson Likelihood function (before BG subtraction)
	# # calculate the M function for each data set ---------------------------
	# tau0 = 903
	# tau = np.zeros(50)
	# M = np.zeros(50)
	
	# # signal: fore[j]; 
	# # N0: norm_GV[j]; 
	# # background: back[j] 
	# # t_i = t_store
	
	# S=fore
	
	# for i in range(0, 50): #loop over different tau
	    # tau[i]=tau0+(i-25)*0.1;
	    # t=tau[i]
	    # index = 0
	    # for j in range(0, len(run_no)): #loop over the run number
	        # tt = t_store[j]-20
	        # if (tt > 0): # not 20 s runs
	            # M[i] = M[i] + S[j]*np.log(norm_unl[j])- S[j]*tt/t-norm_unl[j]*np.exp(-tt/t)
	            # M[i] = M[i]- np.log(S[j]*(np.log(S[j])-1))         
	            # index = index + 1
	
	# n_runs= index
	# print("MAXIMUM LIKELIHOOD")
	# print('Total number of runs=', len(run_no))
	# print('Total 20 s (normalization) =', len(run_no)-index)
	# print('Total runs used for lifetime analysis =', index)
	
	# plt.figure(pltFig)
	# pltFig+=1
	# plt.plot(tau,M,'x',label='ln(L)')
	# plt.ylabel('M=log(L)')
	# plt.xlabel('tau (s)')
	# plt.legend()
	
	# MM=M
	
	# #def quad(x,a,b,c):
	# #    return a + b*(x-c)**2
	
	# popt2,pcov2 = curve_fit(quad,tau,MM,p0=[-1000,-4,tau0])
	
	# plt.plot(tau,quad(tau,*popt2),'r:',label='fit to quadratic function')
	# plt.ylabel('M=log(L)')
	# plt.xlabel('tau (s)')
	# plt.legend()
	
	
	# print("fit parameters = ", popt2)
	# print('\033[31m' + 'Success!' + '\x1b[0m')
	# print('\x1b[36m' + '*** lifetime (likelihood) = ' + str(popt2[2]) +  ' +/- ' + str(np.sqrt(-1/(2*popt2[1]))) + '\x1b[0m')
	
	# plt.figure(pltFig)
	# pltFig+=1
	# L=np.exp(quad(tau,*popt2)-popt2[0])
	# plt.plot(tau, L/np.max(L),'x',label='fit')
	# plt.ylabel('Relative Likelihood Function')
	# plt.xlabel('tau (s)')
	# title_string = "No of runs =" + str(n_runs)
	# plt.title(title_string)
	
	
	# popt3,pcov3 = curve_fit(gaus,tau,L/np.max(L),p0=[1,tau0,1])
	# print("\n fit parameters = ", popt3, pcov3)
	# print("\n")
	# print("*** lifetime (fitted) = ", popt3[1])
	
	# plt.plot(tau,gaus(tau,*popt3),'ro:',label='fit')
	
	# # Make the shaded region
	# a=popt3[1]-popt3[2]
	# b=popt3[1]+popt3[2]
	# ix = np.linspace(a, b)
	# iy = gaus(ix,*popt3)
	# plt.fill_between(ix,iy, facecolor='0.9', edgecolor='0.5', label='1 sigma band')
	# plt.legend()
	
	#-------------------------------------------------------------------
	# Without the background subtraction, the lifetime is way too big!
	# We then subtract the background in a naive fasion: N_i=fore−back
	#-------------------------------------------------------------------
	
	# # Poisson Likelihood function (after BG subtraction)
	# # calculate the M function for each data set ---------------------------
	# tau0 = 887
	# tau = np.zeros(50)
	# M = np.zeros(50)
	
	# # signal: fore[j]; 
	# # N0: norm_GV[j]; 
	# # background: back[j] 
	# # t_i = t_store
	# S=fore-back
	
	# for i in range(0, 50): #loop over different tau
	    # tau[i]=tau0+(i-25)*0.1;
	    # t=tau[i]
	    # index = 0
	    # for j in range(0, len(run_no)): #loop over the run number
	        # tt = t_store[j]-20
	        # if (tt > 0): # not 20 s runs
	            # M[i] = M[i] + S[j]*np.log(norm_unl[j])- S[j]*tt/t-norm_unl[j]*np.exp(-tt/t)
	            # M[i] = M[i]- np.log(S[j]*(np.log(S[j])-1))         
	            # index = index + 1
	# plt.figure(pltFig)
	# pltFig+=1
	# n_runs= index
	# print('Total number of runs=', len(run_no))
	# print('Total 20 s (normalization) =', len(run_no)-index)
	# print('Total runs used for lifetime analysis =', index)
	
	# plt.plot(tau,M,'x',label='ln(L)')
	# plt.ylabel('M=log(L)')
	# plt.xlabel('tau (s)')
	# plt.legend()
	
	# MM=M
	
# #	def quad(x,a,b,c):
# #	    return a + b*(x-c)**2
	
	# popt2,pcov2 = curve_fit(quad,tau,MM,p0=[-1000,-4,tau0])
	# plt.figure(pltFig)
	# pltFig+=1
	# plt.plot(tau,quad(tau,*popt2),'r:',label='fit to quadratic function')
	# plt.ylabel('M=log(L)')
	# plt.xlabel('tau (s)')
	# plt.legend()
		
	# print("fit parameters = ", popt2)
	# print('\033[31m' + 'Success!' + '\x1b[0m')
	# print('\x1b[36m' + '*** lifetime (likelihood) = ' + str(popt2[2]) +  ' +/- ' + str(np.sqrt(-1/(2*popt2[1]))) + '\x1b[0m')
	
	# L=np.exp(quad(tau,*popt2)-popt2[0])
	
	# plt.figure(pltFig)
	# pltFig+=1
	# plt.plot(tau, L/np.max(L),'x',label='fit')
	# plt.ylabel('Relative Likelihood Function')
	# plt.xlabel('tau (s)')
	# title_string = "No of runs =" + str(n_runs)
	# plt.title(title_string)
	
	# # gaussian function, parameter mu, sigma are the fit parameter
	# #def gaus(x,a,x0,sigma):
	# #   return a*np.exp(-(x-x0)**2/(2*sigma**2))
	
	# popt3,pcov3 = curve_fit(gaus,tau,L/np.max(L),p0=[1,tau0,1])
	# print("\n fit parameters = ", popt3, pcov3)
	# print("\n")
	# print("*** lifetime (fitted) = ", popt3[1])
	
	# plt.plot(tau,gaus(tau,*popt3),'ro:',label='fit')
	
	# # Make the shaded region
	# a=popt3[1]-popt3[2]
	# b=popt3[1]+popt3[2]
	# ix = np.linspace(a, b)
	# iy = gaus(ix,*popt3)
	# plt.fill_between(ix,iy, facecolor='0.9', edgecolor='0.5', label='1 sigma band')
	# plt.legend()
	# plt.show()
	
	#-------------------------------------------------------------------
	# To properly takes into account of the background, we will modify the likelihood function:
	#
	# L(τ)=∏i[1Ni!(N0ie−ti/τ+BG)Nie−(N0ie−ti/τ+BG)][BGBiBi!e−BG] describes the probability of measuring our whole data sets the way it is.
	#
	# We calculate the log of the likelihood function: M(τ)=ln(L(τ))=∑i{−ln(Ni!)+Niln(N0ie−ti/τ+BG)−N0ieti/τ−BG−ln(Bi!)+Bi⋅ln(BG)−BG}.
	#
	# Here we need to input the measured data:
	#
	#   - Ni is the foreground.
	#   - Bi is the background.
	#   - N0i is the predicted 20 s counts based on the GV monitor counts.
	#   - ti is t_store
	#   - BG is the mean background counts, determined previously by fitting the background data points by a Poisson function (see the very beginning of the analysis). Because of the change of the dagger detector, we have two values of BG.
	#
	# Poisson Likelihood function (with BG probability function)
	# calculate the M function for each data set ---------------------------
	#from scipy.special import factorial
	
	# tau0 = 887
	# tau = np.zeros(50)
	# M = np.zeros(50)
	
	# # signal: fore[j]; 
	# # N0: norm_GV[j]; 
	# # background: back[j] 
	# # t_i = t_store
	# S = fore
	# B = back
	
	# from scipy.special import factorial
	# # Likelihood calculation:
	# for i in range(0, 50): #loop over different tau
	    # tau[i]=tau0+(i-25)*0.1;
	    # t=tau[i]
	    # index = 0
	    # for j in range(0, len(run_no)): #loop over the run number
	        # tt = t_store[j]-20
	        # BG = 58.6
	        # if run_no[j] < 13209: BG = 56.5 # These numbers are wrong-ish, but work for now.
	        # if run_no[j] >=13209: BG = 31.78
	        # #if flag[j]==0: BG = 58.6
	        # #if flag[j]==1: BG = 37.13
	        # if (tt > 0): # not 20 s runs
	            # M[i] = M[i] + S[j]*np.log(norm_unl[j]*np.exp(-tt/t)+BG)-norm_unl[j]*np.exp(-tt/t)-BG
	            # M[i] = M[i]- np.log(S[j]*(np.log(S[j])-1))
	            # M[i] = M[i] -np.log(factorial(B[j]))+B[j]*np.log(BG)-BG
	            # index = index + 1
	
	# n_runs= index
	# print('Total number of runs=', len(run_no))
	# print('Total 20 s (normalization) =', len(run_no)-index)
	# print('Total runs used for lifetime analysis =', index)
	
	# plt.plot(tau,M,'x',label='ln(L)')
	# plt.ylabel('M=log(L)')
	# plt.xlabel('tau (s)')
	# plt.legend()
	
	# MM=M
	
	# #def quad(x,a,b,c):
	# #    return a + b*(x-c)**2
		
	# popt2,pcov2 = curve_fit(quad,tau,MM,p0=[-1000,-4,tau0])
	# # Fit the likelihood to a quadratic. The lifetime will be at the minimum, with error bars 1 sigma around.
	
	# plt.plot(tau+totally_sick_blinding_factor,quad(tau,*popt2),'r:',label='fit to quadratic function')
	# plt.ylabel('M=log(L)')
	# plt.xlabel('tau (s)')
	# plt.legend()
	# plt.show()
	
	# print("fit parameters = ", popt2)
	# print('\033[31m' + 'Success!' + '\x1b[0m')
	# print('\x1b[36m' + '*** lifetime (likelihood) = ' + str(popt2[2]) +  ' +/- ' + str(np.sqrt(-1/(2*popt2[1]))) + '\x1b[0m')
	# #print('\x1b[0;31;47m' + '*** lifetime (likelihood) = ' + str(popt2[2]) +  ' +/- ' + str(np.sqrt(-1/(2*popt2[1]))) + '\x1b[0m')
	
	# L=np.exp(quad(tau,*popt2)-popt2[0])
	
	# plt.plot(tau+totally_sick_blinding_factor, L/np.max(L),'x',label='fit')
	# plt.ylabel('Relative Likelihood Function')
	# plt.xlabel('tau (s)')
	# title_string = "No of runs =" + str(n_runs)
	# plt.title(title_string)
	
	# # gaussian function, parameter mu, sigma are the fit parameter
	# #def gaus(x,a,x0,sigma):
	# #    return a*np.exp(-(x-x0)**2/(2*sigma**2))
	
	# popt3,pcov3 = curve_fit(gaus,tau,L/np.max(L),p0=[1,tau0,1])
	# print("\n fit parameters = ", popt3, pcov3)
	# print("\n")
	# print("*** lifetime (fitted) = ", popt3[1])
	
	# plt.plot(tau+totally_sick_blinding_factor,gaus(tau,*popt3),'ro:',label='fit')
	
	# # Make the shaded region
	# a=popt3[1]-popt3[2]+totally_sick_blinding_factor
	# b=popt3[1]+popt3[2]+totally_sick_blinding_factor
	# ix = np.linspace(a, b)
	# iy = gaus(ix,*popt3)
	# plt.fill_between(ix,iy, facecolor='0.9', edgecolor='0.5', label='1 sigma band')
	# plt.legend()
	# plt.show()
	
	# Next, we add the position dependent background: -- ignore
	
	    # -0.62037 for run_no < 10163
	    # -0.53677 for 10164 < run_no < 13220
	    # -0.14884 for run_no >= 13220
	
	# Poisson Likelihood function (with BG probability function)
	# # calculate the M function for each data set ---------------------------
	# from scipy.special import factorial
	
	# tau0 = 887
	# tau = np.zeros(50)
	# M = np.zeros(50)
	# #BG_correction = np.ones(len(run_no)) # position dependent background correction -- already included in FMG's analysis
	# #for i in range(0, len(run_no)):
	# #  if run_no[i]<10163: BG_correction[i]=-0.62037
	# #  elif run_no[i]<13220: BG_correction[i]=-0.53677
	# #  else: BG_correction[i]=-0.14884
	
	# # signal: fore[j]; 
	# # N0: norm_GV[j]; 
	# # background: back[j] 
	# # t_i = t_store
	# S = fore
	# B = back-BG_correction # add additional position dependent background.
	
	# for i in range(0, 50): #loop over different tau
	    # tau[i]=tau0+(i-25)*0.1;
	    # t=tau[i]
	    # index = 0
	    # for j in range(0, len(run_no)): #loop over the run number
	        # tt = t_store[j]-20
	        # BG = 56.8
	        # #if flag[j]==0: BG = 58.6
	        # #if flag[j]==1: BG = 37.13
	        # if (tt > 0): # not 20 s runs
	            # M[i] = M[i] + S[j]*np.log(norm_GV[j]*np.exp(-tt/t)+BG)-norm_GV[j]*np.exp(-tt/t)-BG
	            # M[i] = M[i]- np.log(S[j]*(np.log(S[j])-1))
	            # M[i] = M[i] -np.log(factorial(B[j]))+B[j]*np.log(BG)-BG
	            # index = index + 1
	
	# n_runs= index
	# print('Total number of runs=', len(run_no))
	# print('Total 20 s (normalization) =', len(run_no)-index)
	# print('Total runs used for lifetime analysis =', index)
	
	# plt.plot(tau,M,'x',label='ln(L)')
	# plt.ylabel('M=log(L)')
	# plt.xlabel('tau (s)')
	# plt.legend()
	
	# MM=M
	
# #	def quad(x,a,b,c):
# #	    return a + b*(x-c)**2
	
	# popt2,pcov2 = curve_fit(quad,tau,MM,p0=[-1000,-4,tau0])
	
	# plt.plot(tau,quad(tau,*popt2),'r:',label='fit to quadratic function')
	# plt.ylabel('M=log(L)')
	# plt.xlabel('tau (s)')
	# plt.legend()
	# plt.show()
	
	# print("fit parameters = ", popt2)
	# print('\033[31m' + 'Success!' + '\x1b[0m')
	# print('\x1b[36m' + '*** lifetime (likelihood) = ' + str(popt2[2]) +  ' +/- ' + str(np.sqrt(-1/(2*popt2[1]))) + '\x1b[0m')
	# #print('\x1b[0;31;47m' + '*** lifetime (likelihood) = ' + str(popt2[2]) +  ' +/- ' + str(np.sqrt(-1/(2*popt2[1]))) + '\x1b[0m')
	
	# L=np.exp(quad(tau,*popt2)-popt2[0])
	
	# plt.plot(tau, L/np.max(L),'x',label='fit')
	# plt.ylabel('Relative Likelihood Function')
	# plt.xlabel('tau (s)')
	# title_string = "No of runs =" + str(n_runs)
	# plt.title(title_string)
	
	# # gaussian function, parameter mu, sigma are the fit parameter
# #	def gaus(x,a,x0,sigma):
# #	    return a*np.exp(-(x-x0)**2/(2*sigma**2))
	
	# popt3,pcov3 = curve_fit(gaus,tau,L/np.max(L),p0=[1,tau0,1])
	# print("\n fit parameters = ", popt3, pcov3)
	# print("\n")
	# print("*** lifetime (fitted) = ", popt3[1])
	
	# plt.plot(tau,gaus(tau,*popt3),'ro:',label='fit')
	
	# # Make the shaded region
	# a=popt3[1]-popt3[2]
	# b=popt3[1]+popt3[2]
	# ix = np.linspace(a, b)
	# iy = gaus(ix,*popt3)
	# plt.fill_between(ix,iy, facecolor='0.9', edgecolor='0.5', label='1 sigma band')
	# plt.legend()
	# plt.show()
	
# #-----------------------------------------------------------------------
	# # Likelihoods
	# #-----------------------------------------------------------------------
	
	# def mon1_cts(tau,alpha,beta,dt,y,m2):
		# # Inverse relationship for mon1 detector in case we need it
		# # TODO: Incorporate uncertainty into D,G,S(?)
		
		# return y / alpha*np.exp(dt/tau)-beta*m2
	
	# def mon2_cts(tau,alpha,beta,dt,y,m1):
		# # Inverse relationship for mon2 detector in case
		
		# return (y / alpha*np.exp(dt/tau) - m1) / beta

	
	# def ln_pdf(tau,alpha,beta,dt,y,m1,m1E,m2,m2E):
		# #The log of the PDF for each data point. Just a chi2 for now
		
		# Y_0  = d_cts(tau,alpha,beta,dt,m1,m2)
		# M1_0 = mon1_cts(tau,alpha,beta,dt,y,m2)
		# return (y-Y_0)**2 / y # Assuming Gaussian
	
	# #-------------------------------------------------------------------


	
#def linear(x,a,b):
#	return a*x+b
