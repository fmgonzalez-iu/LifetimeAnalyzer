#!/usr/local/bin/python3
#import sys
#import pdb
#import csv
#import datetime
#from math import *
#from statsmodels.stats.weightstats import DescrStatsW
#from scipy import stats, special, loadtxt, hstack
#from scipy.odr import *
from scipy.optimize import curve_fit#, nnls
#from datetime import datetime
#import corner

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from PythonAnalyzer.plotting import *

from PythonAnalyzer.functions import *
from PythonAnalyzer.classes import *
from PythonAnalyzer.dataIO import load_runs_to_datetime
from PythonAnalyzer.pairingandlifetimes import calc_lifetime_paired


#-----------------------------------------------------------------------
# Plotting color things
def color_run_breaks(i,runBreaks):
	# Breaks scatters into color schemes based on runbreaks
	
	# Find x as percentage of range
	try: x=float(i)/(len(runBreaks))
	except ZeroDivisionError: x=0.5
	
	# Convert to 255 bit ints through floats
	red   = int(min(max(4*(x-0.25),0.),1.)*255)
	green = int(min(max(4*fabs(x-0.5)-1.,0.),1.)*255)
	blue  = int(min(max(4*(0.75-x),0.),1.)*255)
	
	return "#%02x%02x%02x" % (red,green,blue)

def get_monitor_format(monitor):
	# Formatting if we're dealing with multiple monitors
	color = ''
	lines = ''
	
	if monitor=='GV':
		color = 'b'
		lines = 'solid'
	if monitor=='SP':
		color = 'y'
		lines = 'dotted'
	if monitor=='RH':
		color = 'c'
		lines = 'densely dashed'
	if monitor=='RHAC':
		color = 'r'
		lines = 'dashdotted'
	if monitor=='DS':
		color = 'g'
		lines = 'dashdotdotted'
	if monitor=='AC':
		color = 'm'
		lines = 'loosely dashed'
	if monitor=='dag':
		color = 'k'
		lines = 'dashed'

	if monitor=='LOW': # Low vs. High comparisons (if just 2)
		color = 'b'
		lines = 'solid'
	if monitor=='HIGH':
		color = 'r'
		lines= 'dashdotted'
		
	return color, lines
#-----------------------------------------------------------------------

def plot_and_fit_unloads(runNum,tStore,ctsSum,monitorSum,holdT = 20, runBreaks = [], fitFcn = 'linear',scale = [],nPlt = 1):
	set_plot_sizing()
	if min(runBreaks) > min(runNum):
		runBreaks.append(min(runNum))
	if max(runBreaks) < max(runNum):
		runBreaks.append(max(runNum))
	runBreaks.sort()
	
	plt.figure(nPlt)
	nPlt += 1
		
	parameters=np.ones((3,2))
	if len(scale) == 0:
		scale = np.ones(len(runNum))
	parList = []
	covList = []
	#parList = np.empty(len(runBreaks) -91) # output list of parameters and errors
	#covList = np.empty(len(runBreaks) -1)
	
	chisq = 0.0
	dof = 0
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
	
		testMon = np.array(monitorSum[condition])
		if len(testMon) == 0: # check that we loaded counts
			parameters = np.array([0.,0.]) # Put in zeros if we didn't
			pov_matrix = np.array([[0.,0.,],[0.,0.]])
			parList.append(parameters)
			covList.append(pov_matrix)
			continue
		testCts = ctsSum[condition]#/scale[condition] # Prior to this, subtract bkg
		fmt = color_run_breaks(i,runBreaks)
				
		# Now do fitting and plotting:
		try:
			x_plot=np.linspace(0.1,max(testMon),1000)			
		except ValueError:
			x_plot=np.linspace(0.1,max(testMon[:,0]),1000)
			
		# Figure out which fitting function to use
		# Quick note -- for ALL of these a (first variable) is the main mon
		#    correction. This way all fit functions can use the same "guess"
		if fitFcn == 'linear':
			def fit_function(x,a): # 1 dimensional linear
				return a*x
				#return linear(x,a,b)
			try:
				testMon=testMon[:,0]
			except IndexError:
				testMon=testMon[:] # You can't spell "Tautology" without "Tau"
			x_fit = x_plot
			guess=[testCts[0]/testMon[0]] # Initial slope is cts/monitor of first 
			plt.plot(testMon,testCts,'.',color=fmt,label = lbl)
		elif fitFcn == 'exponential':
			def fit_function(x,a,b): # 1 dimensional power (really basically linear
				return a*x**b
			testMon=testMon[:,0]
			x_fit = x_plot
			guess=[testCts[0]/testMon[0],1] # Linear with initial power
			plt.plot(testMon,testCts,'.',color=fmt,label = lbl)
		elif fitFcn == 'linear_2det':
			def fit_function(x,a,b): # 2 dimensional, with a ratio of two detectors
				return linear_2det(x,a,b)
			x_fit =np.ones((1000,2))
			x_fit[:,0]=x_plot
			x_fit[:,1] = x_plot*np.mean(testMon[:,1]/testMon[:,0])
			guess=[testCts[0]/testMon[0,0],0.]
			plt.plot(testMon[:,0],testCts,'.',color=fmt,label = lbl)
		elif fitFcn == 'linear_inv_2det':
			def fit_function(x,a,b):
				return linear_inv_2det(x,a,b)
			x_fit =np.ones((1000,2))
			x_fit[:,0] = x_plot
			x_fit[:,1] = x_plot*np.mean(testMon[:,1]/testMon[:,0])
			guess=[testCts[0]/testMon[0,0],0.]
			plt.plot(testMon[:,0],testCts,'.',color=fmt,label = lbl)
		else:
			print("Unknown fit function, reverting to LINEAR")
			fitFcn = 'linear'
			def fit_function (x,a,b):
				return a+b*x
			x_fit = x_plot
			guess=[testCts[0]/testMon[0]]
			plt.plot(testMon,testCts,'.',color=fmt,label = lbl)
				
		#parameters,pov_matrix = curve_fit(fit_function,np.float64(testMon),np.float64(testCts),p0=guess,bounds=([0.,-np.inf],[np.inf,np.inf]))
		parameters,pov_matrix = curve_fit(fit_function,np.float64(testMon),np.float64(testCts),p0=guess)
				
		print("Fitting",fitFcn,":")
		print("fit a,b = ", parameters,pov_matrix)
		print("delta_a,b = ", np.sqrt(np.diag(pov_matrix)))
		
		#w,v = np.linalg.eig(pov_matrix)
		#v = np.transpose(v)
		#print(w,v)
		#print(v*pov_matrix*np.linalg.inv(v))
		
		# Calculate correlated errors in normalization, assuming Gaussian
		# for the normalization monitors. This should be fine since they're high statistics.
		
		if fitFcn == 'linear_2det':
			#sigma2 = fit_function(testMon,*parameters) # assume Gaussian
			sigma2 = calc_linear_norm_sigma(parameters,pov_matrix,testMon[:,0],testMon[:,1])
		elif fitFcn == 'linear_inv_2det':
			#sigma2 = fit_function(testMon,*parameters)
			sigma2 = calc_inv_norm_sigma(parameters,pov_matrix,testMon[:,0],testMon[:,1])
		else:
			sigma2 = fit_function(testMon,*parameters)
			#sigma2 = calc_one_norm_sigma(parameters,pov_matrix,testMon)
					
		chisq += np.sum((testCts-fit_function(testMon, *parameters))**2/(sigma2))
		dof += len(testMon)
		
		plt.plot(x_plot, fit_function(x_fit, *parameters), '-',color=fmt,lw=2)		
		parList.append(parameters)
		covList.append(pov_matrix)
	
	dof -= 2 # Fitting 2 parameters
	print('\nchi2  =',chisq)
	print('dof =', dof, '; chisq/dof =', chisq/dof,"\n")
	plt.title(("Chi2/NDF = %f" % (chisq/float(dof))))
	#plt.show()
	#chisq = 0.
	
	#sys.exit()
	return parList, covList, nPlt

def plot_run_vs_exp(runNum,tStore,ctsSum,monSum,tHold = 20, runBreaks = [],nPlt = 1):
	# Plot the run vs. the expected value
	set_plot_sizing()
	if min(runBreaks) > min(runNum):
		runBreaks.append(min(runNum))
	if max(runBreaks) < max(runNum):
		runBreaks.append(max(runNum))
	runBreaks.sort()

	plt.figure(nPlt)
	nPlt +=1 
	for i, r in enumerate(runBreaks):
		if i==len(runBreaks)-1:
			continue	
		condition=(tStore==tHold)*(r<=runNum)*(runNum<runBreaks[i+1])
		runT = runNum[condition]
		monT = monSum[condition]
		ctsT = ctsSum[condition]
		fmt = color_run_breaks(i,runBreaks)
		plt.plot(runT,ctsT/monT,'.',color=fmt,label=('%d <= run_no < %d' % (r,runBreaks[i+1])))
		
		plt.xlabel('Run Number')
		plt.ylabel('Unload Counts / Monitor (%d s)' % tHold)
		plt.legend()
	return nPlt
	
def plot_counts_vs_mon(runNum, tStore, ctsSum, monitorSum, tHold = 20, runBreaks = [], nPlt = 1, monitor = 'Dagger',monitor2='GV' ):
	# This plots a scatter between two different monitors
	# or between monitors and unload counts
	set_plot_sizing()
	if min(runBreaks) > min(runNum):
		runBreaks.append(min(runNum))
	if max(runBreaks) < max(runNum):
		runBreaks.append(max(runNum))
	runBreaks.sort()

	plt.figure(nPlt)
	nPlt +=1 
	
	for i, r in enumerate(runBreaks):
		if i==len(runBreaks)-1:
			continue	
		if tHold > 0:
			condition=(tStore==tHold)*(r<=runNum)*(runNum<runBreaks[i+1])
			lbl = ('Unload Counts (%d s)' % tHold)
		else:
			condition=(r<=runNum)*(runNum<runBreaks[i+1])
			#lbl = 'Monitor Counts (scaled)'
			lbl = 'Monitor 2 Counts'
			
		monT = monitorSum[condition]
		ctsT = ctsSum[condition]
		fmt = color_run_breaks(i,runBreaks)
		for j, m in enumerate(monT):
			#fmt = color_run_breaks(j,monT)
			plt.plot(monT[j],ctsT[j],'.',color=fmt)#,label=('%d <= run_no < %d' % (r,runBreaks[i+1])))
		plt.title(monitor)
		plt.plot([],[],'.',color=fmt,label=('%d <= run_no < %d' % (r,runBreaks[i+1])))
		if tHold == 0:
			plt.xlabel('GV Counts')
			#plt.xlabel('Monitor Ratio')
		else:
			plt.xlabel('Monitor Counts')
		plt.ylabel(lbl)
		plt.legend()
	
	return nPlt

def plot_monitor_by_run(run_no,n_mon,mon_title = [],mon_label=[]):
	fc, fl = get_monitor_format(mon_title)
	if mon_label == []:
		mon_label = mon_title
	f = (f[0]+'-') # Convert to line
	plt.plot(run_no,n_mon,color=fc,linestyle=fl,label=mon_label)
	plt.title("Monitor Detectors")
	plt.xlabel("Run number")
	plt.ylabel("Counts")
	plt.legend()
	
def histogram_poisson_function(runNum,ctsList,tStore,runBreaks = [], leg = [], holdT = 0,nBins = 0):
	# For checking whether or not a given monitor is a Poisson distribution
	# Spoiler alert: They're not.
	set_plot_sizing()
	if len(runBreaks) < 2:
		if min(runBreaks) > min(runNum):
			runBreaks.append(min(runNum))
		if max(runBreaks) < max(runNum):
			runBreaks.append(max(runNum))
		runBreaks.sort()
	
	if len(leg) == 0:
		leg = 'Counts'
		
	# Find max, min, mean for plotting
	sMu  = np.mean(ctsList)
	sMin = 0
	sMax = 0
	if nBins == 0:
		sMin = int(min(ctsList) - (sMu/4))
		if sMin < 0:
			sMin = 0
		sMax = int(max(ctsList) + (sMu/4))
		
		if sMax - sMin < 500:
			nBins = int((sMax - sMin) + 1) # 1 UCN bin
		else:
			nBins = int((sMax - sMin)/30 + 1) # Assume 30 PE/UCN
	
	parList = np.empty(len(runBreaks) -1)
	covList = np.empty(len(runBreaks) -1)
	for i, r in enumerate(runBreaks):
		if i==len(runBreaks)-1:
			continue	
		if holdT > 0:
			condition=(tStore==holdT)*(r<=runNum)*(runNum<runBreaks[i+1])
			lbl = ("%d s" % holdT)
		else:
			condition=(r<=runNum)*(runNum<runBreaks[i+1])
			lbl = ("all times")
			
		test_array=ctsList[condition]	
		nRuns = len(test_array)
		if nRuns == 0:
			continue
		# Histogram our events
		#bins=np.linspace(0,110,111)
		if sMin > 0 and sMax > 0:
			bins=np.linspace(sMin,sMax,nBins)
			entries, bin_edges, patches = plt.hist(test_array, bins)
		else:
			entries, bin_edges, patches = plt.hist(test_array, bins=nBins)
		bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
		
		entries = np.array(entries)
				
		from scipy.optimize import curve_fit # I have to re-import these here or it breaks?
		from scipy.special import factorial
		def poisson(mu,k):
			return nRuns*(mu**k/factorial(k))*np.exp(-mu)
					
		parameters, cov_matrix = curve_fit(poisson, bin_middles, entries, p0=[sMu]) 
		
		x_plot = bin_middles
		f = make_errorbar_by_hold(holdT)
		f = (f[0]+'-')
		plt.plot(x_plot, poisson(x_plot, parameters), f, lw=2)
		plt.title('%d <= run_no < %d' % (r,runBreaks[i+1]))
		plt.ylabel(leg+' ('+lbl+')')
		
		print("Stats for %d <= run_no < %d" % (r, runBreaks[i+1]))
		print('no. runs = ', nRuns)
		print("mu = ", parameters, cov_matrix)
		print("delta_mu = ", parameters*np.sqrt(cov_matrix))
		print("1/sqrtN = ", 1/np.sqrt(len(test_array)))
		chisq = np.sum((entries-poisson(x_plot, *parameters))**2/(poisson(x_plot,*parameters)))
		dof = len(bin_middles)-1
		print('dof =', dof, '; chisq/dof =', chisq/dof,"\n")
		
		parList[i] = parameters
		covList[i] = cov_matrix
				
	return parList, covList

def histogram_gaussian_function(runNum,ctsList,tStore,runBreaks = [], leg = [], holdT = 0, nBins=0):
	set_plot_sizing()
	if len(runBreaks) < 2:
		if min(runBreaks) > min(runNum):
			runBreaks.append(min(runNum))
		if max(runBreaks) < max(runNum):
			runBreaks.append(max(runNum))
		runBreaks.sort()
	
	if len(leg) == 0:
		leg = 'Counts'
		
	# Find max, min, mean for plotting
	sMu  = np.mean(ctsList)
	sMin = 0
	sMax = 0
	if nBins == 0: # Or we can hardcode in bins
		sMin = int(min(ctsList) - (sMu/4))
		if sMin < 0:
			sMin = 0
		sMax = int(max(ctsList) + (sMu/4))
		
		if sMax - sMin < 150:
			nBins = int((sMax - sMin) + 1) # 1 UCN bin
		else:
			nBins = int((sMax - sMin)/30 + 1) # Assume 30 PE/UCN
	
	parList = np.empty(len(runBreaks) -1)
	covList = np.empty(len(runBreaks) -1)
	
	fig, axs = plt.subplots(len(runBreaks) - 1) # For subplots instead of figures
	for i, r in enumerate(runBreaks):
		if i==len(runBreaks)-1:
			continue	
		if holdT > 0:
			condition=(tStore==holdT)*(r<=runNum)*(runNum<runBreaks[i+1])
			lbl = ("%d s" % holdT)
		else:
			condition=(r<=runNum)*(runNum<runBreaks[i+1])
			lbl = ("all times")
			
		test_array=ctsList[condition]	
		tMu = np.mean(test_array)
		nRuns = len(test_array)
		
		if nRuns == 0:
			continue
		# Histogram our events
		#bins=np.linspace(0,110,111)
		if sMin > 0 and sMax > 0:
			bins=np.linspace(sMin,sMax,nBins)
			#entries, bin_edges, patches = plt.hist(test_array, bins,density=True)
			entries, bin_edges, patches = axs[i].hist(test_array, bins,density=True)
		else:
			#entries, bin_edges, patches = plt.hist(test_array, bins=nBins,density=True)
			weights = np.ones_like(test_array) / len(test_array) # hack to get density
			entries, bin_edges, patches = axs[i].hist(test_array, bins=nBins,weights=weights)#density=True)
		bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
		bin_width = bin_edges[1]-bin_edges[0]
		
		# Brute force renorm.
		#entries = entries / (np.sum(entries)*(bin_width*len(bin_middles)))
		#from scipy.optimize import curve_fit # I have to re-import these here or it breaks?
		#from scipy.special import factorial
		def gaus(x,x0):		
			if x0 > 0: # Counts must be positive!
				#return 1/np.sqrt(2*np.pi*30.*x0)*np.exp(-(x-x0)**2/(60.*x0))
				#return 1/np.sqrt(2*np.pi*x0)*np.exp(-(x-x0)**2/(2*x0))
				return np.exp(-(x-x0)**2/(2*x0))
			else:
				return np.inf
	
		
		#plt.plot(x_plot,test_array)
		#plt.show()
		#print(bin_middles,entries,tMu)
		print(tMu)
		#parameters, cov_matrix = curve_fit(gaus, np.float64(bin_middles), np.float64(entries), p0=[tMu]) 
		parameters, cov_matrix = curve_fit(gaus, bin_middles, entries, p0=tMu,bounds=(0.0,np.inf)) 
		
		x_plot = bin_middles
		chisq = 0.0
		print(tMu,parameters)
		for j,e in enumerate(entries):
			if gaus(x_plot[j], *parameters) > 0:
				chisq += (e-gaus(x_plot[j],*parameters))**2 / gaus(x_plot[j],*parameters)
				#chisq += (e-gaus(x_plot[j],tMu))**2 / gaus(x_plot[j],tMu)
			elif e != 0:
				chisq += 1
			else:
				chisq = np.inf
				break
		dof = len(bin_middles)-1
		
		print("Stats for %d <= run_no < %d" % (r, runBreaks[i+1]))
		print('no. runs = ', nRuns)
		print("mu = ", parameters, cov_matrix)
		print("delta_mu = ", parameters*cov_matrix)
		print("1/sqrtN = ", 1/sqrt(len(test_array)))
		
		print('dof =', dof, '; chisq/dof =', chisq/dof,"\n")
		f = make_errorbar_by_hold(holdT)
		f = (f[0]+'-')
	#	plt.plot(x_plot, gaus(x_plot, *parameters), f, lw=2)
	#	plt.title('%d <= run_no < %d, chi^2/NDF = %f' % (r,runBreaks[i+1],chisq/dof))
	#	plt.ylabel(leg+' ('+lbl+')')
		#axs[i].plot(x_plot, gaus(x_plot, *parameters), f, lw=2)
		axs[i].plot(x_plot, gaus(x_plot, tMu), f, lw=2)
		axs[i].set_title('%d <= run_no < %d' % (r,runBreaks[i+1]))
		#axs[i].ylabel(leg+' ('+lbl+')')
				
		#print(gaus(x_plot,*parameters))
		#chisq = 0.0
		#for j,e in enumerate(entries):
		#	if gaus(x_plot[j], *parameters) > 0:
		#		chisq += (e-gaus(x_plot[j],*parameters))**2 / gaus(x_plot[j],*parameters)
		#	elif e != 0:
		#		chisq += 1
		#	else:
		#		chisq = np.inf
		#		break
		#chisq = np.sum((entries-gaus(x_plot, *parameters))**2/(gaus(x_plot,*parameters)))
		
		#dof = len(bin_middles)-1
		#print('dof =', dof, '; chisq/dof =', chisq/dof,"\n")
		
		parList[i] = parameters
		covList[i] = cov_matrix
	plt.xlabel(leg)
	return parList, covList
