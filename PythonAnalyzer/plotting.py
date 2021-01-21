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

from PythonAnalyzer.functions import *
from PythonAnalyzer.classes import *
from PythonAnalyzer.dataIO import load_runs_to_datetime
from PythonAnalyzer.pairingandlifetimes import calc_lifetime_paired

#-----------------------------------------------------------------------
# If we want to blind things for plotting purposes:
#totally_sick_blinding_factor = 4.2069
totally_sick_blinding_factor = 0.0

#-----------------------------------------------------------------------
# I got fed up with this being a mess and divided the plotter into 3 
# different plotting functions.
#
# This has color coding formulae, as well as general lifetime plotting.
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# This contains a lot of plot-making stuff.
# Most of the (admittedly long) code is dedicated to making labels
#
# I've sorted these into:
#    Functions
#    Run-Number plots
#    Histograms
#    and Other
#-----------------------------------------------------------------------
def set_plot_sizing():
#-------------------------------------------------------------------
# Set font sizes for plots
# According to IU's regulations plots should be between 10 and 12 point.
# (https://graduate.indiana.edu/thesis-dissertation/formatting/doctoral.html)
# I'm setting these this up way.

	#plt.rcParams["font.family"] = "serif"
	#plt.rcParams["font.serif"] = "Times New Roman"
	SMALL_SIZE = 10
	MEDIUM_SIZE = 11
	BIGGER_SIZE = 12
	plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	
	return 0
#-------------------------------------------------------------------
	
#-----------------------------------------------------------------------
# General Plotting Functions
#-----------------------------------------------------------------------
def make_errorbar_by_hold(t,xVal = 0,yVal = 0,xErr=0,yErr=0):
	# This sorts runs by holding time, so that we can plot things neatly
	# Switch 4 cases: x and y with and without errors
		
	if 19.0 < t < 21.0: # Figure out formatting:
		f='r.'
	elif 49.0 < t < 51.0:
		f='y.'
	elif 99.0 < t < 101.0:
		f='g.'
	elif 199.0 < t < 201.0:
		f='b.'
	elif 1549.0 < t < 1551.0:
		f='c.'
	else:
		f='k.'
		
	# Figure out errorbar case	
	if xErr <= 0 and yErr > 0: # Case 1: x has no error, y does (most common)
		plt.errorbar(xVal, yVal, yerr=yErr, fmt=f)
	elif xErr > 0 and yErr > 0: # Case 2: x and y have error
		plt.errorbar(xVal, yVal, xerr=xErr, yerr=yErr, fmt=f)	
	elif yErr <= 0 and xErr > 0: # Case 3: x has error, y does not (rare!)
		plt.errorbar(xVal, yVal, xerr=xErr, fmt=f)
	elif xVal != 0 and yVal != 0: # Case 4: No errors!
		plt.plot(xVal,yVal,f)
	# Hidden case 5: Don't plot and just return format
		
	return f
	
def make_legend_by_hold(holdVec):
	# Run this function on a holdVec to generate a legend on a plot	
	set_plot_sizing()
	# Hardcode in 20, 50, 100, 200, 1550s, "other"
	timesInRun = []
	for t in holdVec: # figure out which runs we need
		if round(float(t)) not in timesInRun:
			timesInRun.append(round(float(t)))
			
	if (20 in timesInRun): # Now sort
		plt.errorbar([],[],fmt='r.',label="20s Hold")
		ind = int(np.argwhere(np.array(timesInRun) == 20))
		timesInRun.pop(ind)
	if (50 in timesInRun):
		plt.errorbar([],[],fmt='y.',label="50s Hold")
		ind = int(np.argwhere(np.array(timesInRun) == 50))
		timesInRun.pop(ind)
	if (100 in timesInRun):
		plt.errorbar([],[],fmt='g.',label="100s Hold")
		ind = int(np.argwhere(np.array(timesInRun) == 100))
		timesInRun.pop(ind)
	if (200 in timesInRun):
		plt.errorbar([],[],fmt='b.',label="200s Hold")
		ind = int(np.argwhere(np.array(timesInRun) == 200))
		timesInRun.pop(ind)
	if (1550 in timesInRun):
		plt.errorbar([],[],fmt='c.',label="1550s Hold")
		ind = int(np.argwhere(np.array(timesInRun) == 1550))
		timesInRun.pop(ind)
	if len(timesInRun) > 0: # Popping off timesInRun for other values
		plt.errorbar([],[],fmt='k.',label="Other Hold")
	plt.legend(loc='upper right') # Hardcoding location because otherwise it's laggy.
	
	return holdVec

def make_run_breaks_verts(runBreaks = [], colorCode=False,mkL=False):
	# Plotting utensil function to make vertical lines on runBreaks
	set_plot_sizing()
	if len(runBreaks) <= 0: # If there's no runBreaks, don't bother
		return runBreaks	# I'm paranoid, which is why there's '<='
	
	if not colorCode:
		for xl in runBreaks:
			plt.axvline(x=xl)
		return runBreaks
	
	# OK, now I'm hardcoding the runBreaks by "type" and thus color.
	rB_year = [4230,9767,14509] # Year breaks
	rB_al   = [4711,7326]       # Runs with the aluminum block
	rB_RH   = [9960,10936,10988,11085,11669,12516] # RH Changes
	rB_dag  = [13209] # Dagger Breaks
	rB_pmtG = [4304,5955,6126,6429,7490,13307]    # Dagger PMT Gain shifts
	rB_foil = [5713,6754,6930]               # Foil Gain shifts
	rB_src  = [4391,4415,5453,5475]          # Bad sources
	if mkL:
		plt.figure(10)

	for xl in runBreaks:
		if xl in rB_year:
			plt.axvline(x=xl,color='black')
		elif xl in rB_al:
			plt.axvline(x=xl,color='silver')
		elif xl in rB_RH:
			plt.axvline(x=xl,color='red')
		elif xl in rB_dag:
			plt.axvline(x=xl,color='lime')
		elif xl in rB_pmtG:
			plt.axvline(x=xl,color='green')
		elif xl in rB_foil:
			plt.axvline(x=xl,color='magenta')
		elif xl in rB_src:
			plt.axvline(x=xl,color='cyan')
	if mkL:
		plt.plot([],[],color='black',  label='New Year')
		plt.plot([],[],color='silver', label='Aluminum Block')
		plt.plot([],[],color='red',    label='Roundhouse')
		plt.plot([],[],color='lime',   label='Dagger Breaks')
		plt.plot([],[],color='green',  label='PMT Gain Shift')
		plt.plot([],[],color='magenta',label='Foil Gain Shift')
		plt.plot([],[],color='cyan',   label='Major Source Change')
		set_plot_sizing()
		
		plt.title("Discrete Changes in Normalization")
		plt.ylabel("N/A")
		plt.xlabel("Run Number")
		plt.legend(loc='upper right')
	return runBreaks

def make_histogram_by_hold(holdVec, dataVec, holdT = 20):
	# If we want to histogram given data by holding time
	set_plot_sizing()
	histBuff   = []
	for i, cts in enumerate(dataVec):
		if holdT - 1.0 < float(holdVec[i]) < holdT + 1.0:
			histBuff.append(float(cts))

	# Return mean and std too -- this lets us shift values if we want
	meanV = np.average(histBuff)
	meanE = np.std(histBuff) # Can get actual error with meanE / np.sqrt(len(histBuff))
	
	# Finally, return color scheme for histograms by holdT
	if holdT == 20:
		col = 'r'
		line = 'solid'
	elif holdT == 50:
		col = 'y'
		line = 'dashdotted'
	elif holdT == 100:
		col = 'g'
		line = 'dashdotdotted'
	elif holdT == 200:
		col = 'b'
		line = 'dotted'
	elif holdT == 1550:
		col = 'c'
		line = 'dashed'
	else:
		col = 'k'
		line = 'dashed'
	
	return histBuff, meanV, meanE, col, line
	
def make_histogram_short_long(holdVec, dataVec, short=True):
	set_plot_sizing()
	histBuff = []
	if short: # Just for separating short/long
		for i, cts in enumerate(dataVec):
			if float(holdVec[i]) <= 1000:
				histBuff.append(float(cts))
		col='r' # Short runs are solid
		line='solid'
	else:
		for i, cts in enumerate(dataVec):
			if float(holdVec[i]) > 1000:
				histBuff.append(float(cts))
		col='c' # Long runs are dashed
		line='dashed'
	
	# Return mean and std too -- this lets us shift values if we want
	meanV = np.average(histBuff)
	meanE = np.std(histBuff) # Can get actual error with meanE / np.sqrt(len(histBuff))
	
	# Now return means, buffers, format
	return histBuff, meanV, meanE, col, line
	
				
	
#-----------------------------------------------------------------------



def plot_lifetime_scatter(lifetime,lts,rates,bkgs,hlds,norms,plot_gen = 1):
	# This function was for a note to make plots for paired lifetimes vs.
	# assorted observable functions -- just to see if there was some 
	# trend that I wasn't paying attention to.
	set_plot_sizing()
	
	
	plt.figure(plot_gen)
	plot_gen += 1
	pmtStr = 'PMT 1, 2018'
	
	ltBins = np.linspace(-50,50,50)
	
	rS = np.zeros(len(lts))
	bS = np.zeros(len(lts))
	bRS = np.zeros(len(lts))
	normS = np.zeros(len(lts))
	
	ltS = np.zeros(len(lts))
	ltE = np.zeros(len(lts))
	
	for i in range(len(lts)):
		ltS[i] = lts[i].val-lifetime.val
		ltE[i] = lts[i].err
		if lts[i].err < 5.: # Minimum lifetime unc. should be more than 5
			print("Too Short!",lts[i])
			ltE[i] = np.inf
		if not (-50 < ltS[i] < 50):
			ltE[i] = np.inf
				
		rScale = rates[i][0]*np.exp((hlds[i][0] - 20)/lifetime)
		#plt.errorbar(lts[i].val,rScale.val,xerr=lts[i].err,yerr=rScale.err,fmt='b.')
		rS[i] = rScale.val
		
		bScale = bkgs[i][0]/bkgs[i][1]
		bS[i] = bScale.val
		if bS[i] > 2. or bS[i] < 0.5:
			print("Bad Ratio!")
			ltE[i] = np.inf
		
		bScale2 = bkgs[i][0]
		bRS[i] = bScale2.val	
		
		nScale = norms[i][0]
		normS[i] = nScale.val 
		#if ltS[i] > 100: # There's 2 bad pairs, I need to figure out why...
		#	print(hlds[i],norms[i],bkgs[i])
	#cR,coR = curve_fit(linear,ltS,rS,p0=(1000,0))
	lineR = np.linspace(np.min(rS)-(np.max(rS)-np.min(rS))*0.01,np.max(rS)+(np.max(rS)-np.min(rS))*0.01,50)
	cR,coR = curve_fit(linear,rS,ltS,sigma=ltE)#,p0=(1e-3,0))
	plt.errorbar(rS,ltS,yerr=ltE,fmt='b.',\
			 label=(r'$\tau_0 = %f \pm %f$' % (cR[0],np.sqrt(coR[0][0]))))
	plt.plot(lineR,linear(lineR,*cR),'c',\
			 label=(r'$\alpha = %f \pm %f$' % (cR[1],np.sqrt(coR[1][1]))))
	plt.ylim(-50,50)
	
	plt.title("Paired Lifetime vs. Raw Unload Rate, "+pmtStr)
	plt.ylabel("Lifetime")
	plt.xlabel("Average Unload Rate (Hz) Scaled to 20s")
	plt.legend()
	
	plt.figure(plot_gen)
	plot_gen += 1
	plt.hist2d(rS,ltS,bins=([lineR,ltBins]),cmap='cool')
	#H,xedges,yedges = np.histogram2d(rS,ltS,bins=(lineR,ltBins))
	#plt.imshow(H,cmap='spring',interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
	#plt.imshow([rS,ltS],cmap='spring',interpolation='nearest')
	#sns.heatmap([rS,ltS],cmap='spring')
	plt.title("Paired Lifetime vs. Raw Unload Rate, "+pmtStr)
	plt.ylabel("Lifetime")
	plt.xlabel("Average Unload Rate (Hz) Scaled to 20s")
	plt.colorbar()
	
	plt.figure(plot_gen)
	plot_gen += 1
	#for i in range(len(lts)):
		
		#plt.errorbar(lts[i].val,bScale.val,xerr=lts[i].err,yerr=bScale.err,fmt='r.')
		#plt.errorbar(lts[i].val-lifetime.val,bScale.val,fmt='r.')
	lineB = np.linspace(np.min(bS)-(np.max(bS)-np.min(bS))*0.01,np.max(bS)+(np.max(bS)-np.min(bS))*0.01,50)
	cB,coB = curve_fit(linear,bS-np.mean(bS),ltS,sigma=ltE)#,p0=(10000,0))
	plt.errorbar(bS,ltS,yerr=ltE,fmt='r.',\
			 label=(r'$\tau_0 = %f \pm %f$' % (cB[0],np.sqrt(coB[0][0]))))
	plt.plot(lineB,linear(lineB-np.mean(bS),*cB),'y',\
			 label=(r'$\alpha = %f \pm %f$' % (cB[1],np.sqrt(coB[1][1]))))
	#unc_0 = np.sqrt(coB[1][1] - np.sqrt(
	plt.ylim(-50,50)
	plt.legend()
	plt.title("Paired Lifetime vs. Background Ratio, "+pmtStr)
	plt.ylabel("Lifetime")
	plt.xlabel("Short Bkg. (Hz) / Long Bkg. (Hz)")
	
	plt.figure(plot_gen)
	plot_gen += 1	
	plt.hist2d(bS,ltS,bins=([lineB,ltBins]),cmap='autumn')
	#H,xedges,yedges = np.histogram2d(bS,ltS,bins=(lineB,ltBins))
	#plt.imshow(H,cmap='summer',interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
	#plt.xlim(np.min(lineB),np.max(lineB))
	#plt.ylim(-50,50)
	#sns.heatmap([bS,ltS],cmap='summer')
	plt.title("Paired Lifetime vs. Background Ratio, "+pmtStr)
	plt.ylabel("Lifetime")
	plt.xlabel("Short Bkg. (Hz) / Long Bkg. (Hz)")
	plt.colorbar()
	
	plt.figure(plot_gen)
	plot_gen += 1
	
		#plt.errorbar(lts[i].val-lifetime.val,bScale.val,xerr=lts[i].err,yerr=bScale.err,fmt='g.')
		#plt.errorbar(lts[i].val-lifetime.val,bScale.val,fmt='g.')
	lineBR = np.linspace(np.min(bRS)-(np.max(bRS)-np.min(bRS))*0.01,np.max(bRS)+(np.max(bRS)-np.min(bRS))*0.01,50)
	cB2,coB2 = curve_fit(linear,bRS,ltS,sigma=ltE)#,p0=(100,0))
	plt.errorbar(bRS,ltS,yerr=ltE,fmt='g.',\
			 label=(r'$\tau_0 = %f \pm %f$' % (cB2[0],np.sqrt(coB2[0][0]))))
	plt.plot(lineBR,linear(lineBR,*cB2),'chartreuse',\
			 label=(r'$\alpha = %e \pm %e$' % (cB2[1],np.sqrt(coB2[1][1]))))
	#plt.ylim(-50,50)
	plt.legend()
	plt.title("Paired Lifetime vs. Background, "+pmtStr)
	plt.ylabel("Lifetime")
	plt.xlabel("Short Hold Bkg (Hz.)")
	
	plt.figure(plot_gen)
	plot_gen += 1
	plt.hist2d(bRS,ltS,bins=([lineBR,ltBins]),cmap='summer')
	#H,xedges,yedges = np.histogram2d(bRS,ltS,bins=(lineBR,ltBins))
	#plt.imshow(H,cmap='autumn',interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
	#plt.imshow(np.array([bRS,ltS]),cmap='autumn',interpolation='nearest')
	#sns.heatmap([bRS,ltS],cmap='autumn')
	plt.title("Paired Lifetime vs. Background, "+pmtStr)
	#plt.ylim(-50,50)
	#plt.xlim(0.8,1.2)
	plt.ylabel("Lifetime")
	plt.xlabel("Short Hold Bkg (Hz.)")
	plt.colorbar()
	
	plt.figure(plot_gen)
	plot_gen += 1
	hS = np.zeros(len(lts))
	for i in range(len(lts)):
		hScale = hlds[i][1]-hlds[i][0]
		hS[i] = float(hScale)
		#plt.errorbar(lts[i].val-lifetime.val,bScale.val,xerr=lts[i].err,yerr=bScale.err,fmt='g.')
		#plt.errorbar(lts[i].val-lifetime.val,bScale.val,fmt='g.')
	lineH = np.linspace(1200,1550,50)	
	cH,coH = curve_fit(linear,hS,ltS,sigma=ltE)#p0=(100,0))
	plt.errorbar(hS,ltS,yerr=ltE,fmt='.',color='black',\
			 label=(r'$\tau_0 = %f \pm %f$' % (cH[0],np.sqrt(coH[0][0]))))
	plt.plot(lineH,linear(lineH,*cH),color='gray',\
			 label=(r'$\alpha = %e \pm %e$' % (cH[1],np.sqrt(coH[1][1]))))
	plt.ylim(-50,50)
	plt.xlim(1250,1550)
	plt.legend()
	plt.title("Difference in Hold vs. Lifetime, "+pmtStr)
	plt.ylabel("Lifetime")
	plt.xlabel(r"$\delta t$ (s)")

	plt.figure(plot_gen)
	plot_gen += 1
	plt.hist2d(hS,ltS,bins=([lineH,ltBins]),cmap='gist_yarg')
	#H,xedges,yedges = np.histogram2d(hS,ltS,bins=(lineH,ltBins))
	#plt.imshow(H,cmap='gist_yarg',interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
	#plt.imshow(np.array([hS,ltS]),cmap='gist_yarg',interpolation='nearest')
	#sns.heatmap([hS,ltS],cmap='gist_yarg')
	#plt.ylim(-50,50)
	#plt.xlim(1250,1550)
	plt.title("Difference in Hold vs. Lifetime, "+pmtStr)
	plt.ylabel("Lifetime")
	plt.xlabel(r"$\delta t$ (s)")
	plt.colorbar()
	
	plt.figure(plot_gen)
	plot_gen += 1
	
		#plt.errorbar(lts[i].val-lifetime.val,bScale.val,xerr=lts[i].err,yerr=bScale.err,fmt='g.')
		#plt.errorbar(lts[i].val-lifetime.val,bScale.val,fmt='g.')
	lineN = np.linspace(0.8,1.2,50)	
	cNorm,coNorm = curve_fit(linear,normS-np.mean(normS),ltS,sigma=ltE)#,p0=(10000,0))
	plt.errorbar(normS,ltS,yerr=ltE,fmt='.',color='violet',\
			 label=(r'$\tau_0 = %f \pm %f$' % (cNorm[0],np.sqrt(coNorm[0][0]))))
	plt.plot(lineN,linear(lineN-np.mean(normS),*cNorm),'m',\
			 label=(r'$\alpha = %e \pm %e$' % (cNorm[1],np.sqrt(coNorm[1][1]))))
	plt.ylim(-50,50)
	plt.legend()
	plt.title("Paired Lifetime vs. Expected Counts, "+pmtStr)
	plt.ylabel("Lifetime")
	plt.xlabel("Short Expected Counts / Long Expected Counts")
	
	plt.figure(plot_gen)
	plot_gen += 1
	#sns.heatmap([normS,ltS],cmap='cool')
	#H,xedges,yedges = np.histogram2d(normS,ltS,bins=(lineN,ltBins))
	plt.hist2d(normS,ltS,bins=([lineN,ltBins]),cmap='spring')
	#plt.imshow(H,cmap='cool',interpolation='nearest',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])
	#plt.imshow(np.array([normS,ltS]),cmap='cool',interpolation='nearest')
	plt.ylim(-50,50)
	plt.xlim(0.8,1.2)
	plt.title("Paired Lifetime vs. Expected Counts, "+pmtStr)
	plt.ylabel("Lifetime")
	plt.xlabel("Short Expected Counts / Long Expected Counts")
	plt.colorbar()
	
	#corrected_liftetimes = np.zeros(len(ltS),dtype=measurement)
	corrected_lifetimes = []
	ltS_test = []
	for i in range(len(ltS)):
		ratio_shift = linear(normS[i]-np.mean(normS),*cNorm)
		ratio_unc   = np.sqrt((cNorm[0]*cNorm[0])+(cNorm[1]*(normS[i]-np.mean(normS)))**2 \
							    + 2 * cNorm[0]*cNorm[1]*(normS[i]-np.mean(normS)))
		corrected_lifetimes.append(measurement(ltS[i]+lifetime.val-ratio_shift,np.sqrt(ltE[i]*ltE[i]+ratio_unc*ratio_unc)))
		ltS_test.append(ltS[i]-ratio_shift)
	cB_test,covB_test = curve_fit(linear,normS-np.mean(normS),ltS_test,sigma=ltE)
	plt.figure()
	plt.errorbar(normS,ltS_test,yerr=ltE,fmt='r.',\
			 label=(r'$\tau_0 = %f \pm %f$' % (cB_test[0],np.sqrt(covB_test[0][0]))))
	plt.plot(lineN,linear(lineN-np.mean(normS),*cB_test),'y',\
			 label=(r'$\alpha = %f \pm %f$' % (cB_test[1],np.sqrt(covB_test[1][1]))))
	plt.legend()
	calc_lifetime_paired(corrected_lifetimes,bS)
	

#-----------------------------------------------------------------------
# Plotting some numbers (sorted by Run)
# I had previously written these in dumb ways with like 8 blobbed lists
# and I'm keeping it that way.
# These are in order of when they appeared in the output config file
#-----------------------------------------------------------------------
def plot_lifetime_paired(runPair,ltVec,runBreaks = [],ltAvg=measurement(0.0,np.inf),holdPair = []):

	set_plot_sizing()
	print("Generating plot: Paired Lifetime...")
	plt.figure(69) # Nice. 
	
	holdBuff = [] # Color coding!
	if len(holdPair) == len(ltVec):
		for i in range(0,len(ltVec)):
			holdBuff.append(float(holdPair[i][0])) # Short hold time length
	else:
		for i in range(0,len(ltVec)):
			holdBuff.append(20) # Make everything red as error
	
	for i, lt in enumerate(ltVec):
		if lt.err < 100:
			f = make_errorbar_by_hold(holdBuff[i])
			plt.errorbar(runPair[i][0], lt.val, yerr=lt.err, fmt=f)
			
	_b = make_run_breaks_verts(runBreaks)
			
	plt.title("Paired Lifetime")
	plt.xlabel("Short Run Number")
	plt.ylabel("Lifetime (s)")
	
def plot_lifetime_sections(ltBlobs,ltMeas = measurement(880.0,0.0)):
	
	set_plot_sizing()
	plt.figure(96) # Hardcoded, I might use this one elsewhere
	print("Generating plot: Lifetime Sections...")
	majorSections = [4200,4711,7326,9600,11669,13209] #14517? 
	# ltBlobs should be a list of lists of lifetimes
	# Order should be (Low S12) --> (Hi S12)
	for j in range(len(ltBlobs)): #
		# First get the formatting / label: (see above)
		# TODO: make a "lifetime format" object
		# if j == 0:
			# c = 'green'
			# l = 'Telescoping Coincidence, Low'
		# elif j == 1:
			# c = 'violet'
			# l = 'Telescoping Coincidence, High'
		# elif j == 2:
			# c = 'black'
			# l = 'Fixed Coincidence, Low'			
		# elif j == 3:
			# c = 'red'
			# l = 'Fixed Coincidence, High'
		
		if j == 0:
			c = 'red'
			l = 'Total Singles, Low'
		elif j == 1:
			c = 'black'
			l = 'Total Singles, High'
		elif j == 2:
			c = 'yellow'
			l = 'PMT 1, Low'			
		elif j == 3:
			c = 'blue'
			l = 'PMT 1, High'
		elif j == 4:
			c = 'orange'
			l = 'PMT 2, Low'
		elif j == 5:
			c = 'cyan'
			l = 'PMT 2, High'
		elif j == 6:
			c = 'green'
			l = 'Coincidence, Low'
		elif j == 7:
			c = 'violet'
			l = 'Coincidence, High'
		
		ltBySec = ltBlobs[j] # Now we have the lifetime list by section
		for i,s in enumerate(majorSections):
			if ltBySec[i].val > 1:
				plt.errorbar(s+25*j,ltBySec[i].val - ltMeas.val,xerr=0,yerr=ltBySec[i].err,marker='X',color=c)
		plt.errorbar([],[],marker='X',color=c,label=l)
	plt.title("Lifetime By Major Hardware Section, 500ns Singles Deadtime")
	plt.xlabel("Starting Run")
	plt.ylabel(r'$\tau_{i} - \tau_{L,C}$')
	plt.ylim(-3.5,4.5)
	plt.legend(loc='upper right')

def histogram_lifetime_paired(runPair, ltVec, ltAvg=measurement(0.0,np.inf),holdPair = []):
	
	set_plot_sizing()
	print("Generating plot: Histogram of Paired Lifetime...")
	plt.figure(70)
	ltBuff = []
	ltErr  = []
	for lt in ltVec:
		ltBuff.append(lt.val-totally_sick_blinding_factor) # For blinding
		ltErr.append(lt.err)
			
	holdBuff = []
	if len(holdPair) > 0:
		for i in range(0,len(ltBuff)):
			holdBuff.append(float(holdPair[i][0])) # Short hold time length
	else:
		for i in range(0,len(ltBuff)):
			holdBuff.append(20)
		
	holdTS = [20,50,100,200] # Short hold times	
	histList = [] 
	colList = []
	for t in holdTS:
		histLT, mLT, sLT, col, line= make_histogram_by_hold(holdBuff, ltBuff, t)
		histList.append(histLT)
		colList.append(col)
		
	nBins = int((max(ltBuff) - min(ltBuff)+2) / 3 ) # 3 second bins
	#rangeLT = np.linspace(int(min(ltBuff)-1.0),int(max(ltBuff)+1.0),nBins)
	#plotLT  = np.linspace(int(min(ltBuff)-1.0),int(max(ltBuff)+1.0),nBins*10) 
	#LTFit   = 1/(ltAvg.err*np.sqrt(len(ltVec))*np.sqrt(2*np.pi))*np.exp(-(plotLT - ltAvg.val+totally_sick_blinding_factor)**2/(2*(ltAvg.err*np.sqrt(len(ltVec)))**2))
		
	#histLT,binsLT = np.histogram(ltBuff,bins=rangeLT,density=True)
	#plt.plot(plotLT, LTFit, label="Distribution Fitted to Gaussian")
	#plt.hist(rangeLT[:-1],bins=rangeLT,weights=histLT,density=False,ec="r",facecolor="none")
	plt.hist(histList,nBins,stacked=True,density=True,color=colList)#ec=colList,facecolor="none")
	plt.title("Distribution of paired lifetimes")
	plt.xlabel("Time (blinded) (s)")
	plt.ylabel("Density (arb)")
	
	#chisq = sum(((spectral_norm_meas(np.array([m1,m2]),co[0],co[1]) -np.array(val))/np.array(verr))**2)
	#chisq /= len(runPair)-1
	
	
	#plt.figure(72)
	#for i, lt in enumerate(ltVec):
	#	dev = (lt.val - ltAvg.val) / (ltAvg.err*np.sqrt(len(ltVec)))
	#	avgDis = (runPair[i][0] - runPair[i][1]) / 2.0
	#	plt.errorbar(avgDis,dev,yerr=lt.err/(ltAvg.err*np.sqrt(len(ltVec))),fmt='g.')

#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
def make_legend_multiscan(rRedL):
	set_plot_sizing()
	for r in rRedL:
		if r[0].thresh: # True for High
			if r[0].pmt1 and r[0].pmt2:
				plt.errorbar([],[],marker='.',color='black',label='Total Singles, High')
			elif r[0].pmt1:
				plt.errorbar([],[],marker='.',color='blue',label='PMT 1, High')
			elif r[0].pmt2:
				plt.errorbar([],[],marker='.',color='cyan',label='PMT 2, High')
		else: # False for Low
			if r[0].pmt1 and r[0].pmt2:
				plt.errorbar([],[],marker='.',color='red',label='Total Singles, Low')
			elif r[0].pmt1:
				plt.errorbar([],[],marker='.',color='yellow',label='PMT 1, Low')
			elif r[0].pmt2:
				plt.errorbar([],[],marker='.',color='orange',label='PMT 2, Low')
	return rRedL	
#-----------------------------------------------------------------------
def plot_PMT_comp(rRedL,plot_gen,rB):
	set_plot_sizing()
	print("Generating plot: PMT 1 vs PMT2...")
	plt.figure(plot_gen)
	nPlt = 1
	
	_b = make_run_breaks_verts(rB,True) # For if we're plotting runBreaks
	
	for rRed in rRedL:
		if rRed[0].thresh: # True for High
			if rRed[0].pmt1 and rRed[0].pmt2:
				c = 'black'
			elif rRed[0].pmt1:
				c = 'blue'
			elif rRed[0].pmt2:
				c = 'cyan'
		else: # False for Low
			if rRed[0].pmt1 and rRed[0].pmt2:
				c = 'red'
			elif rRed[0].pmt1:
				c = 'yellow'
			elif rRed[0].pmt2:
				c = 'orange'
		for r in rRed: 
			if 1549 < r.hold < 1551:
				plt.errorbar(r.run,r.cts.val,xerr=0,yerr=r.cts.err,marker='.',color=c)	
	make_legend_multiscan(rRedL)
	plt.legend(loc='upper right')
	plt.title("Singles 1550s Yields")
	plt.xlabel("Run")
	plt.ylabel("Counts")
	return nPlt
#-----------------------------------------------------------------------
def plot_PMT_bkg_comp(rRedL,plot_gen,rB):
	set_plot_sizing()
	print("Generating plot: PMT 1 vs PMT2 Backgrounds...")
	plt.figure(plot_gen)
	nPlt = 1
	
	_b = make_run_breaks_verts(rB,True) # For if we're plotting runBreaks
	
	for rRed in rRedL:
		if rRed[0].thresh: # True for High
			if rRed[0].pmt1 and rRed[0].pmt2:
				c = 'black'
			elif rRed[0].pmt1:
				c = 'blue'
			elif rRed[0].pmt2:
				c = 'cyan'
		else: # False for Low
			if rRed[0].pmt1 and rRed[0].pmt2:
				c = 'red'
			elif rRed[0].pmt1:
				c = 'yellow'
			elif rRed[0].pmt2:
				c = 'orange'
		for r in rRed: 
			plt.errorbar(r.run,r.bkgSum.val,xerr=0,yerr=r.bkgSum.err,marker='.',color=c)	
	make_legend_multiscan(rRedL)
	plt.legend(loc='upper right')
	plt.title("Singles Backgrounds")
	plt.xlabel("Run")
	plt.ylabel("Counts (arb.)")
	
	return nPlt
#-----------------------------------------------------------------------
def plot_PMT_cts_diff(rRedL,plot_gen,rB):
	
	print("Generating plot: Differential PMT Counts...")
	set_plot_sizing()
	plt.figure(plot_gen)
	nPlt = 1
	
	_b = make_run_breaks_verts(rB,True) # For if we're plotting runBreaks
	
	for i in range(0,3): # 2 thresholds
		rRed = [] # Fill a list of differentials
		for j in range(len(rRedL[2*i])): # Recall that evens are low
			rRed.append(rRedL[2*i][j] - rRedL[2*i+1][j]) 	
		if rRed[0].pmt1 and rRed[0].pmt2: # Low PMTs for diffs.
			continue # ignore combined singles
			c = 'green'
			plt.errorbar([],[],marker='.',color=c,label='Differential Singles')
		elif rRed[0].pmt1:
			c = 'chartreuse'
			plt.errorbar([],[],marker='.',color=c,label='Differential PMT 1')
		elif rRed[0].pmt2:
			c = 'goldenrod'
			plt.errorbar([],[],marker='.',color=c,label='Differential PMT 2')
		for r in rRed:
			if 1549 < r.hold < 1551:
				plt.errorbar(r.run,r.cts.val,xerr=0,yerr=0,marker='.',color=c)
				#plt.errorbar(r.run,r.cts.val,xerr=0,yerr=r.cts.err,marker='.',color=c)
	plt.legend(loc='upper right')
	plt.title("Low Threshold - High Threshold 1550s Yields")
	plt.xlabel("Run")
	plt.ylabel("Counts (arb.)")	
	return nPlt
#-----------------------------------------------------------------------
def plot_PMT_bkg_diff(rRedL,plot_gen,rB):

	print("Generating plot: Differential PMT Backgrounds...")
	plt.figure(plot_gen)
	nPlt = 1
	
	_b = make_run_breaks_verts(rB,True) # For if we're plotting runBreaks
	
	for i in range(0,3):
		rRed = [] # Fill a list of differentials
		for j in range(len(rRedL[2*i])): # Recall that evens are low
			rRed.append(rRedL[2*i][j] - rRedL[2*i+1][j]) 	
		if rRed[0].pmt1 and rRed[0].pmt2: # Low PMTs for diffs.
			continue # Ignore total singles for differences
			c = 'green'
			plt.errorbar([],[],marker='.',color=c,label='Differential Singles')
		elif rRed[0].pmt1:
			c = 'chartreuse'
			plt.errorbar([],[],marker='.',color=c,label='Differential PMT 1')
		elif rRed[0].pmt2:
			c = 'goldenrod'
			plt.errorbar([],[],marker='.',color=c,label='Differential PMT 2')
		for r in rRed:
			plt.errorbar(r.run,r.bkgSum.val,xerr=0,yerr=0,marker='.',color=c)	
			#plt.errorbar(r.run,r.bkgSum.val,xerr=0,yerr=r.bkgSum.err,marker='.',color=c)	
	plt.legend(loc='upper right')
	plt.title("Low Threshold - High Threshold Backgrounds")
	plt.xlabel("Run")
	plt.ylabel("Counts (arb.)")
	return nPlt
#-----------------------------------------------------------------------
def plot_PMT_n_cts(rRedL,plot_gen,rB):
	
	print("Generating plot: Differential PMT Normalized Unloads...")
	set_plot_sizing()
	plt.figure(plot_gen)
	nPlt = 1
	
	_b = make_run_breaks_verts(rB,True) # For if we're plotting runBreaks
	
	# 0,1 are singles
	# 2,3 are PMT 1
	# 4,5 are PMT 2
	for i in range(0,3):
		rRed = [] # Fill a list of differentials
		for j in range(len(rRedL[2*i])): # Recall that evens are low
			rRed.append(rRedL[2*i][j] - rRedL[2*i+1][j]) 	
		if rRed[0].pmt1 and rRed[0].pmt2: # Low PMTs for diffs.
			continue # Total singles should just be the sum of the other two
			c = 'green'
			plt.errorbar([],[],marker='.',color=c,label='Differential Singles')
		elif rRed[0].pmt1:
			c = 'chartreuse'
			plt.errorbar([],[],marker='.',color=c,label='Differential PMT 1')
		elif rRed[0].pmt2:
			c = 'goldenrod'
			plt.errorbar([],[],marker='.',color=c,label='Differential PMT 2')
		for r in rRed:
			#plt.errorbar(r.run,r.nCts.val,0,r.nCts.err,marker='.',color=c)
			plt.errorbar(r.run,r.nCts.val / np.exp(-r.hold / 1550), \
						 xerr=0,yerr=0, \
						 marker='.',color=c) # Removing errorbars for now
	plt.legend(loc='upper right')
	plt.title("Low Threshold - High Threshold Normalized Counts Difference")
	plt.xlabel("Run")
	plt.ylabel("Difference")

	plt.figure(plot_gen+nPlt)
	nPlt += 1
	
	_b = make_run_breaks_verts(rB) # For if we're plotting runBreaks
	
	# 0,1 are singles
	# 2,3 are PMT 1
	# 4,5 are PMT 2
	for i in [2,3]:
		rRed = [] # Fill a list of differentials
		for j in range(len(rRedL[i])): # Now splitting PMTs
			rRed.append(rRedL[i][j] - rRedL[i+2][j])
		if rRed[0].thresh: # Low PMTs for diffs.
			c = 'DarkViolet'
			plt.errorbar([],[],marker='.',color=c,label='High Threshold')
		else:
			c = 'Magenta'
			plt.errorbar([],[],marker='.',color=c,label='Low Threshold')
		for r in rRed:
			plt.errorbar(r.run,r.nCts.val / np.exp(-r.hold / 1550),\
						 xerr=0,yerr=0, \
						 marker='.',color=c)	
	plt.legend(loc='upper right')
	plt.title("PMT 1 - PMT 2 Normalized Counts Difference")
	plt.xlabel("Run")
	plt.ylabel("Difference")	

	return nPlt
	
	
def plot_lifetime_paired_by_week(runPair,ltVec):
	set_plot_sizing()
	dt1,dt2 = load_runs_to_datetime(runPair,"/home/frank/run_and_start.txt")
	
	print("Writing out: Lifetime_Paired_Times.csv")
	runsOut = open("Lifetime_Paired_Times.csv", "w")
	runsOut.write("R_S,t_s,R_L,t_l,lt_val,lt_err\n")
	for i,lt in enumerate(ltVec):
		if 4374 <= runPair[i][0] < 7326:
			lt += measurement(0.3,0.2)
		runsOut.write("%05d,%d,%05d,%d,%f,%f" % (runPair[i][0],int(dt1[i].timestamp()),runPair[i][1],int(dt2[i].timestamp()),lt.val,lt.err))
	runsOut.close()
	
	
	nlt = []
	wks = range(0,54)
	lts = []
	for w in wks:
		lts.append([])
	for i,d in enumerate(dt1):
		wk = int(d.strftime('%W')) # Monday as first day of week
		lts[wk].append(float(ltVec[i]))
	ltV = []
	ltE = []
	for w in wks:
		if len(lts[w]) > 0:
			ltV.append(np.mean(lts[w]))
			ltE.append(np.std(lts[w])/np.sqrt(len(lts[w])))
		else:
			ltV.append(np.inf) # Turns out np.inf won't get plotted!
			ltE.append(np.inf)
	plt.figure(96)
	plt.errorbar(wks,ltV,yerr=ltE,fmt='r.')
	plt.title("Paired Lifetime, checking for annual variations")
	plt.xlabel("Week of the year")
	plt.ylabel("Lifetime (s)")
	
	
#-----------------------------------------------------------------------	
	
# def plot_PMT_yields(runNum, holdVec, pmt1V, pmt2V, ctsVec,nPlt = 1, runBreaks = [], coincLT = False, pmt1=True, pmt2=True):
			
	# if coincLT:
		# print("Generating plot: Total Coincidence Yield by Run...")
	# else:
		# print("Generating plot: Total Singles Yield by Run...")
	
	# plt.figure(nPlt)
	# nPlt +=1
	
	# _b = make_run_breaks_verts(runBreaks)
	
	# for i, r in enumerate(runNum):
		# if float(holdVec[i]) <= 2999.0:
			# plt.plot(r,pmt1V[i],'r.')
			# plt.plot(r,pmt2V[i],'b.')
			# plt.plot(r,ctsVec[i],'k.')
	# plt.plot([],[],'r.',label='PMT 1')
	# plt.plot([],[],'b.',label='PMT 2')
	# plt.plot([],[],'k.',label='Both')
	# plt.legend()
	
	# if coincLT: # Title and axis labels from bools
		# plt.title("Total Coincidence Yield by Run")
	# else:
		# if pmt1 and pmt2:
			# #plt.title("Total Singles Yield by Run")
			# plt.title("Ratio of High and Low Threshold Yields by Run")
		# elif pmt1:
			# plt.title("Total PMT1 Yield by Run")
		# elif pmt2:
			# plt.title("Total PMT2 Yield by Run")
	# plt.xlabel("Run Number")
	# #plt.ylabel("Dagger Counts During Unload")
	# plt.ylabel("High Threshold Yield / Low Threshold Yield")
			
	# return nPlt


# 
# #-----------------------------------------------------------------------# #-----------------------------------------------------------------------
# #-----------------------------------------------------------------------		
# #-----------------------------------------------------------------------
	

# #-----------------------------------------------------------------------
# # Plotting histograms of stuff
# #-----------------------------------------------------------------------
# 	
	# # pdfBuff makes a Gaussian out of the data -- might be useful to have a make_gaussian_from_hist (and a calc_chi2_from_hist)
	# #pdfbuff20 = np.exp(-((range20)**2) /(2*stat20[1]**2)) #stats.norm.pdf(range20,stat20[0ddd],stat20[1])
	# #sum20 = np.sum(pdfbuff20)
	# #for i, cts in enumerate(pdfbuff20):
	# #	pdfbuff20[i] = cts/sum20
		
	# #plt.plot(range20,pdfbuff20, 'r-')
	
		
		

	
	# # histBuff20 = []
	# # histBuff50 = []
	# # histBuff100 = []
	# # histBuff200 = []
	# # histBuff1550 = []
	# # for i, cts in enumerate(nCtsVec):
		# # if 19.0 < float(holdVec[i]) < 21.0:
			# # histBuff20.append(nCtsVec[i])
		# # elif 49.0 < float(holdVec[i]) < 51.0:
			# # histBuff50.append(nCtsVec[i])
		# # elif 99.0 < float(holdVec[i]) < 101.0:
			# # histBuff100.append(nCtsVec[i])
		# # elif 199.0 < float(holdVec[i]) < 201.0:
			# # histBuff200.append(nCtsVec[i])
		# # elif 1549.0 < float(holdVec[i]) < 1551.0:
			# # histBuff0.append(nCtsVec[i])
	
	# # # Calc. averages and std.
	# # stat20   = [np.average(histBuff20),   np.std(histBuff20)]
	# # stat50   = [np.average(histBuff50),   np.std(histBuff50)]
	# # stat100  = [np.average(histBuff100),  np.std(histBuff100)]
	# # stat200  = [np.average(histBuff200),  np.std(histBuff200)]
	# # stat1550 = [np.average(histBuff1550), np.std(histBuff1550)]
	
	# # # Shift values
	# # histBuff20   = histBuff20 - stat20[0]
	# # histBuff50   = histBuff50 - stat50[0]
	# # histBuff100  = histBuff100 - stat100[0]
	# # histBuff200  = histBuff200 - stat200[0]
	# # histBuff1550 = histBuff1550 - stat1550[0]
	
	# # # Calculate and plot 20s
	# # #plt.subplot(2,1,2)
	# # # range20 = np.linspace(stat20[0]-3*stat20[1],stat20[0]+3*stat20[1],100)
	# # range20 = np.linspace(-3*stat20[1],+3*stat20[1],20)
	# # hist20, bins20 = np.histogram(histBuff20, bins=range20,density=True)
	# # sum20 = np.sum(hist20)
	# # #print sum20
	# # for i, cts in enumerate(hist20):
		# # hist20[i] = cts/sum20
	# # plt.hist(range20[:-1], bins=range20,weights=hist20,density=False,ec="r",facecolor="none",label=("20s (%0.04f +/- %0.04f)" % (stat20[0], stat20[1])))
	
	# # #pdfbuff20 = np.exp(-((range20-stat20[0])**2) /(2*stat20[1]**2))#stats.norm.pdf(range20,stat20[0],stat20[1])
	# # #range20 = np.linspace(-3*stat20[1],+3*stat20[1],100)

	
	# # range1550 = np.linspace(-3*stat1550[1],+3*stat1550[1],20)
	# # hist1550, bins1550 = np.histogram(histBuff1550, bins=range20,density=True)
	# # sum1550 = np.sum(hist1550)
	
	# # for i, cts in enumerate(hist1550):
		# # hist1550[i] = cts/sum1550
	# # plt.hist(range20[:-1], bins=range20,weights=hist1550,density=False,ec="c",facecolor="none",label=("1550s (%0.04f +/- %0.04f)" % (stat1550[0], stat1550[1])))
	
	# # #pdfbuff20 = np.exp(-((range20-stat20[0])**2) /(2*stat20[1]**2))#stats.norm.pdf(range20,stat20[0],stat20[1])
	# # #range1550 = np.linspace(-3*stat1550[1],+3*stat1550[1],100)
	# # pdfbuff1550 = np.exp(-((range20)**2) /(2*stat1550[1]**2)) #stats.norm.pdf(range20,stat20[0ddd],stat20[1])
	# # sum1550 = np.sum(pdfbuff1550)
	# # for i, cts in enumerate(pdfbuff1550):
		# # pdfbuff1550[i] = cts/sum1550
		
	# # plt.plot(range20,pdfbuff1550, 'c-')		
	# # Calculate and plot 1550s
	# #plt.subplot(2,1,1)
	# # # range1550 = np.linspace(stat1550[0]-3*stat1550[1],stat1550[0]+3*stat1550[1],100)
	# # range1550 = np.linspace(-3*stat1550[1],+3*stat1550[1],100)
	# # #hist1550, bins1550 = np.histogram(histBuff1550, bins=20,density=True)
	# # hist1550 = plt.hist(histBuff1550, bins=20, ec="c",density=True,facecolor="none",label=("1550s (%0.04f +/- %0.04f)" % (stat1550[0], stat1550[1])))
	# # #pdfbuff1550 = np.exp(-((range1550-stat1550[0])**2) /(2*stat1550[1]**2))
	# # pdfbuff1550 = 1/len(histBuff1550) * np.exp(-((range1550)**2) /(2*stat1550[1]**2))
	# # #pdfbuff1550 = stats.norm.pdf(range1550,stat1550[0],stat1550[1])
	# # plt.plot(range1550,pdfbuff1550,'c-')
	# # Plot
	# #plt.hist(histBuff20, bins=20, ec="r",normed=1,facecolor="none",label=("20s (%0.04f +/- %0.04f)" % (stat20[0], stat20[1])))
	# #plt.hist(histBuff50, bins=20, ec="y",normed=1,facecolor="none",label=("50s (%0.04f +/- %0.04f)" % (stat50[0], stat50[1])))
	# #plt.hist(histBuff100, bins=20, ec="g",normed=1,facecolor="none",label=("100s (%0.04f +/- %0.04f)" % (stat100[0], stat100[1])))
	# #plt.hist(histBuff200, bins=20, ec="b",normed=1,facecolor="none",label=("200s (%0.04f +/- %0.04f)" % (stat200[0], stat200[1])))
	# #plt.hist(histBuff1550, bins=20, ec="c",normed=1,facecolor="none",label=("1550s (%0.04f +/- %0.04f)" % (stat1550[0], stat1550[1])))
			

# #-----------------------------------------------------------------------


# #-----------------------------------------------------------------------
# def plot_run_breaks(rL,mR1, mR2,ratioList,rB = [], nDet1 = 3,nDet2 = 5,pctBreak=0.8,nPlt = 1, as1 = [], as2 = []):
	
	# print("Generating plots: Run Breaks Raw Counts")
	
	# plt.figure(nPlt)
	# plt.plot(rL,mR1,'b.', label=("Detector %d" % nDet1))
	# plt.plot(rL,mR2,'r.', label=("Detector %d" % nDet2))
	# plt.xlabel("Run Number")
	# plt.ylabel("Raw Counts of Loading Monitors")
	
	# if len(rB) > 0:
		# for xl in rB:
			# plt.axvline(x=xl)
	
	# nPlt+=1
	# mR = []
	# for i, m in enumerate(mR1):
		# if mR2[i] > 0:
			# mR.append(m / mR2[i])
		# else:
			# mR.append(np.inf)
		
	# plt.figure(nPlt)
	# plt.plot(rL,mR,'b.')
	# plt.xlabel("Run Number")
	# plt.ylabel("Ratio of (detector %d) / (detector %d)" % (nDet1, nDet2))
	
	# if len(rB) > 0:
		# for xl in rB:
			# plt.axvline(x=xl)
	
	# nPlt+=1
	
	# plt.figure(nPlt)
	# plt.plot(rL,ratioList,'r.')
	# plt.xlabel("Run Number")
	# plt.ylabel("Ratio of sequential runs")
	# plt.hlines(pctBreak,min(rL),max(rL))
	# plt.hlines(1.0/pctBreak,min(rL),max(rL))
	# for xl in rB:
		# plt.axvline(x=xl)
	# nPlt += 1
	
	# # plt.figure(nPlt)
	# # plt.plot(rL,as1,'r.')
	# # plt.title("Asymmetry List 1")
	
	# # for xl in rB:
		# # plt.axvline(x=xl)
	# # nPlt +=1
	
	# # plt.figure(nPlt)
	# # plt.plot(rL,as2,'b.')
	# # plt.title("asymmetry list 2")
	
	# # for xl in rB:
		# # plt.axvline(x=xl)
	# # nPlt +=1
	
	# # #plt.show()
	# return nPlt
# #-----------------------------------------------------------------------
	
# #-----------------------------------------------------------------------
# 
# #-----------------------------------------------------------------------


# #-----------------------------------------------------------------------	
# def plot_normalization_counts(runNum,normVec1, normVec2, nPlt = 1):
	
	# print("Generating plot: Real Normalization Factor by Run...")
	# plt.figure(nPlt)
	# nPlt +=1
	# tmpNorm = measurement(0.0,0.0)
	# for i, cts in enumerate(normVec1):
		# tmpNorm = normVec1[i] + normVec2[i]
		# plt.errorbar(runNum[i], tmpNorm.val,yerr=tmpNorm.err,fmt='r.')
	# plt.title("Expected Yield From Normalized Monitor Rates")
	# plt.xlabel("Run Number")
	# plt.ylabel("Normalized Expected Yield")
	
	# print("Generating plot: Normalization Scatter Plot...")
	# plt.figure(nPlt)
	# nPlt +=1
	
	# # I'm hard coding this because fuck it
	# nv2017_1 = []
	# nv2017_2 = []
	# nv2018_1 = []
	# nv2018_2 = []
	# nv2017E_1 = []
	# nv2017E_2 = []
	# nv2018E_1 = []
	# nv2018E_2 = []
	# for i, cts in enumerate(normVec1): #x, y scatter plot
		
		# if (runNum[i] < 9600):
			# nv2017_1.append(normVec1[i].val)
			# nv2017_2.append(normVec2[i].val)
			# nv2017E_1.append(normVec1[i].err)
			# nv2017E_2.append(normVec2[i].err)
			# # plt.errorbar(normVec1[i].val,normVec2[i].val,xerr=normVec1[i].err,yerr=normVec2[i].err,fmt='b.',label='2017 runs')
		# #elif (9600 <= runNum[i] < 14596):
		# elif (9600 <= runNum[i] < 19999):
			# nv2018_1.append(normVec1[i].val)
			# nv2018_2.append(normVec2[i].val)
			# nv2018E_1.append(normVec1[i].err)
			# nv2018E_2.append(normVec2[i].err)
			# #plt.errorbar(normVec1[i].val,normVec2[i].val,xerr=normVec1[i].err,yerr=normVec2[i].err,fmt='r.',label='2018 runs')
	
	# #Rescale Errors
	# nv2017E_1 = nv2017E_1 / np.std(nv2017_1)
	# nv2018E_1 = nv2018E_1 / np.std(nv2018_1)
	# nv2017E_2 = nv2017E_2 / np.std(nv2017_2)
	# nv2018E_2 = nv2018E_2 / np.std(nv2018_2)
		
	# # Rescale values
	# nv2017_1 = (nv2017_1 - np.mean(nv2017_1)) / np.std(nv2017_1)
	# nv2018_1 = (nv2018_1 - np.mean(nv2018_1)) / np.std(nv2018_1)
	# nv2017_2 = (nv2017_2 - np.mean(nv2017_2)) / np.std(nv2017_2)
	# nv2018_2 = (nv2018_2 - np.mean(nv2018_2)) / np.std(nv2018_2)
		
	# plt.errorbar(nv2017_1,nv2017_2,xerr=nv2017E_1,yerr=nv2017E_2,fmt='b.',label='No Roundhouse')
	# plt.errorbar(nv2018_1,nv2018_2,xerr=nv2018E_1,yerr=nv2018E_2,fmt='r.',label='With Roundhouse')
		
	# plt.title("Normalization Monitor Counts")
	# plt.legend()
	# plt.xlim([-2.5,2.5])
	# plt.ylim([-3.5,3.5])
	# plt.xlabel("Normalized Low Detector (arb.)")
	# plt.ylabel("Normalized High Detector (arb.)")
	# return nPlt

# def plot_background_by_PMT(runNum,holdVec,bSubVec,bkgSub1 = [],bkgSub2 = [],nPlt = 1, runBreaks = []):
	
	# print("Generating plot: Singles background by PMT...")
			
	# if len(bkgSub1) == 0 and len(bkgSub2) == 0:
		# nPlt = plot_background_subtracted(runNum,holdVec,bSubVec,nPlt,runBreaks,False)
		# return nPlt
		
	# plt.figure(nPlt)
	# nPlt += 1
	
	# if len(runNum) != len(bSubVec):
		# return nPlt
	# for i,r in enumerate(runNum):
		# if float(holdVec[i]) >= 2900:
			# continue
		# plt.plot(r, float(bSubVec[i]),'k.')
		# if len(bkgSub1) > 0:
			# plt.plot(r,float(bkgSub1[i]),'r.')
		# if len(bkgSub2) > 0:
			# plt.plot(r,float(bkgSub2[i]),'b.')
	
	# # Labels:
	# plt.plot([],[],'k.',label="Singles")
	# plt.plot([],[],'r.',label="PMT 1")
	# plt.plot([],[],'b.',label="PMT 2")
	# plt.legend()
	# plt.xlabel("Run Number")
	# plt.ylabel("Photons")
	# #plt.ylabel("High Thresh. / Low Thresh. Photons")
	# plt.title("Average Photons in Coincidence Events (Hi Thresh)")
	
	# _b = make_run_breaks_verts(runBreaks)
	
	# plt.figure(nPlt)
	# nPlt += 1
	
	# for i,r in enumerate(runNum):
		# if float(holdVec[i]) >= 2900:
			# continue
		# if len(bkgSub1) > 0:
			# plt.plot(r,float(bkgSub1[i]/bSubVec[i]),'r.')
		# if len(bkgSub2) > 0:
			# plt.plot(r,float(bkgSub2[i]/bSubVec[i]),'b.')
	
	# # Labels:
	# plt.plot([],[],'rx',label="PMT 1")
	# plt.plot([],[],'bx',label="PMT 2")
	# plt.legend()
	# plt.xlabel("Run Number")
	# plt.ylabel("Fraction")
	# plt.title("Fraction of Coincidence Photons in Each PMT (Hi Thresh)")
			
	# return nPlt
# 
# #-----------------------------------------------------------------------
# 
# 
# #-----------------------------------------------------------------------	

# def plot_expected(runVec,holdVec,nCtsVec,nCorr,nPlt=[]):
	
	# print("Plotting expected values...")
	# plt.figure(nPlt)
	# nPlt += 1
	
	# # This one is retroactivcely scaled
	
	# for i, r in enumerate(runVec):
		# _h = make_errorbar_by_hold(float(holdVec[i]),r,(
				# float(nCorr[i]) - float(nCtsVec[i])*np.exp((float(holdVec[i])-20.)/887) / float(nCorr[i]) ))
		
	# plt.title("Difference Between Actual and Expected Values")
	# plt.xlabel("Run Number")
	# plt.ylabel("Difference (Std. Devs Away From Mean)")
	# #plt.show()
	# return nPlt
	
	
# def plot_expected_values(runVec,rawCtsVec,nCorr,holdVec,runBreaks = [], nPlt=1,normT=[],nSig = 3.0):
		
	# print("Plotting expected values...")
	
	# plt.figure(nPlt)
	# nPlt += 1
	# if len(normT) != len(holdVec):
		# print(len(normT),len(holdVec))
		# normT = np.zeros(len(holdVec))
		# for i,t in enumerate(holdVec):
			# normT[i] = round(t)
	
	# timeByT = np.unique(normT)
	# meanByT = np.zeros(len(timeByT))
	# stdByT  = np.zeros(len(timeByT))
	# meanAll = 0
	# stdAll  = 0
	# ratioAll  = []
	# for i,t in enumerate(timeByT):
		# ratio = []
		# for j, t in enumerate(holdVec):
			# if float(normT[j]) - 1.0 < float(holdVec[j]) < float(normT[j]) + 1.0:
				# ratio.append(float(rawCtsVec[j]/nCorr[j]))
				# ratioAll.append(float(rawCtsVec[j]/nCorr[j]))
		# meanByT[i] = np.mean(ratio)
		# stdByT[i]  = np.std(ratio)
	
	# meanAll = np.mean(ratioAll)
	# stdAll = np.std(ratioAll)
	# runFilters = []
	# for i, r in enumerate(runVec):
		# if float(normT[i]) - 1.0 < float(holdVec[i]) < float(normT[i]) + 1.0:
			# ind = int(np.argwhere(timeByT == round(normT[i])))			
			# #_h = make_errorbar_by_hold(float(holdVec[i]),r,(float(rawCtsVec[i]/nCorr[i]) - meanByT[ind])/stdByT[ind])
			# _h = make_errorbar_by_hold(float(holdVec[i]),r,(float(rawCtsVec[i]/nCorr[i]) - meanAll)/stdAll)
			# if -3.0 <  (float(rawCtsVec[i]/nCorr[i]) - meanByT[ind])/stdByT[ind] < 3.0:
				# runFilters.append(r)
	# print(len(runFilters))
	# _b = make_run_breaks_verts(runBreaks)
	# horizon = np.linspace(min(runVec),max(runVec),1000)
	# horizon_min = -3.0 * np.ones(1000)
	# horizon_max = 3.0 * np.ones(1000)
	# plt.plot(horizon,horizon_min,'g-')
	# plt.plot(horizon,horizon_max,'g-')
	
	# plt.title("Difference Between Actual and Expected Values")
	# plt.xlabel("Run Number")
	# plt.ylabel("Difference (Std. Devs Away From Mean)")
	# plt.show()
	# return nPlt,runFilters
	

# #-----------------------------------------------------------------------
# 
	
			
# #-----------------------------------------------------------------------
# # Lifetime Plots
# #-----------------------------------------------------------------------
# def plot_lifetime_exponential(runNum,nCtsVec,times,lt = measurement(877.7,0.0), scale= measurement(1.0,0.0),pltAll = True, p2017 = True, p2018 = True):
		
	# print("Generating plot: Exponential Lifetime...")
	# plt.figure(420) # Nice. 

	# if pltAll: # Plot every value in the lifetime exponential
		# for i, cts in enumerate(nCtsVec): # loop through the cts 
			# try:
				# tVal = times[i].val
				# tErr = times[i].err
			# except AttributeError:
				# tVal = times[i].val
				# tErr = 0.0
			# if (runNum[i] < 9600): # 2017				
				# plt.errorbar(tVal,cts.val,xerr=tErr,yerr=cts.err, fmt='b.')
			# elif (9600 <= runNum[i] < 14516): # 2018
				# plt.errorbar(tVal,cts.val,xerr=tErr,yerr=cts.err, fmt='r.')
	# else:
		
		# ax1 = plt.subplot(3,1,(1,2)) # Initializing runs for times
		# if min(runNum) < 9960: # have 2017 data
			# avg2017  = []
			# avgT2017 = []
			# cnt2017  = []
			# for i in range(0,9): # [20, 50, 100, 200, 1300, 1550, 3000, 4000, 5000] are the runs used in 2017-2018 at some point 
				# avg2017.append(measurement(0.0,0.0))
				# avgT2017.append(measurement(0.0,0.0))
				# cnt2017.append(0)
		# if 9600 <= max(runNum): # have 2018 data
			# avg2018  = []
			# avgT2018 = []
			# cnt2018  = []
			# for i in range(0,9):
				# avg2018.append(measurement(0.0,0.0))
				# avgT2018.append(measurement(0.0,0.0))
				# cnt2018.append(0)

		# for i, cts in enumerate(nCtsVec):
		
			# if not useMeanArr:
				# try:
					# len(meanArr)
				# except NameError:
					# meanArr = []
										
				# if not len(meanArr) == len(nCtsVec):
					# for i in range(0,len(nCtsVec)):
						# meanArr.append(measurement(0.0,0.0))				
				# meanArr[i] = measurement(holdVec[i],0.0)
			# if use2017:
				# if (runNum[i] < 9600): # 2017				
					# if 19.0 < holdVec[i] < 21.0:
						# avg2017[0]  += nCtsVec[i]
						# avgT2017[0] += meanArr[i] 
						# cnt2017[0]  += 1
					# elif 49.0 < holdVec[i] < 51.0:
						# avg2017[1]  += nCtsVec[i]
						# avgT2017[1] += meanArr[i] 
						# cnt2017[1]  += 1
					# elif 99.0 < holdVec[i] < 101.0:
						# avg2017[2]  += nCtsVec[i]
						# avgT2017[2] += meanArr[i] 
						# cnt2017[2]  += 1
					# elif 199.0 < holdVec[i] < 201.0:
						# avg2017[3]  += nCtsVec[i]
						# avgT2017[3] += meanArr[i] 
						# cnt2017[3]  += 1
					# elif 1549.0 < holdVec[i] < 1551.0:
						# avg2017[4]  += nCtsVec[i]
						# avgT2017[4] += meanArr[i] 
						# cnt2017[4]  += 1
					# elif 2999.0 < holdVec[i] < 3001.0:
						# avg2017[5]  += nCtsVec[i]
						# avgT2017[5] += meanArr[i] 
						# cnt2017[5]  += 1
					# elif 3999.0 < holdVec[i] < 4001.0:
						# avg2017[6]  += nCtsVec[i]
						# avgT2017[6] += meanArr[i] 
						# cnt2017[6]  += 1
					# elif 4999.0 < holdVec[i] < 5001.0:
						# avg2017[7]  += nCtsVec[i]
						# avgT2017[7] += meanArr[i] 
						# cnt2017[7]  += 1
			# if use2018:	
				# if (9600 <= runNum[i] < 14516): # 2018
					# if 19.0 < holdVec[i] < 21.0:
						# avg2018[0]  += nCtsVec[i]
						# avgT2018[0] += meanArr[i] 
						# cnt2018[0]  += 1
					# elif 49.0 < holdVec[i] < 51.0:
						# avg2018[1]  += nCtsVec[i]
						# avgT2018[1] += meanArr[i] 
						# cnt2018[1]  += 1
					# elif 99.0 < holdVec[i] < 101.0:
						# avg2018[2]  += nCtsVec[i]
						# avgT2018[2] += meanArr[i] 
						# cnt2018[2]  += 1
					# elif 199.0 < holdVec[i] < 201.0:
						# avg2018[3]  += nCtsVec[i]
						# avgT2018[3] += meanArr[i] 
						# cnt2018[3]  += 1
					# elif 1549.0 < holdVec[i] < 1551.0:
						# avg2018[4]  += nCtsVec[i]
						# avgT2018[4] += meanArr[i] 
						# cnt2018[4]  += 1
					# elif 2999.0 < holdVec[i] < 3001.0:
						# avg2018[5]  += nCtsVec[i]
						# avgT2018[5] += meanArr[i] 
						# cnt2018[5]  += 1
					# elif 3999.0 < holdVec[i] < 4001.0:
						# avg2018[6]  += nCtsVec[i]
						# avgT2018[6] += meanArr[i] 
						# cnt2018[6]  += 1
					# elif 4999.0 < holdVec[i] < 5001.0:
						# avg2018[7]  += nCtsVec[i]
						# avgT2018[7] += meanArr[i] 
						# cnt2018[7]  += 1
		
		# if use2017:
			# res2017 = []
			# for i, cts in enumerate(avg2017):
				
				# if cnt2017[i] != 0:
					# avg2017[i] = measurement(cts.val / cnt2017[i], cts.err / np.sqrt(cnt2017[i]))
					# avgT2017[i] = measurement(avgT2017[i].val / cnt2017[i], avgT2017[i].err / np.sqrt(cnt2017[i]))
					
				# plt.errorbar(avgT2017[i].val,avg2017[i].val,xerr=avgT2017[i].err,yerr=avg2017[i].err,fmt='b.',label='2017 Data')
				# res2017.append(measurement((avg2017[i].val - explt(avgT2017[i].val,*pFitE)) / avg2017[i].err, 0.0))
				# print (res2017[i])
		# if use2018:
			# res2018 = []
			# for i, cts in enumerate(avg2018):
				# if cnt2018[i] > 0:
					# avg2018[i] = measurement(cts.val / cnt2018[i], cts.err / np.sqrt(cnt2018[i]))
					# avgT2018[i] = measurement(avgT2018[i].val / cnt2018[i], avgT2018[i].err / np.sqrt(cnt2018[i]))
					
				# plt.errorbar(avgT2018[i].val,avg2018[i].val,xerr=avgT2018[i].err,yerr=avg2018[i].err,fmt='r.',label='2018 Data')
				# res2018.append(measurement((avg2018[i].val - explt(avgT2018[i].val,*pFitE)) / avg2018[i].err, 0.0))
				# print (res2018[i])
				
	
	# plt.legend()
	
	# # Exponential fitting curve
	# eTPlot = np.linspace(20.0, 5500.0, 5000)
	# eCPlot = explt(eTPlot,scale.val,lt.val)
	# plt.plot(eTPlot,eCPlot)
	
	# plt.title("Normalized Run Lifetime")
	# plt.yscale('log')
	# plt.ylabel("Normalized Yield (arb.)")
	
	# if not pltAll:
		# plt.subplot(313, sharex=ax1)
		# if use2017:
			# for i, cts in enumerate(res2017):
				# if abs(cts.val) > 0.5:
					# continue
				# plt.errorbar(avgT2017[i].val, res2017[i].val, xerr=avgT2017[i].err, yerr=res2017[i].err,fmt='b.')
		# if use2018:
			# for i, cts in enumerate(res2018):
				# if abs(cts.val) > 0.5:
					# continue
				# plt.errorbar(avgT2018[i].val, res2018[i].val, xerr=avgT2018[i].err, yerr=res2018[i].err,fmt='r.')		
		# plt.ylabel("Residual (std dev.)")
		
	# plt.xlabel("Holding Time (s)")
# #--------------------------------------------------------------------------------------------------------------
	
# #--------------------------------------------------------------------------------------------------------------

# #-----------------------------------------------------------------------
# # Corner-type plots --- Should I move these to likelihood_lifetime?
# #-----------------------------------------------------------------------
# def color_grad(x):
	# #helper to choose a color
	# #based upon an index between 0 and 1
	# sx = 1-x**1.0
	# return (
			# 0.900*sx+0.000*(1-sx),
			# 0.100*sx+0.100*(1-sx),
			# 0.000*sx+0.900*(1-sx),
			# 1.0
			# )

# def plot_global_normalization(input_data,samples = [], nPlt = 1):
	
	# #load data from potentially multiple files, combining them all
	# if samples == []:
		# samples = loadtxt("./test_out.csv")

	# #print samples
	# #take means of posterior distributions as a "best fit"
	# tau_0  = np.average(samples[:,0])
	# N_0    = np.average(samples[:,1])
	# beta_0 = np.average(samples[:,2])

	# #get the full range of chi2 for plotting purposes later
	# print("Global Lifetime Fit is: "+str(tau_0))
	# print (str(N_0)+str(beta_0))
	# #chi2_min = amin(blobs[:])
	# #chi2_max = amin(blobs[:])
	# chi2_min = np.amin(samples[:,3])
	# chi2_max = np.amax(samples[:,3])
	# chi2_delta = chi2_max-chi2_min

    # #corner plot all 3 parameters
	# corner.corner(samples,
					# labels=['$\\tau$','$N$','$\\beta$','$\\chi^{2}$'],
					# quantiles=[0.16,0.5,0.84],
					# show_titles=True,
					# bins=32
				# )

	# #Load the input data used for the MCMC fit -- for this is just incorporated here
	# #input_data = np.genfromtxt(argv[1],delimiter=',')
	
	# plt.figure(2) # Residual plot
	# dTs = np.arange(0.0,1600.0) # Time goes from 0 - 1550s (ignore extra long runs)
	# for s in samples: # Plot each MCMC result 
		# plt.plot(dTs,(s[1]*np.exp(-dTs/s[0])-N_0*np.exp(-dTs/tau_0))/(N_0*np.exp(-dTs/tau_0)), # Plot
				# color = color_grad((s[3]-chi2_min)/chi2_delta),linewidth=0.3)
	# plt.plot(dTs,dTs-dTs,color='k',linestyle=':',linewidth=2) # Plot a dotted line across y=zero
	# plt.plot(input_data[:,1],(input_data[:,2]/(input_data[:,3]+beta_0*input_data[:,5]) \
		# -N_0*np.exp(-input_data[:,1]/tau_0))/(N_0*np.exp(-input_data[:,1]/tau_0)),color='k',linestyle='None',marker='.',markersize=6)
	# plt.xlim([0.0,1600.0])
	# plt.ylim([-0.03,0.03])
	# plt.grid()
	# plt.xlabel('$t$ [s]',fontsize='xx-large')
	# plt.ylabel('global fit residuals',fontsize='xx-large')

	# D_20s    = np.array([x[2] for x in input_data if 19<x[1]<21])
	# mon1_20s = np.array([x[3] for x in input_data if 19<x[1]<21])
	# mon2_20s = np.array([x[5] for x in input_data if 19<x[1]<21])

	# #Plot the variance of the 20 second holds versus beta to compare to the best fit beta above
	# plt.figure(3)
	# betas = np.arange(-0.1,0.0,0.005)
	# normed = map(lambda b : np.std(D_20s/(mon1_20s+b*mon2_20s))/np.average(D_20s/(mon1_20s+b*mon2_20s)),betas)
	# plt.plot(betas,normed,c='orange',lw=3)
	# plt.grid()
	# plt.xlabel('$\\beta$',fontsize='xx-large')
	# plt.ylabel('normalization factors',fontsize='xx-large')
	# plt.show()
	
	# return nPlt
# #-----------------------------------------------------------------------

# #-----------------------------------------------------------------------
# # Re-make some of these things if you want...
# #-----------------------------------------------------------------------
# #def make_gaussian_from_hist():
# #	
# #	
# #	pdfbuff20 = np.exp(-((range20)**2) /(2*stat20[1]**2)) #stats.norm.pdf(range20,stat20[0ddd],stat20[1])
# #	sum20 = np.sum(pdfbuff20)
# #	for i, cts in enumerate(pdfbuff20):
# #		pdfbuff20[i] = cts/sum20
# #		
# #	plt.plot(range20,pdfbuff20, 'r-')
