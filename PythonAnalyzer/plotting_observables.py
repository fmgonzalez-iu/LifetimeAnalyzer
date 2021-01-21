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
from PythonAnalyzer.classes import measurement, reduced_run
from PythonAnalyzer.dataIO import load_runs_to_datetime
from PythonAnalyzer.pairingandlifetimes import calc_lifetime_paired

# For plotting of general observables
def plot_raw_yields(rRed,plot_gen,rB, vb=True):
	if vb: # I could go back and make this an out.vb setting, but eh
		if rRed[0].sing:
			if rRed[0].pmt1 and rRed[0].pmt2:
				print("Generating plot: Total Singles Yield by Run...")
			elif rRed[0].pmt1:
				print("Generating plot: Total PMT 1 Yield by Run...")
			elif rRed[0].pmt2:	
				print("Generating plot: Total PMT 2 Yield by Run...")
		else:
			print("Generating plot: Total Coincidence Yield by Run...")

	set_plot_sizing()
	plt.figure(plot_gen)
	nPlt = 1 # counting buffer
		
	_b = make_run_breaks_verts(rB)
	holdVec = np.zeros(len(rRed)) # Need to initialize holding time vector
	for i,r in enumerate(rRed): # Loop through our runs to make plots
		holdVec[i] = r.hold
		_f = make_errorbar_by_hold(r.hold,r.run,float(r.cts))
	_h = make_legend_by_hold(holdVec) # Formatting labels
		
	if rRed[0].sing: # Generate Title
		if rRed[0].pmt1 and rRed[0].pmt2:
			plt.title("Total Singles Yield by Run")
		elif rRed[0].pmt1:
			plt.title("Total PMT 1 Yield by Run")
		elif rRed[0].pmt2:	
			plt.title("Total PMT 2 Yield by Run")
	else:
		plt.title("Total Coincidence Yield by Run")
	plt.grid(True)
	plt.xlabel("Run Number") # And axis lables
	plt.ylabel("Dagger Counts During Unload")				
	plt.yscale('log')
	return nPlt
#-----------------------------------------------------------------------	
def plot_normalized_counts(rRed,plot_gen,rB):
	
	print("Generating plot: Normalized Yield by Run...")
	set_plot_sizing()
	plt.figure(plot_gen)
	nPlt = 1
	_b = make_run_breaks_verts(rB)
	holdVec = np.zeros(len(rRed))
	norm20 = [] # Check that 20s normalization is reasonable
	for i,r in enumerate(rRed):
		holdVec[i] = r.hold
		nc = r.normalize_cts()
		_f = make_errorbar_by_hold(r.hold,r.run,nc.val,0,nc.err)
		if 19 < r.hold < 21:
			norm20.append(nc.val)
		
	_h = make_legend_by_hold(holdVec)
	
	if rRed[0].sing: # Generate Title
		if rRed[0].pmt1 and rRed[0].pmt2:
			plt.title("Normalized Singles Yield by Run")
		elif rRed[0].pmt1:
			plt.title("Normalized PMT 1 Yield by Run")
		elif rRed[0].pmt2:	
			plt.title("Normalized PMT 2 Yield by Run")
	else:
		plt.title("Normalized Coincidence Yield by Run")
	plt.xlabel("Run Number")
	plt.ylabel("Normalized Counts During Unload")
	plt.grid(True)
	# Also print the mean and std of the 20s ncts
	print("Normalization Counts:",np.mean(norm20),np.std(norm20)/np.sqrt(len(norm20)))
	return nPlt
	
#-----------------------------------------------------------------------
def histogram_n_counts(rRed,plot_gen,holdTs = [20,50,100,200,1550]):
	
	print("Generating plot: Normalized counts during unload (histogram, shifted)")
	set_plot_sizing()
	plt.figure(plot_gen)
	nPlt = 1
	# First generate holdVec and nCtsVec from rRed
	holdVec = np.zeros(len(rRed))
	nCtsVec = np.zeros(len(rRed))
	for i in range(len(rRed)):
		holdVec[i] = rRed[i].hold
		nCtsVec[i] = rRed[i].normalize_cts()
	# Next find the mean and std to plot these all on the same axes
	tS = []
	for t in holdTs:
		histBuff,mu,sig,col,lines = make_histogram_by_hold(holdVec,nCtsVec,t)
		tS.append(sig)
	rangeT = np.linspace(-3.0*np.max(tS),3.0*np.max(tS),30) # Histogram range 3 sigma of max
		
	for t in holdTs: # Now loop through the holding times
		histBuff, mu, sig, col, lines = make_histogram_by_hold(holdVec,nCtsVec,t) # Auto-generate histogram buffers
		histBuff -= mu 	# We actually want to shift these histograms so they're centered at 0
		#rangeT = np.linspace(-3.0*sig,3.0*sig,30) # Histogram range 3 sigma
		histT,binsT = np.histogram(histBuff,bins=rangeT,density=False) # Scale histograms
		plt.hist(rangeT[:-1],bins=rangeT,weights=histT,density=True,ec=col,facecolor='none', \
					label=('%ds (%0.04f +/- %0.04f)' % (t,mu,sig))) # And plot them
	plt.title("Normalized Unload Counts Histogram")
	plt.xlabel("Normalized Counts (arb.)")
	plt.ylabel("Frequency")
	plt.legend(loc='upper right')
	
	return nPlt
#-----------------------------------------------------------------------
def plot_normalization_monitor(rRed,plot_gen,rB):
	# The normalization monitor plotting in likelihood_lifetime is better.
	print("Generating plot: Normalization Factor by Run...")
	set_plot_sizing()
	plt.figure(plot_gen)
	nPlt = 1
	_b = make_run_breaks_verts(rB)
	for r in rRed:
		# Hardcoded 2 monitors, which is why there's a better plotter.
		plt.errorbar(r.run,r.mon[0].val,yerr=r.mon[0].err,fmt='r.')
		plt.errorbar(r.run,r.mon[1].val,yerr=r.mon[1].err,fmt='b.')
	
	# Labeling
	plt.errorbar([],[],yerr=[],fmt= 'r.',label="Norm Monitor 1 (Low)")
	plt.errorbar([],[],yerr=[],fmt= 'b.',label="Norm Monitor 2 (High)")
	plt.title("Normalization Monitor by Run")
	plt.ylabel("Normalization Counts")
	plt.xlabel("Run Number")
	plt.legend(loc='upper right')
	plt.grid(True)
	plt.yscale('log')
	
	plt.figure(plot_gen+nPlt)
	nPlt += 1
	_b = make_run_breaks_verts(rB)
	for r in rRed:
		plt.errorbar(r.run,(r.mon[1]/r.mon[0]).val,\
					 yerr=(r.mon[1]/r.mon[0]).err, fmt='g.')
	plt.title("Ratio of Normalization Montitor Counts (High / Low)")
	plt.ylabel("Ratio")
	plt.xlabel("Run Number")
	plt.grid(True)
	plt.yscale('log')
	
	plt.figure(plot_gen+nPlt)
	nPlt += 1
	_b = make_run_breaks_verts(rB)
	for r in rRed:
		plt.errorbar(r.run,r.norm.val,yerr=r.norm.err,fmt='b.')
	plt.errorbar([],[],yerr=[],fmt='b.',label="Normalization Counts")
	plt.title("Normalization Counts")
	plt.ylabel("Expected Counts in Short Hold")
	plt.xlabel("Run Number")
	plt.legend(loc='upper right')
	plt.grid(True)
	plt.yscale('log')	
	
	return nPlt

def plot_single_monitor_normalization(rRed,plot_gen,rB):
	# Plotting single
	print("Generating plot: Normalization Factor by Run...")
	set_plot_sizing()
	plt.figure(plot_gen)
	nPlt = 1
	_b = make_run_breaks_verts(rB)
	for r in rRed:
		# Hardcoded 2 monitors, which is why there's a better plotter.
		plt.errorbar(r.run,r.mon[0].val,yerr=r.mon[0].err,fmt='r.')
		plt.errorbar(r.run,r.mon[1].val,yerr=r.mon[1].err,fmt='b.')
	
	# Labeling
	plt.errorbar([],[],yerr=[],fmt= 'r.',label="Norm Monitor 1 (Low)")
	plt.errorbar([],[],yerr=[],fmt= 'b.',label="Norm Monitor 2 (High)")
	plt.title("Normalization Monitor by Run")
	plt.ylabel("Normalization Counts")
	plt.xlabel("Run Number")
	plt.legend(loc='upper right')
	plt.grid(True)
	plt.yscale('log')
	
	plt.figure(plot_gen+nPlt)
	nPlt += 1
	_b = make_run_breaks_verts(rB)
	for r in rRed:
		plt.errorbar(r.run,(r.mon[1]/r.mon[0]).val,\
					 yerr=(r.mon[1]/r.mon[0]).err, fmt='g.')
	plt.title("Ratio of Normalization Montitor Counts (High / Low)")
	plt.ylabel("Ratio")
	plt.xlabel("Run Number")
	plt.grid(True)
	plt.yscale('log')
	
	plt.figure(plot_gen+nPlt)
	nPlt += 1
	_b = make_run_breaks_verts(rB)
	for r in rRed:
		plt.errorbar(r.run,r.norm.val,yerr=r.norm.err,fmt='b.')
	plt.errorbar([],[],yerr=[],fmt='b.',label="Normalization Counts")
	plt.title("Normalization Counts")
	plt.ylabel("Expected Counts in Short Hold")
	plt.xlabel("Run Number")
	plt.legend(loc='upper right')
	plt.grid(True)
	plt.yscale('log')	
	
	return nPlt


#-----------------------------------------------------------------------
def plot_background_subtracted(rRed,plot_gen,rB):
	set_plot_sizing()
	if rRed[0].sing:
		if rRed[0].pmt1 and rRed[0].pmt2:
			print("Generating plot: Total Singles Background by Run...")
		elif rRed[0].pmt1:
			print("Generating plot: Total PMT 1 Background by Run...")
		elif rRed[0].pmt2:	
			print("Generating plot: Total PMT 2 Background by Run...")
	else:
		print("Generating plot: Total Coincidence Background by Run...")
	
	plt.figure(plot_gen)
	nPlt = 1
	holdVec = []
	for r in rRed:
		holdVec.append(r.hold)
		_h = make_errorbar_by_hold(r.hold,r.run,float(r.bkgSum)/r.len)
		
	_b = make_run_breaks_verts(rB)	
	_h = make_legend_by_hold(holdVec)
		
	if rRed[0].sing:
		if rRed[0].pmt1 and rRed[0].pmt2:
			plt.title("Combined Singles Background by Run")
		elif rRed[0].pmt1:
			plt.title("PMT 1 Background by Run")
		elif rRed[0].pmt2:	
			plt.title("PMT 2 Background by Run")
	else:
		plt.title("Coincidence Background by Run")
	plt.xlabel("Run Number")
	plt.ylabel("Background Rate During Unload (Hz.)")
	
	return nPlt
#-----------------------------------------------------------------------	
def plot_phasespace_evolution(rRed,plot_gen,rB,cfg):

	print("Generating plot: Difference between hold and mean arrival time...")
	set_plot_sizing()
	plt.figure(plot_gen)
	nPlt = 1
	
	_b = make_run_breaks_verts(rB) # Generate run breaks if wanted
	holdVec = []
	for r in rRed:
		if r.hold > 2000: # PSE should always ignore extra long holds
			continue
		holdVec.append(r.hold)
		_h = make_errorbar_by_hold(r.hold,r.run,(r.mat-r.hold).val,0,r.mat.err)
	
	#Formatting labels
	_h = make_legend_by_hold(holdVec)
	
	if cfg.dips == cfg.normDips: # Dip separation routine
		plt.title("Difference between hold and mean arrival time (All Peaks)")
	else:
		t = "Difference between hold and mean arrival time (Peak"
		if len(cfg.dips) == 1:
			t += " "+str(cfg.dips+1)+")"
		else:
			t += "s "
			for i,d in enumerate(cfg.dips):
				t += str(d+1)
				if i+1 < len(cfg.dips) - 1:
					t+= "+"
				else:
					t+= ")"
	plt.xlabel("Run Number")
	plt.ylabel("$t_{mean} - t_{count}$ (s)")

	return nPlt
#-----------------------------------------------------------------------
def plot_dip_percents(rRed,plot_gen,rB,cfg):
	
	print("Generating plot: Percent in Each Dip")	
	set_plot_sizing()
	nPlt = 0
	for d in cfg.dips:	# Could probably optimize this loop
		plt.figure(plot_gen + nPlt)
		nPlt += 1
		holdVec = [] 
		for r in rRed:
			if r.hold > 2000: # PSE should always ignore extra long holds
				continue
			holdVec.append(r.hold)
			_h = make_errorbar_by_hold(r.hold,r.run,r.pcts[d])
		
		t = "Percent in Dip "+str(d+1)
		plt.title(t)
		plt.xlabel("Run Number")
		plt.ylabel("Percent")
	
		#Formatting labels
		_l = make_legend_by_hold(holdVec)

	return nPlt
#-----------------------------------------------------------------------
def histogram_phasespace(rRed,plot_gen,holdTs = [20,50,100,200,1550],dip=1):
		
	print("Generating plot: Histogram of Dip",dip+1,"Distribution")
	set_plot_sizing()
	plt.figure(plot_gen)
	nPlt = 1
	
	# First generate holdVec and dip2Vec from rRed
	holdVec = np.zeros(len(rRed))
	dip2Vec = np.zeros(len(rRed))
	for i in range(len(rRed)):
		holdVec[i] = rRed[i].hold
		dip2Vec[i] = rRed[i].pcts[dip]
	# Next find the mean and std to plot these all on the same axes
	tmin = []
	tmax = []	
	for t in holdTs:
		histBuff,mu,sig,col,lines = make_histogram_by_hold(holdVec,dip2Vec,t)
		tmin.append(np.min(histBuff))
		tmax.append(np.max(histBuff))
	rangeT = np.linspace(np.min(tmin)-0.01,np.max(tmax)+0.01,30) # Histogram range 3 sigma of max
	
	for t in holdTs: # Now loop through the holding times
		histBuff, mu, sig, col, lines = make_histogram_by_hold(holdVec,dip2Vec,t) # Auto-generate histogram buffers
		#rangeT = np.linspace(np.min(histBuff)-0.01,np.max(histBuff)+0.01,20)
		histT,binsT = np.histogram(histBuff,bins=rangeT,density=False) # Scale histograms
		plt.hist(rangeT[:-1],bins=rangeT,weights=histT,density=True,ec=col,facecolor='none', \
					label=('%ds (%0.04f +/- %0.04f)' % (t,mu,sig))) # And plot them
	title = "Histogram of Dip "+str(dip+1)+" Distribution"
	plt.title(title)
	plt.xlabel("Percent")
	plt.ylabel("Counts (arb.)")
	plt.legend(loc='upper right')
	
	return nPlt

#-----	
def histogram_phasespace_SL(rRedAll,plot_gen,holdTs = [20,50,100,200,1550],dip=1):
		
	print("Generating plot: Histogram of Dip",dip+1,"Distribution")
	set_plot_sizing()
	
	# Splitting for 2017/2018
	rRedAl = []
	rRed2017 = []
	rRed2018 = []
	for r in rRedAll:
		if r.run >= 9600:
			rRed2018.append(r)
		elif 4711 <= r.run < 7326:
			rRedAl.append(r)
		else:
			rRed2017.append(r)
	# First generate holdVec and dip2Vec from rRed
	for i,rRed in enumerate([rRed2017, rRedAl, rRed2018]):
		if len(rRed) == 0:
			continue
		plt.figure(plot_gen)
		nPlt = 1
	
		holdVec = np.zeros(len(rRed))
		dip2Vec = np.zeros(len(rRed))
		for i in range(len(rRed)):
			holdVec[i] = rRed[i].hold
			dip2Vec[i] = rRed[i].pcts[dip]
		# Next find the mean and std to plot these all on the same axes
		tmin = []
		tmax = []
		histBuffS,muS,sigS,colS,linesS = make_histogram_short_long(holdVec,dip2Vec,True)
		tmin.append(np.min(histBuffS))
		tmax.append(np.min(histBuffS))
		histBuffL,muL,sigL,colL,linesL = make_histogram_short_long(holdVec,dip2Vec,False)
		tmin.append(np.min(histBuffL))
		tmax.append(np.min(histBuffL))
		rangeT = np.linspace(np.min(tmin)-0.01,np.max(tmax)+0.01,30) # Set range of histogram
		
		# And use our buffers to actually generate short/long plots
		histTS, binsTS = np.histogram(histBuffS,bins=rangeT,density=False)
		plt.hist(rangeT[:-1],bins=rangeTS,weights=histTS, density=True,ec=colS,facecolor='none',\
					linestyle=linesS,label=('Short Runs (%0.04 +/- %0.04f)' % (mu,sig/np.sqrt(len(histBuffS))))) # Plot short			
		plt.hist(rangeT[:-1],bins=rangeTL,weights=histTL, density=True,ec=colS,facecolor='none',\
					linestyle=linesS,label=('Long Runs (%0.04 +/- %0.04f)' % (mu,sig/np.sqrt(len(histBuffL))))) # Plot long
		if i == 0:
			title = "Histogram of Dip "+str(dip+1)+" Distribution, 2017"	
		elif i==1:
			title = "Histogram of Dip "+str(dip+1)+" Distribution, Aluminum Block"	
		elif i==2:
			title = "Histogram of Dip "+str(dip+1)+" Distribution, 2018"
		plt.title(title)
		plt.xlabel("Percent")
		plt.ylabel("Counts (arb.)")
		plt.legend(loc='upper right')
	
	return nPlt

	
#-----------------------------------------------------------------------
def histogram_mean_arr(rRedAll,plot_gen,holdTs = [20,50,100,200,1550]):
	set_plot_sizing()
		
	print("Generating plot: Histogram of Mean Arrival Times")
	
	# First generate holdVec and dip2Vec from rRed
	# Splitting for 2017/2018
	rRedAl = []
	rRed2017 = []
	rRed2018 = []
	for r in rRedAll:
		if r.run >= 9600:
			rRed2018.append(r)
		elif 4711 <= r.run < 7326:
			rRedAl.append(r)
		else:
			rRed2017.append(r)
	# First generate holdVec and dip2Vec from rRed
	for i,rRed in enumerate([rRed2017, rRedAl, rRed2018]):
		if len(rRed) == 0:
			continue
		plt.figure(plot_gen)
		nPlt = 1
		
		holdVec = np.zeros(len(rRed))
		matVec  = np.zeros(len(rRed))
		for i in range(len(rRed)):
			holdVec[i] = rRed[i].hold
			matVec[i]  = rRed[i].mat
		# Next find the mean and std to plot these all on the same axes
		tmin = []
		tmax = []
		for t in holdTs:
			histBuff,mu,sig,col,line = make_histogram_by_hold(holdVec,matVec-holdVec,t)
			tmin.append(np.min(histBuff))
			tmax.append(np.max(histBuff))
		rangeT = np.linspace(np.min(tmin)-0.1,np.max(tmax)+0.1,30) # Histogram range 3 sigma of max
			
		for t in holdTs: # Now loop through the holding times
			histBuff, mu, sig, col, line = make_histogram_by_hold(holdVec,matVec-holdVec,t) # Auto-generate histogram buffers
			#rangeT = np.linspace(np.min(histBuff)-1,np.max(histBuff)+1,20)
			histT,binsT = np.histogram(histBuff,bins=rangeT,density=False) # Scale histograms
			plt.hist(rangeT[:-1],bins=rangeT,weights=histT,density=True,ec=col,linestyle=line,facecolor='none', \
						label=('%ds (%0.04f +/- %0.04f)' % (t,mu,sig))) # And plot them
		if i == 0:
			t = "Histogram of Mean Arrival Times, 2017"
		elif i==1:
			t = "Histogram of Mean Arrival Times, Aluminum Block"	
		elif i==2:
			t = "Histogram of Mean Arrival Times, 2018"
		plt.title(t)
		plt.xlabel("Percent")
		plt.ylabel("MAT (s)")
		plt.legend(loc='upper right')
	
	return nPlt
	
def histogram_mean_arr_SL(rRedAll,plot_gen,holdTs = [20,50,100,200,1550]):
	set_plot_sizing()
		
	print("Generating plot: Histogram of Mean Arrival Times")
	# First generate holdVec and dip2Vec from rRed
	
	# Splitting for 2017/2018
	rRedAl = []
	rRed2017 = []
	rRed2018 = []
	for r in rRedAll:
		if r.run >= 9600:
			rRed2018.append(r)
		elif 4711 <= r.run < 7326:
			rRedAl.append(r)
		else:
			rRed2017.append(r)
	# First generate holdVec and dip2Vec from rRed
	for i,rRed in enumerate([rRed2017, rRedAl, rRed2018]):
		if len(rRed) == 0:
			continue
		plt.figure(plot_gen)
		nPlt = 1
	
		
		holdVec = np.zeros(len(rRed))
		matVec  = np.zeros(len(rRed))
		for i in range(len(rRed)):
			holdVec[i] = rRed[i].hold
			matVec[i]  = rRed[i].mat
		# Next find the mean and std to plot these all on the same axes
		tmin = []
		tmax = []
		for t in holdTs:
			histBuff,mu,sig,col,line = make_histogram_by_hold(holdVec,matVec-holdVec,t)
			tmin.append(np.min(histBuff))
			tmax.append(np.max(histBuff))
		rangeT = np.linspace(np.min(tmin)-0.1,np.max(tmax)+0.1,30) # Histogram range 3 sigma of max
			
		for t in holdTs: # Now loop through the holding times
			histBuff, mu, sig, col, line = make_histogram_by_hold(holdVec,matVec-holdVec,t) # Auto-generate histogram buffers
			#rangeT = np.linspace(np.min(histBuff)-1,np.max(histBuff)+1,20)
			histT,binsT = np.histogram(histBuff,bins=rangeT,density=False) # Scale histograms
			plt.hist(rangeT[:-1],bins=rangeT,weights=histT,density=True,ec=col,linestyle=line,facecolor='none', \
						label=('%ds (%0.04f +/- %0.04f)' % (t,mu,sig))) # And plot them
		if i == 0:
			t = "Histogram of Mean Arrival Times, 2017"
		elif i==1:
			t = "Histogram of Mean Arrival Times, Aluminum Block"	
		elif i==2:
			t = "Histogram of Mean Arrival Times, 2018"
		plt.title(t)
		plt.xlabel("Percent")
		plt.ylabel("MAT (s)")
		plt.legend(loc='upper right')
	
	return nPlt
#-----------------------------------------------------------------------
def plot_signal2noise(rRed,plot_gen,rB):
	# I think that this is for whatever odd plot I've wanted to use it for
	
	set_plot_sizing()
	print("Generating plot: Signal to Noise...")
	plt.figure(plot_gen)
	nPlt = 1
	
	_b = make_run_breaks_verts(rB) # For if we're plotting runBreaks
	
	for r in rRed:
		s2n =r.cts/(r.bkgSum+r.tCSum)
		_h = make_errorbar_by_hold(r.hold,r.run,s2n.val,0,s2n.err)
	
	#Formatting labels
	_l = make_legend_by_hold(holdVec)
	
	plt.title("Signal to Noise")
	plt.xlabel("Run Number")
	plt.ylabel("Signal to Noise")
	plt.legend(loc='upper right')	
	
	return nPlt
