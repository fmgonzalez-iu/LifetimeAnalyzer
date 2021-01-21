#!/usr/local/bin/python3
import numpy as np
from scipy.optimize import curve_fit
#import sys
#import pdb
#import csv
#from math import *

#import emcee 					#MCMC module
#import matplotlib.pyplot as plt

#from statsmodels.stats.weightstats import DescrStatsW
#from scipy import stats, special
#from scipy.odr import *

from PythonAnalyzer.classes import *
from PythonAnalyzer.functions import map_height,\
									 spectral_norm_meas,\
									 spectral_norm_meas_inv,\
									 spectral_norm_meas_inv2
from PythonAnalyzer.backgrounds import *

#from scipy.special import factorial
#from datetime import datetime
#from multiprocessing import Pool  #for parallelizing

#from PythonAnalyzer.writeOut import write_summed_counts

#from numpy.random import normal   #just a normal thing to do in any code

def analyze_single_norm(runList,cts,nMon,bkgs,bkgT,anlz,runBreaks = []):
	# This is the single-norm analyzer, creating a reduced_run list.
	#-------------------------------------------------------------------
	# Inputs:
	#    Automated runList
	#    Data from *-det.csv
	#    Data from *-dag.csv OR MAD*.csv
	#    analyzer_cfg object
	#    (Optional) list of runBreaks
	#-------------------------------------------------------------------
	# Outputs:
	#    reduced_runs list
	#-------------------------------------------------------------------
	
	if (runBreaks == []):
		runBreaks = [runList[0],max(runList)+1]
	
	reducedRun = extract_reduced_runs(runList,cts,nMon,bkgs,bkgT,anlz) # Get run info
	reducedRun = normalization_reduced(reducedRun,runBreaks,anlz) # normalization
	reducedRun = expand_reduced_run(reducedRun,runBreaks,anlz) # Expand normalization to everyone else
	
	#for r in reducedRun:
	#	print(r.total_cts(),r.norm)
	if anlz.vb:
		print("Managed to successfully normalize %d runs!" % len(reducedRun)) 
		print("   This is a ratio of: %f" % (float(len(reducedRun))/float(len(runList))))
		print(" ")
	if len(reducedRun) < 8:
		sys.exit("Error! Unable to normalize an entire octet's worth of runs! Exiting...")
	
	return reducedRun

def extract_reduced_runs(rL, cts, nMon, bkgsH,bkgsT,cfg):
	# Extract the reduced values from our runs. 
	#-------------------------------------------------------------------
	# Input values:
	# rL = runlist
	# cts =  *-dag.csv list in numpy format
	# nMon = *-det.csv list in numpy format
	# bkgs = list of bkgHgtDep objects (and maybe bkgTimeDep objects)
	# cfg  = analyzer_cfg settings
	#-------------------------------------------------------------------
	# Output values;
	# rRedL = array of reduced runs
	#-------------------------------------------------------------------
	
	rRedL = []
	
	# Which monitors are we picking out?
	# Note that this is in cfg -- want to be able to edit by changing cfg
	if cfg.year==2017: 
		monStr1  = ('mon'+str(cfg.det17[0]))
		monEStr1 = ('mon'+str(cfg.det17[0])+'E')
		monStr2  = ('mon'+str(cfg.det17[1]))
		monEStr2 = ('mon'+str(cfg.det17[1])+'E')
	else:
		monStr1  = ('mon'+str(cfg.det18[0]))
		monEStr1 = ('mon'+str(cfg.det18[0])+'E')
		monStr2  = ('mon'+str(cfg.det18[1]))
		monEStr2 = ('mon'+str(cfg.det18[1])+'E')
		
	bkgOut = open("BkgSep.csv","a")
	for run in rL:
		# Initialize reduced run. This initializes the sums.
		rRed = reduced_run(run) 
		rRed.sing = cfg.sing # Port info from config
		rRed.pmt1 = cfg.pmt1
		rRed.pmt2 = cfg.pmt2
		rRed.thresh = cfg.thresh
		# Load the raw data from each run -- the file is separated by dip already.
		ctsRaw  = cts[cts['run']==run]
		nMonRaw = nMon[nMon['run']==run]
		
		# For each run define nDips and holdT
		holdT = (nMonRaw['ts'] - nMonRaw['td']) - 50.0 # include cleaning time
		try:
			rRed.hold = float(holdT)
		except TypeError:
			print(run,holdT)
			continue
			
			
		if (holdT < 0.0):
			if cfg.vb:
				print("Hold Time is negative for run %d!" %  run)
			continue
			
		# Load Background Height Dependence 
		bHDep = bkgHgtDep(run,run+1) # Initialize with this run
		if cfg.usePosBkgs: # Should initialize at 1 for everything
			for b in bkgsH:
				if b.is_run(run):
					bHDep = b
					break
		# Load Background Time Dependence
		bTDep = bkgTimeDep(run,run+1)
		if cfg.useTimeBkgs: # Should initialize with 0, inf for (a,b),(t1,t2)
			for b in bkgsT:
				if b.is_run(run):
					bTDep = b
					break
		
		if cfg.useBkgs:
			bkg_EOR = extract_background_obj(nMonRaw,cfg)
			bkg_Unl = extract_background_unload_obj(ctsRaw,cfg)
			# Load up the background dependencies
			bkg_EOR.hDep = bHDep
			bkg_EOR.tDep = bTDep
			bkg_Unl.hDep = bHDep
			bkg_Unl.tDep = bTDep
			#if cfg.sing:
			#	if cfg.pmt1 and cfg.pmt2:
			#		bkgOut.write("%05d,%f,%f,%f,%f,%f,%f\n"%(rRed.run,bkg_EOR.time,bkg_EOR.dt,bkg_EOR.pmt1+bkg_EOR.pmt2,bkg_Unl.time,bkg_Unl.dt,bkg_Unl.pmt1+bkg_Unl.pmt2))	
			#	elif efg.pmt1:
			#		bkgOut.write("%05d,%f,%f,%f,%f,%f,%f\n"%(rRed.run,bkg_EOR.time,bkg_EOR.dt,bkg_EOR.pmt1,bkg_Unl.time,bkg_Unl.dt,bkg_Unl.pmt1))	
			#	else:
			#		bkgOut.write("%05d,%f,%f,%f,%f,%f,%f\n"%(rRed.run,bkg_EOR.time,bkg_EOR.dt,bkg_EOR.pmt2,bkg_Unl.time,bkg_Unl.dt,bkg_Unl.pmt2))	
			#else:
			#	bkgOut.write("%05d,%f,%f,%f,%f,%f,%f\n"%(rRed.run,bkg_EOR.time,bkg_EOR.dt,bkg_EOR.coinc,bkg_Unl.time,bkg_Unl.dt,bkg_Unl.coinc))	
			
			bkg_tot = bkg_EOR + bkg_Unl
			#print("%05d,%f,%f,%f\n"%(rRed.run,bkg_tot.coinc,bkg_tot.time,bkg_tot.dt))
		else:
			bkg_tot = bkgStruct(run) # Initializes at 0!
			# Add height/time dependence to struct	
		
		nDips = max(ctsRaw['dip']) # Note that this is zero indexed
		if nDips != cfg.ndips - 1: # Check that we're doing the right number of dips
			if cfg.vb:
				print("Run %d is not a proper %d dip run" % (run, cfg.ndips))
			continue
		
		rRed.eff = get_efficiencies(ctsRaw,cfg)
		if rRed.sing: # Now do our main meat calculations
			#if cfg.scaleSing:
			#rRed.eff = get_efficiencies(ctsRaw,cfg)
			#else:
			#rRed.eff = [measurement(0.5,0.0),measurement(0.5,0.0)]
			ctsSumL, dtSumL, bkgSumL,bkgHSumL,tCSumL = sum_singles_counts(ctsRaw, cfg, bkg_tot)# normBKG1, normBKG2, bkgTime)
		else:
			#[ctsSumL, dtSumL, bkgSumL],[d1c,d2c],tCSumL = sum_singles_counts(ctsRaw, normBKG1, normBKG2, bkgTime, maxUnl, pmt1,pmt2,dBkg,normBKG)
			#rRed.eff = [measurement(0.5,0.0),measurement(0.5,0.0)]
			ctsSumL, dtSumL, bkgSumL,bkgHSumL,tCSumL = sum_coinc_counts(ctsRaw,cfg,bkg_tot)#normBKG,bkgTime)#maxUnl, dBkg)
			#[ctsSumL, dtSumL, bkgSumL] = sum_coinc_counts(ctsRaw, normBKG,bkgTime, maxUnl, dBkg)
			#[d1c,d2c] = [measurement(0.5,0.0),measurement(0.5,0.0)] # Efficiency equalized between the two PMTs
			#tCSumL = []
			#for k in range(len(bkgSumL)):
			#	tCSumL.append(measurement(0.0,0.0)) # Time dependence for coincidences is presently zero.
		totCts = measurement(0.0,0.0)
		
		for i in range(cfg.ndips):
			# And add these together into our object
			rRed.ctsSum  += ctsSumL[i]
			rRed.dtSum   += dtSumL[i]
			rRed.bkgSum  += bkgSumL[i]
			rRed.bkgHSum += bkgHSumL[i]
			rRed.tCSum   += tCSumL[i]
		# And loop again to get the dip percents
		for i in range(cfg.ndips):
			dipCts = ctsSumL[i]
			if cfg.useDTCorr:
				dipCts += dtSumL[i]
			if cfg.useBkgs:
				dipCts -= bkgSumL[i]
			rRed.pcts[i] = (dipCts / rRed.total_cts(cfg)).val
			
		rRed = find_mean_arrival(rRed,ctsRaw,bkg_tot,cfg)
		rRed.mat -= measurement(nMonRaw['td']+50.0,0.0) # Don't care about filling/cleaning
		rRed.mon = [measurement(nMonRaw[monStr1][0],nMonRaw[monEStr1][0]), \
					measurement(nMonRaw[monStr2][0],nMonRaw[monEStr2][0])]
		rRedL.append(rRed)
		
	bkgOut.close()
	return rRedL

def extract_reduced_runs_all_mon(rL, cts, nMon, bkgsH,bkgsT,cfg):
	# Extract the reduced values from our runs. 
	#-------------------------------------------------------------------
	# Input values:
	# rL = runlist
	# cts =  *-dag.csv list in numpy format
	# nMon = *-det.csv list in numpy format
	# bkgs = list of bkgHgtDep objects (and maybe bkgTimeDep objects)
	# cfg  = analyzer_cfg settings
	#-------------------------------------------------------------------
	# Output values;
	# rRedL = array of reduced runs
	#-------------------------------------------------------------------
	
	rRedL = []	
	for run in rL:
		# Initialize reduced run. This initializes the sums.
		rRed = reduced_run(run) 
		rRed.sing = cfg.sing # Port info from config
		rRed.pmt1 = cfg.pmt1
		rRed.pmt2 = cfg.pmt2
		rRed.thresh = cfg.thresh
		# Load the raw data from each run -- the file is separated by dip already.
		ctsRaw  = cts[cts['run']==run]
		nMonRaw = nMon[nMon['run']==run]
		
		# For each run define nDips and holdT
		holdT = (nMonRaw['ts'] - nMonRaw['td']) - 50.0 # include cleaning time
		try:
			rRed.hold = float(holdT)
		except TypeError:
			print(run,holdT)
			continue
			
			
		if (holdT < 0.0):
			if cfg.vb:
				print("Hold Time is negative for run %d!" %  run)
			continue
			
		# Load Background Height Dependence 
		bHDep = bkgHgtDep(run,run+1) # Initialize with this run
		if cfg.usePosBkgs: # Should initialize at 1 for everything
			for b in bkgsH:
				if b.is_run(run):
					bHDep = b
					break
		# Load Background Time Dependence
		bTDep = bkgTimeDep(run,run+1)
		if cfg.useTimeBkgs: # Should initialize with 0, inf for (a,b),(t1,t2)
			for b in bkgsT:
				if b.is_run(run):
					bTDep = b
					break
		
		if cfg.useBkgs:
			bkg_EOR = extract_background_obj(nMonRaw,cfg)
			bkg_Unl = extract_background_unload_obj(ctsRaw,cfg)
			
			# Load up the background dependencies
			bkg_EOR.hDep = bHDep
			bkg_EOR.tDep = bTDep
			bkg_Unl.hDep = bHDep
			bkg_Unl.tDep = bTDep			
			bkg_tot = bkg_EOR + bkg_Unl
		else:
			bkg_tot = bkgStruct(run) # Initializes at 0!
		
		nDips = max(ctsRaw['dip']) # Note that this is zero indexed
		if nDips != cfg.ndips - 1: # Check that we're doing the right number of dips
			if cfg.vb:
				print("Run %d is not a proper %d dip run" % (run, cfg.ndips))
			continue
		
		rRed.eff = get_efficiencies(ctsRaw,cfg)
		if rRed.sing: # Now do our main meat calculations
			ctsSumL, dtSumL, bkgSumL,bkgHSumL,tCSumL = sum_singles_counts(ctsRaw, cfg, bkg_tot)
		else:
			ctsSumL, dtSumL, bkgSumL,bkgHSumL,tCSumL = sum_coinc_counts(ctsRaw,cfg,bkg_tot)
			
		totCts = measurement(0.0,0.0)
		
		for i in range(cfg.ndips):
			# And add these together into our object
			rRed.ctsSum  += ctsSumL[i]
			rRed.dtSum   += dtSumL[i]
			rRed.bkgSum  += bkgSumL[i]
			rRed.bkgHSum += bkgHSumL[i]
			rRed.tCSum   += tCSumL[i]
		# And loop again to get the dip percents
		for i in range(cfg.ndips):
			dipCts = ctsSumL[i]
			if cfg.useDTCorr:
				dipCts += dtSumL[i]
			if cfg.useBkgs:
				dipCts -= bkgSumL[i]
			rRed.pcts[i] = (dipCts / rRed.total_cts(cfg)).val
			
		rRed = find_mean_arrival(rRed,ctsRaw,bkg_tot,cfg)
		rRed.mat -= measurement(nMonRaw['td']+50.0,0.0) # Don't care about filling/cleaning
		monL = np.zeros(10,dtype=measurement)
		for i in range(1,11):
			monStr  = 'mon' + str(i)
			monEStr = 'mon' + str(i) + 'E' 
			monL[i-1] = measurement(nMonRaw[monStr][0],nMonRaw[monEStr][0])# Offset by 1, because 0 vs. 1 indexing
		rRed.mon = monL
		rRedL.append(rRed)
	
	return rRedL

def expand_reduced_run(rRed,rBreaks,cfg):
	# We've normalized to a single holding time, now let's do everything 
	# else
	#-------------------------------------------------------------------
	# Double check runBreak boundaries
	if rRed[0].run < rBreaks[0]:
		if cfg.vb:
			print("Warning: First element of runList is less than the first runBreak!")
			print("       I'll try to run this but something might be un-normalized!")
		rBreaks.append(rRed[0].run)
		rBreaks.sort()
	if rRed[-1].run > rBreaks[-1]:
		if cfg.vb:
			print("Warning: Last element of runList is more than the last runBreak!")
			print("        I'll try to run this but something might be un-normalized!")
		rBreaks.append(rRed[-1].run)
	
	#rRed = weighted_average_efficiencies(rRed,rBreaks)
	# Create list of runs to normalize to
	runList = []
	normList = []
	indList  = [] # For putting indices in the right places
	if cfg.hold > 0:
		holdList = []
		totalInds = []
		totalHolds = []
		for i,r in enumerate(rRed):
			# Normalization list, +/- 1s slop:
			runList.append(r.run) # runList is all runs
			totalHolds.append(int(round(r.hold)))
			totalInds.append(i)
			if cfg.hold - 1 < r.hold < cfg.hold + 1:
				normList.append(r.run)
				indList.append(i) # These are the runs we've already normalized
				# First get any runs with the right holding time
			elif int(round(r.hold)) not in holdList:
				holdList.append(int(round(r.hold)))
		runList = np.array(runList)
		totalInds = np.array(totalInds)
		totalHolds = np.array(totalHolds)
		normList = np.array(normList)
		holdList.sort() # sort minimum extra holds
		# If there are runs without the right holding time, deal with that now
		for b in range(len(rBreaks)): # Where are our runbreaks?
			if b == len(rBreaks) - 1: # End on last runBreak
				continue
			bCond = (rBreaks[b] <= normList)*(normList < rBreaks[b+1])
			rL = normList[bCond] # Which subset of runs is in the break?
			if len(rL) == 0: # OK, now we need to add additional runs.
				for h in holdList:
					hCond = (totalHolds==h)
					bCondTmp = (rBreaks[b] <= runList)*(runList < rBreaks[b+1])
					newRuns = runList[bCondTmp*hCond]
					newInds = totalInds[bCondTmp*hCond]
					if len(newRuns) > 0:
						np.append(normList,newRuns)
						np.append(indList,newInds)
						break
	else:
		return rRed
	runList = np.array(runList) # Cast to numpy arrays
	normList = np.array(normList)
	indList = np.array(indList) 
	
	# And loops
	for i, r in enumerate(runList):		
		for b in range(len(rBreaks)): # Get the correct run break
			if b == len(rBreaks) - 1:
				continue	
			if not (rBreaks[b] <= r < rBreaks[b+1]): # Forcing correct runbreaks
				continue
			bCond = (rBreaks[b] <= normList)*(normList < rBreaks[b+1])
			nLR = normList[bCond]
			iLR = indList[bCond]
			
			for j, n in enumerate(nLR): # now find the closest normList
				if j == len(nLR)-1: # Can't have anything higher
					rRed[i].alpha = rRed[iLR[j]].alpha
					rRed[i].beta  = rRed[iLR[j]].beta
					rRed[i].alphaE = rRed[iLR[j]].alphaE
					rRed[i].betaE  = rRed[iLR[j]].betaE
					rRed[i].cov    = rRed[iLR[j]].cov
					rRed[i].nTime  = rRed[iLR[j]].nTime
					break
				if j == 0: # For the first pass we can just do the easy thing
					if (abs(r - n) < abs(r - nLR[j+1])) and (r < nLR[j+1]):
						rRed[i].alpha = rRed[iLR[j]].alpha
						rRed[i].beta  = rRed[iLR[j]].beta
						rRed[i].alphaE = rRed[iLR[j]].alphaE
						rRed[i].betaE  = rRed[iLR[j]].betaE
						rRed[i].cov    = rRed[iLR[j]].cov
						rRed[i].nTime  = rRed[iLR[j]].nTime
						break
				if r < nLR[j-1]: # Now check that we're in a decent region
					continue
				elif r > nLR[j+1]: # Too far!
					continue
				if abs(r-n) < abs(r - nLR[j+1]):
					rRed[i].alpha = rRed[iLR[j]].alpha
					rRed[i].beta  = rRed[iLR[j]].beta
					rRed[i].alphaE = rRed[iLR[j]].alphaE
					rRed[i].betaE  = rRed[iLR[j]].betaE
					rRed[i].cov    = rRed[iLR[j]].cov
					rRed[i].nTime  = rRed[iLR[j]].nTime
					break
		rRed[i].norm = rRed[i].norm_unl(cfg)	
		rRed[i].cts  = rRed[i].pct_cts(cfg)		
	return rRed

def get_efficiencies(cts,cfg):
	# Just modify runRed directly...
	if not cfg.sing: # Coincidence uses both!
		return [measurement(0.5,0.0),measurement(0.5,0.0)]
	if not cfg.scaleSing: # Singles uses the two independently
		return [measurement(1.0,0.0),measurement(1.0,0.0)]
	# Figure out the number of PEs/UCN
	d1cT = [] # Want to scale singles to coincidences
	d2cT = []
	p1Scale = [] # Weight the averages by number of photons
	p2Scale = [] # This should be # of coincidences, but this is a proxy
	try: # Check if we have coincidence scaling incorporated 
		for row in cts: # Sum up all the bins in our dip				
			p1Scale.append(row['d1'])
			p2Scale.append(row['d2'])
			if row['d1C'] > 0:
				d1cT.append(row['d1C'])
			else:
				d1cT.append(0.0)
			if row['d2C'] > 0:
				d2cT.append(row['d2C'])
			else:
				d2cT.append(0.0)	
	except ValueError: # No coincidence scale recorded 
		d1cT = []
		d2cT = []
		
	if len(d1cT) > 0: # Take the mean across our dip
		p1Scale = np.array(p1Scale)
		d1cT = np.array(d1cT)
		d1c = np.sum(p1Scale*d1cT/np.sum(p1Scale))		
		# From error propagation, uncertainty is given by:
		d1cE = d1c*np.sqrt(1+d1c)/np.sqrt(np.sum(p1Scale))	
		#d1cE = np.sqrt(d1c)
	else:
		d1c = 8.0 # Default is twice the threshold
		d1cE = 0.0
	if len(d2cT) > 0:
		p2Scale = np.array(p2Scale)
		d2cT = np.array(d2cT)
		d2c = np.sum(p2Scale*d2cT/np.sum(p2Scale))
		d2cE = d2c*np.sqrt(1+d2c)/np.sqrt(np.sum(p2Scale))
		#d2cE = np.sqrt(d2c)
	else:
		d2c = 8.0
		d2cE = 0.0
	return [measurement(d1c,d1cE),measurement(d2c,d2cE)]

def weighted_average_efficiencies(rRed, rBreaks):
	# We actually want to calculate the weighted average of the efficiency
	# in each runbreak. This way we can maximize statistics and won't have
	# weird (long) runs with a statistically significant different norm.
	#
	# Long runs are noisier (fewer counts) but the noise is non-symmetric.
	# It's more likely to have a lower efficiency in long than higher.
	#-------------------------------------------------------------------
	if not rRed[0].sing: # Only average over singles
		return rRed
	
	# Double check runBreak boundaries
	if rRed[0].run < rBreaks[0]:
		rBreaks.append(rRed[0].run)
		rBreaks.sort()
	if rRed[-1].run > rBreaks[-1]:
		rBreaks.append(rRed[-1].run+1)
	
	# Create list of runs to normalize to
	runList = []
	indList = [] # For putting indices in the right places	
	for i,r in enumerate(rRed):
		runList.append(r.run)
		indList.append(i)
	
	runList = np.array(runList) # Cast to numpy arrays
	indList = np.array(indList) 
	for b in range(len(rBreaks)):
		if b == len(rBreaks)-1:
			continue
		# Find all the runs inside the break
		bCond = (rBreaks[b] <= runList)*(runList < rBreaks[b+1])
		iL = indList[bCond]
		mu1	= 0.0
		mu2 = 0.0
		wts1 = 0.0
		wts2 = 0.0
		for i in iL: # Weighted mean
			v1 = rRed[i].eff[0].val/(rRed[i].eff[0].err*rRed[i].eff[0].err)
			e1 = 1.0/(rRed[i].eff[0].err*rRed[i].eff[0].err)
			v2 = rRed[i].eff[1].val/(rRed[i].eff[1].err*rRed[i].eff[1].err)
			e2 = 1.0/(rRed[i].eff[1].err*rRed[i].eff[1].err)
			mu1 += v1
			mu2 += v2
			wts1 += e1
			wts2 += e2
		mu1 /= wts1
		mu2 /= wts2
		for i in iL: # And update rRed
			# Keeping uncertainty the same though?
			rRed[i].eff = [measurement(mu1,1/np.sqrt(wts1)),\
						   measurement(mu2,1/np.sqrt(wts2))]
			#rRed[i].eff = [measurement(mu1,rRed[i].eff[0].err),\
			#			   measurement(mu2,rRed[i].eff[1].err)]
	return rRed		
	
#def sum_singles_counts(cts, cfg, bkg1 = 0.0,bkg2 = 0.0, bkgT = np.inf): 
def sum_singles_counts(cts, cfg, bkg):#bkg1 = 0.0,bkg2 = 0.0, bkgT = np.inf): 
					   #maxUnl = 100, pmt1 = True, pmt2 = True,
					   #dBkg = True, bkgC = 0):
	# Sum all the counts we want to use. This is just for singles.
	# Input a cts vector separated by dips/slices, a background vector
	# and some bools to get this working.
	#
	# Returns measurement arrays [ctsSumL, dtSumL, bkgSumL]	
	#-------------------------------------------------------------------
	
	# Figure out the dagger heights
	#nDips = max(ct['dip'])+1 # +1 to get rid of zero indexing
	# hMap 4 is for deep dagger cleaning
	
	hgt = map_height(cfg)
	
	eff = get_efficiencies(cts,cfg) # Figure out the number of PEs/UCN
	d1c = eff[0].val
	d2c = eff[1].val
	
	# Load output sums by dip
	ctsSumL  = [] # Raw counts
	dtSumL   = [] # Deadtime Correction
	bkgSumL  = [] # Background sum
	bkgHSumL = [] # Height Dependent Background
	tCSumL   = [] # Time Dependence correction
	
	for i in range(0,cfg.ndips): # Doing this so that we can skip dips
		ctsSumL.append(measurement(0.0,0.0))
		dtSumL.append( measurement(0.0,0.0))
		bkgSumL.append(measurement(0.0,0.0))
		bkgHSumL.append(measurement(0.0,0.0))
		tCSumL.append( measurement(0.0,0.0))
		
	# Add all counts from the different dips.
	for i in range(0,cfg.ndips):
		ctsSum = measurement(0.0,0.0) # (Re-)Initialize tmp sums
		dtSum  = measurement(0.0,0.0)
		bkgSum = measurement(0.0,0.0)
		bkgHD  = measurement(0.0,0.0)
		tCSum  = measurement(0.0,0.0)
		#hgtD1 = measurement(0.0,0.0)
		#hgtD2 = measurement(0.0,0.0)
	
		ctsByDip = cts[cts['dip']==i] # Separate by dip
		
		for row in ctsByDip: # Now step through the dip slices
			
			if (not cfg.useMoving): # Ignore the first step when the dagger is moving
				if (row['ts'] == ctsByDip[0]['ts']):
					continue
			if (row['te'] - ctsByDip[0]['ts'] > cfg.maxUnl + 1): # Check max position
				break	
																
			if cfg.pmt1: # Separate by PMT
				cts1 = float(row['d1']) # Counts for dip
				dt1  = float(row['d1DT']) # Deadtime correction for step
			else:
				cts1    = 0.0
				dt1     = 0.0
			#	d1c     = 0.0
			if cfg.pmt2:
				cts2 = float(row['d2'])
				dt2  = float(row['d2DT'])
			else:
				cts2 = 0.0
				dt2  = 0.0
			#	d2c  = 0.0
			#sT = (row['ts'] + row['te']) / 2.0 # mean step time
			sT = (cts1*row['m1'] +cts2*row['m2']) / (cts1+cts2) # mean step time
			
			# Background (either position dependent or not). 
			# If a PMT is off, the bkg[i] should all be 0
			bkgFcn1,bkg1_Pos,bkg1_t = bkgFunc_obj(bkg,1,hgt[i],sT)
			bkgFcn2,bkg2_Pos,bkg2_t = bkgFunc_obj(bkg,2,hgt[i],sT)
			#bkgFcn2 = bkgFunc_obj(bkg,2,hgt[i],sT)
			bkgFcn1.err = bkgFcn1.err*np.sqrt(d1c)
			bkgFcn2.err = bkgFcn2.err*np.sqrt(d2c)
			bkg1_Pos.err *= np.sqrt(d1c)
			bkg2_Pos.err *= np.sqrt(d2c)
			bkg1_t.err   *= np.sqrt(d1c)
			bkg2_t.err   *= np.sqrt(d2c)
			
			#bkgFcn1_TInd = bkgFunc_obj(bkg,1,hgt[i],bkg.time)
			#bkgFcn2_TInd = bkgFunc_obj(bkg,2,hgt[i],bkg.time)
			#bkgFcn1_TInd.err = bkgFcn1_TInd.err*np.sqrt(d1c)
			#bkgFcn2_TInd.err = bkgFcn2_TInd.err*np.sqrt(d2c)
			
			#if cfg.usePosBkgs: 
			#	
			#	bkgFcn1 = bkgFunc(bkg.pmt1,bkg.hDep,row['run'], 1, hgt[i], bkg.hgt, sT, bkg.time)
			#	bkgFcn2 = bkgFunc(bkg.pmt2,bkg.hDep,row['run'], 2, hgt[i], bkg.hgt, sT, bkg.time)
			#	bkgFcn1.err = bkgFcn1.err*np.sqrt(d1c)
			#	bkgFcn2.err = bkgFcn2.err*np.sqrt(d2c)
				
			#	bkgFcn1_TInd = bkgFunc(bkg.pmt1,bkg.hDep,row['run'],1,hgt[i],bkg.hgt,bkg.time,bkg.time)
			#	bkgFcn2_TInd = bkgFunc(bkg.pmt2,bkg.hDep,row['run'],2,hgt[i],bkg.hgt,bkg.time,bkg.time)
			#	bkgFcn1_TInd.err = bkgFcn1_TInd.err*np.sqrt(d1c)
			#	bkgFcn2_TInd.err = bkgFcn2_TInd.err*np.sqrt(d2c)
			
			# This was a test to separate out the poisson-ish coincidence backgrounds
			tryCoinc = False
			if tryCoinc:
					
					# Background is dark noise + coincidence background
					# Correct PMT backgrounds for coincidence (for height-dep. backgrounds)
					bkg1_corr = bkg1 - bkgC*d1c
					bkg2_corr = bkg2 - bkgC*d2c
					
					#bkgFcn1 = dBKG1(hgt[i], sT, bkg1, row['run'], bkgT)
					#bkgFcn2 = dBKG2(hgt[i], sT, bkg2, row['run'], bkgT)
					bkgFcn1_U = bkgFunc(bkg1,bkg.hDep, row['run'], 1, hgt[i], 10.0, sT, bkgT)
					bkgFcn2_U = bkgFunc(bkg2,bkg.hDep, row['run'], 2, hgt[i], 10.0, sT, bkgT)
					
					bkgFcn1 = bkgFunc(bkg1_corr,bkg.hDep,row['run'], 1, hgt[i], 10.0, sT, bkgT)
					bkgFcn2 = bkgFunc(bkg2_corr,bkg.hDep,row['run'], 2, hgt[i], 10.0, sT, bkgT)
					bkgFcnC = bkgFunc(bkgC,     bkg.hDep,row['run'], 0, hgt[i], 10.0, sT, bkgT)
					
					bkgFcn1.err = bkgFcn1.err*np.sqrt(d1c)
					bkgFcn2.err = bkgFcn2.err*np.sqrt(d2c)
					
					# Recuperate PMT backgrounds with PMT 
					bkgFcn1 += bkgFcnC*measurement(d1c,np.sqrt(d1c))
					bkgFcn2 += bkgFcnC*measurement(d2c,np.sqrt(d2c))
									
					# Now calculate the time-independent part of this
					bkgFcn1_TInd = bkgFunc(bkg1_corr,bkg.hDep,row['run'],1,hgt[i],10.0,bkgT,bkgT)
					bkgFcn2_TInd = bkgFunc(bkg2_corr,bkg.hDep,row['run'],2,hgt[i],10.0,bkgT,bkgT)
					bkgFcnC_TInd = bkgFunc(bkgC,     bkg.hDep,row['run'],0,hgt[i],10.0,bkgT,bkgT) # Not necessary with no time dependence but whatever
								
					# Same scaling as before	
					bkgFcn1_TInd.err = bkgFcn1_TInd.err*np.sqrt(d1c)
					bkgFcn2_TInd.err = bkgFcn2_TInd.err*np.sqrt(d2c)
					
					# Recuperate PMT backgrounds with PMT 
					bkgFcn1_TInd += bkgFcnC_TInd*measurement(d1c,np.sqrt(d1c))
					bkgFcn2_TInd += bkgFcnC_TInd*measurement(d2c,np.sqrt(d2c))
					
				#hgtD1 += (bkg1-bkgFcn1.val)
				#hgtD2 += (bkg2-bkgFcn2.val)
				#hgtD1 += bkgFcn1 - bkgFcn1_TInd
				#hgtD2 += bkgFcn2 - bkgFcn2_TInd
				#print(i,(hgtD1+hgtD2))
				# Scale up errors on backgrounds to be poissonian (for neutrons, not photons)
				#bkgFcn1.val = bkgFcn1.val
				#bkgFcn2.val = bkgFcn2.val
				
				
			# else: # Assigning poissonian error (for non-position corrected)
				# if bkg1 > 0.0: 
					# #bkgFcn1 = measurement(bkg1,sqrt(bkg1*d1c))
					# #bkgFcn1 = measurement(bkg1,sqrt(bkg1*d1c))
					# bkgFcn1 = measurement(bkg.pmt1,np.sqrt(bkg.pmt1*d1c))
				# else:
					# bkgFcn1 = measurement(0.0,0.0) 
				# if bkg2 > 0.0:
					# #bkgFcn2 = measurement(bkg2,sqrt(bkg2*d2c))
					# #bkgFcn2 = measurement(bkg2,sqrt(bkg2*d2c))
					# bkgFcn2 = measurement(bkg.pmt2,np.sqrt(bkg.pmt2*d2c))
				# else:
					# bkgFcn2 = measurement(0.0,0.0)
				# bkgFcn1_TInd = bkgFcn1
				# bkgFcn2_TInd = bkgFcn2
				#hgtD1 += measurement(0.0,0.0)
				#hgtD2 += measurement(0.0,0.0)
			# Now sum it up (must have signal over background!)
			#if (float(cts1 + dt1) + float(cts2 + dt2)) > (bkgFcn1 + bkgFcn2).val * (row['te']-row['ts']):
			#ctsSum += measurement(cts1 + cts2, np.sqrt((cts1 + cts2)*(d1c+d2c))) # Scale error by average PEs in coinc
			# if pmt1 and pmt2:
				# ctsSum += measurement(cts1 + cts2, np.sqrt(cts1*d1c + cts2*d2c)) # Scale error by average PEs in coinc
				# dtSum  += measurement(dt1 + dt2, np.sqrt(dt1*d1c + dt2*d2c))
				# #ctsSum += measurement(cts1 + cts2, np.sqrt(cts1 + cts2)) # Scale error by average PEs in coinc
				# #dtSum  += measurement(dt1 + dt2, np.sqrt(dt1 + dt2))
				# bkgSum += (bkgFcn1 + bkgFcn2) * measurement(row['te']-row['ts'],0.0)
				# bkgRaw += (bkgFcn1-bkgFcn1_TInd + bkgFcn2-bkgFcn2_TInd) * measurement(row['te']-row['ts'],0.0)
			# elif pmt1:
				# ctsSum += measurement(cts1, np.sqrt(cts1*d1c)) # Scale error by average PEs in coinc
				# dtSum  += measurement(dt1, np.sqrt(dt1*d1c))
				# #ctsSum += measurement(cts1, np.sqrt(cts1)) # Scale error by average PEs in coinc
				# #dtSum  += measurement(dt1, np.sqrt(dt1))
				# bkgSum += (bkgFcn1) * measurement(row['te']-row['ts'],0.0)
				# bkgRaw += (bkgFcn1-bkgFcn1_TInd) * measurement(row['te']-row['ts'],0.0)
			# elif pmt2:
				# ctsSum += measurement(cts2, np.sqrt(cts2*d2c)) # Scale error by average PEs in coinc
				# dtSum  += measurement(dt2, np.sqrt(dt2*d2c))
				# #ctsSum += measurement(cts2, np.sqrt(cts2c)) # Scale error by average PEs in coinc
				# #dtSum  += measurement(dt2, np.sqrt(dt2))
				# bkgSum += (bkgFcn2) * measurement(row['te']-row['ts'],0.0)
				# bkgRaw += (bkgFcn2-bkgFcn2_TInd) * measurement(row['te']-row['ts'],0.0)
		
			
			#ctsSum += measurement(cts1 + cts2, np.sqrt(cts1*d1c + cts2*d2c)) # Scale error by average PEs in coinc
			#dtSum  += measurement(dt1 + dt2, np.sqrt(dt1*d1c + dt2*d2c))
			ctsSum += measurement(cts1,np.sqrt(cts1*d1c))+measurement(cts2,np.sqrt(cts2*d2c)) # Scale error by average PEs in coinc
			dtSum  += measurement(dt1,np.sqrt(dt1*d1c))+measurement(dt2,np.sqrt(dt2*d2c))
			#dtSum  += measurement(dt1 + dt2, np.sqrt(dt1*d1c + dt2*d2c))
			bkgSum += (bkgFcn1 + bkgFcn2) * measurement(row['te']-row['ts'],0.0)
			#bkgSum += (bkgFcn1_TInd + bkgFcn2_TInd) * measurement(row['te']-row['ts'],0.0)
			bkgHD += (bkg1_Pos + bkg2_Pos) * measurement(row['te']-row['ts'],0.0)
			tCSum  += (bkg1_t + bkg2_t) * measurement(row['te']-row['ts'],0.0)
			#tCSum  += (bkgFcn1-bkgFcn1_TInd + bkgFcn2-bkgFcn2_TInd) * measurement(row['te']-row['ts'],0.0)
			
			
				#ctsSum += measurement(cts1 + cts2, np.sqrt(cts1 + cts2)) # Scale error by average PEs in coinc
				#dtSum  += measurement(dt1 + dt2, np.sqrt(dt1 + dt2))
			
			#elif pmt1:
			#	ctsSum += measurement(cts1, np.sqrt(cts1*d1c)) # Scale error by average PEs in coinc
			#	dtSum  += measurement(dt1, np.sqrt(dt1*d1c))
				#ctsSum += measurement(cts1, np.sqrt(cts1)) # Scale error by average PEs in coinc
				#dtSum  += measurement(dt1, np.sqrt(dt1))
			#	bkgSum += (bkgFcn1) * measurement(row['te']-row['ts'],0.0)
			#	bkgRaw += (bkgFcn1-bkgFcn1_TInd) * measurement(row['te']-row['ts'],0.0)
			#elif pmt2:
			#	ctsSum += measurement(cts2, np.sqrt(cts2*d2c)) # Scale error by average PEs in coinc
			#	dtSum  += measurement(dt2, np.sqrt(dt2*d2c))
				#ctsSum += measurement(cts2, np.sqrt(cts2c)) # Scale error by average PEs in coinc
				#dtSum  += measurement(dt2, np.sqrt(dt2))
			#	bkgSum += (bkgFcn2) * measurement(row['te']-row['ts'],0.0)
			#	bkgRaw += (bkgFcn2-bkgFcn2_TInd) * measurement(row['te']-row['ts'],0.0)
		
		#ctsSum.err = ctsSum.err*(np.sqrt(d1c+d2c))
		#dtSum.err  = dtSum.err *(np.sqrt(d1c+d2c))
			
		ctsSumL[i] = ctsSum
		dtSumL[i]  = dtSum
		bkgSumL[i] = bkgSum # Time Independent Background
		bkgHSumL[i] = bkgHD
		tCSumL[i]  = tCSum  # Background correction for time dependence		
		
	return ctsSumL, dtSumL, bkgSumL,bkgHSumL,tCSumL
	
#def sum_coinc_counts(ct, bkgC,bkgT = np.inf, maxUnl = 100, dBkg = True):
def sum_coinc_counts(cts,cfg, bkg):#bkgC = 0.0,bkgT = np.inf):
	# Sum all the dips that we plan on using, making corrections for backgrounds
	# Input a cts vector separated by dips/slices, a background vector, and a pseudo-bool
	# Returns measurements [ctsSum, dtSum, bkgSum]

	#-------------------------------------------------------------------
	hgt = map_height(cfg)
	#dBkg = False
	# Load output sums by dip
	ctsSumL = [] # Raw counts
	dtSumL  = [] # Deadtime Correction
	bkgSumL = [] # Background sum
	bkgHSumL = []
	tCSumL = []
	for i in range(0,cfg.ndips): # Doing this so that we can skip dips
		ctsSumL.append(measurement(0.0,0.0))
		dtSumL.append(measurement(0.0,0.0))
		bkgSumL.append(measurement(0.0,0.0))
		bkgHSumL.append(measurement(0.0,0.0))
		tCSumL.append(measurement(0.0,0.0))
	# Add all counts from the different dips.
	for i in range(0,cfg.ndips):

		ctsSum = measurement(0.0,0.0) # (Re-)Initialize tmp sums
		dtSum  = measurement(0.0,0.0)
		bkgSum = measurement(0.0,0.0)
		bkgHD  = measurement(0.0,0.0)
		tCSum  = measurement(0.0,0.0)
		
		ctsByDip = cts[cts['dip']==i] # Separate by dip
		for row in ctsByDip: # Now step through the dip slices
			
			if (not cfg.useMoving): # Ignore the first step when the dagger is moving
				if (row['ts'] == ctsByDip[0]['ts']):
					continue
			if (row['te'] - ctsByDip[0]['ts'] > cfg.maxUnl + 1): # Check max position
				break	
			
			sT = float(row['ts'] + row['te']) / 2.0 # mean step time
	
			ctsC = row['coinc']
			dtC  = row['dt']
	
			# Background (either position dependent or not). 
			# If a PMT is off, the bkg[i] should all be 0			
			bkgFcnC,bkgC_Pos,bkgC_T = bkgFunc_obj(bkg,0,hgt[i],sT)
			#bkgFcnC_Pos  = bkgFunc_obj(bkg,0,10.0,sT)
			#bkgFcnC_TInd = bkgFunc_obj(bkg,0,hgt[i],bkg.time)
			#if cfg.usePosBkgs: 
			#	bkgFcnC = bkgFunc(bkg.coinc,bkg.hDep,row['run'], 0, hgt[i], bkg.hgt, sT, bkg.time)
			#else:
			#	if bkgC > 0.0: # Assume poissonian error
			#		bkgFcnC = measurement(bkg.coinc,sqrt(bkg.coinc))
			#	else:
			#		bkgFcnC = measurement(0.0,0.0)

			if cfg.useDTCorr:
				#ctsSum += measurement(ctsC,np.sqrt(ctsC))
				ctsSum += measurement(ctsC,np.sqrt(ctsC+dtC))
			else:
				ctsSum += measurement(ctsC,np.sqrt(ctsC))
			if np.isfinite(dtC):
				dtSum  += measurement(dtC, 0)#np.sqrt(np.abs(dtC)))
			else: 
				print(row['run'], dtC)
				dtSum += measurement(0.,np.inf)
			bkgSum += (bkgFcnC) * measurement(float(row['te']-row['ts']),0.0)
			bkgHD  += (bkgC_Pos) * measurement(float(row['te']-row['ts']),0.0)
			tCSum  += (bkgC_T) * measurement(float(row['te']-row['ts']),0.0)
			
		ctsSumL[i]  = ctsSum
		dtSumL[i]   = dtSum
		bkgSumL[i]  = bkgSum
		bkgHSumL[i] = bkgHD
		tCSumL[i]   = tCSum
		
	return ctsSumL, dtSumL, bkgSumL,bkgHSumL,tCSumL

def find_mean_arrival(rRed,cts,bkg,cfg):
	# Calculation of mean arrival time
	# Returns a mean arrival time. 

	# First make sure heights and dips are right length
	hgt = map_height(cfg)
	mean = 0.0 # Initialize mean and weight counters
	wSum = 0.0
	rRed.len = 0.0
	for i in cfg.dips: # Separated out here by dip
		
		# Now step through the dip slices
		ctsByDip = cts[cts['dip']==i]
		for row in ctsByDip:
			if (not cfg.useMoving): # Ignore the first step when the dagger is moving
				if (row['ts'] == ctsByDip[0]['ts']):
					continue
			# Slicing data to give a maximum position
			if ((row['te'] - ctsByDip[0]['ts']) > (cfg.maxUnl + 1)):
				break
			rRed.len += row['te']-row['ts'] # Calculate length here
			# Mean step time
			sT = (row['ts'] + row['te']) / 2.0
			# Extract Background
			bkgFcn1,_x,_x = bkgFunc_obj(bkg,1,hgt[i],sT)
			bkgFcn2,_x,_x = bkgFunc_obj(bkg,2,hgt[i],sT)
			bkgFcnC,_x,_x = bkgFunc_obj(bkg,0,hgt[i],sT)
				
			# We actually need background counts, not rates
			bHits1 = bkgFcn1 * measurement(row['te']-row['ts'],0.0)
			bHits2 = bkgFcn2 * measurement(row['te']-row['ts'],0.0)
			bHitsC = bkgFcnC * measurement(row['te']-row['ts'],0.0)
			
			# Find percentages of counts for each PMT, and weight
			sT_M = measurement(sT-row['ts'],0.0) # Now shift the step time
			if cfg.sing: # Singles:
				if cfg.pmt1: # Do we use PMT 1?
					cts1 = measurement(row['d1'] , 0.)#np.sqrt(row['d1']*rRed.eff[0].val)) # Counts for dip
					dt1  = measurement(row['d1DT'],0.)#np.sqrt(row['d1DT']*rRed.eff[1].val)) # Deadtime correction for step
					sum1 = cts1 # Sum the total number of counts in this step
					sum1_w = cts1
					if cfg.useDTCorr:
						sum1 += dt1
						sum1_w += dt1
					if cfg.useBkgs:
						sum1 += bHits1
						sum1_w -=bHits1
					if row['m1'] > row['ts']: # Should always be positive (but this is a check)
						mt1 = measurement(row['m1']-row['ts'], 0.0) # Mean arrival time (scaled from beginning of unload)
						
						#mean1 = (mt1 * (cts1 + dt1) - (bHits1) * sT_M) / sum1 # Mean Arrival Time (offset from beginning)
						# Mean Arrival Time calculation
						mean1 = (mt1 * cts1) / sum1
						if cfg.useDTCorr:
							mean1 += (dt1 * mt1) / sum1
						if cfg.useBkgs:
							mean1 += (bHits1 * sT_M) / sum1
						#mean1 = (mt1 * (cts1) - (bHits1) * sT_M) / sum1 # Mean Arrival Time (offset from beginning)
					else:
						sum1  = measurement(0.0,0.0)
						sum1_w = measurement(0.0,0.0)
						mean1 = measurement(sT,np.inf)
				else: # Ignore PMT 1 here
					sum1  = measurement(0.0,0.0)
					sum1_w = measurement(0.0,0.0)
					mean1 = measurement(sT,np.inf)
				if cfg.pmt2: # Do we use PMT 2?
					cts2 = measurement(row['d2'],  0.)#np.sqrt(row['d2']*rRed.eff[1].val))
					dt2  = measurement(row['d2DT'],0.)#np.sqrt(row['d2DT']*rRed.eff[1].val))
					# Figure out how many total counts are in this step
					sum2 = cts2
					sum2_w = cts2
					if cfg.useDTCorr:
						sum2 += dt2
						sum2_w += dt2
					if cfg.useBkgs:
						sum2 += bHits2
						sum2_w -= bHits2
					#sum2  = cts2 - bHits2
					if row['m2'] > row['ts']: # Should always be positive (but this is a check)
						mt2  = measurement(row['m2']-row['ts'], 0.0)
						mean2 = (mt2 * cts2) / sum2
						if cfg.useDTCorr:
							mean2 += (dt2 * mt2) / sum2
						if cfg.useBkgs:
							mean2 += (bHits2 * sT_M) / sum2
						#mean2 = (mt2 * (cts2 + dt2) - (bHits2) * sT_M) / sum2
						#mean2 = (mt2 * (cts2) - (bHits2) * sT_M) / sum2
					else:
						sum2 = measurement(0.0,0.0)
						sum2_w = measurement(0.0,0.0)
						mean2 = measurement(sT,np.inf)
				else: # Ignore PMT 2 here
					sum2  = measurement(0.0,0.0)
					sum2_w = measurement(0.0,0.0)
					mean2 = measurement(sT,np.inf)
				# And shift back to row
				step1 = mean1 + measurement(row['ts'],0.0)			
				step2 = mean2 + measurement(row['ts'],0.0)	
				#step = ((mean1*sum1) + (mean2*sum2)) / (sum1 + sum2) + measurement(row['ts'], 0.0) # Step is MAT	
			else:	# Extract coincidence data
				ctsC = measurement(row['coinc'],0.)
				#ctsC = measurement(row['coinc'],np.sqrt(row['coinc']))
				dtC  = measurement(row['dt'],0.)
				#dtC  = measurement(row['dt'],np.sqrt(row['dt']))
				sumC = ctsC
				sum2C = ctsC
				if cfg.useDTCorr:
					sumC += dtC
					sum2C += dtC
				if cfg.useBkgs:
					sumC += bHitsC
					sum2C -= bHitsC
				#sumC  = ctsC + dtC - bHitsC
				m = measurement(row['m'],0.0)
				if row['m'] > row['ts']: # Must have a positive time
					mtC  = measurement(row['m']-row['ts'], 0.0)
					meanC = (mtC * ctsC) / sumC
					if cfg.useDTCorr:
						#meanC += (dtC * sT_M) / sumC
						meanC += (dtC * mtC) / sumC
					if cfg.useBkgs:
						meanC += (bHitsC * sT_M) / sumC
						#meanC -= (bHitsC * sT_M) / sumC
					#meanC = (mtC * (ctsC + dtC) - (bHitsC) * sT_M) / sumC
				else:
					meanC = measurement(sT-row['ts'],np.inf)
				step = meanC + measurement(row['ts'],0.0)
			
			#dpPct[row['dip']] += (pt1 + pt2 + ptC).val
			# And here do a weighted sum
			if cfg.sing:
				# Have to separate out weightings by PMT
				if step1.err != 0:
					#wgt1 = (1/step1.err**2)
					wgt1 = sum1_w.val#(1/step1.err**2)
				else:
					wgt1 = 0
				
				if step2.err != 0:
					#wgt2 = (1/step2.err**2)
					wgt2 = sum2_w.val#(1/step2.err**2)
				else:
					wgt2 = 0
				wSum += wgt1+wgt2
				mean += step1.val*wgt1 + step2.val*wgt2
			else: # Coincidence is easier
				#print(row['run'],row['m'],sT,step)
				if step.err != 0:
					#wgt = (1/step.err**2)
					wgt = (sum2C.val)
				else:
					wgt = 0
				wSum += wgt
				mean += step.val * wgt
	
	if wSum > 0: # Did we get counts in our dips?
		mean = mean / wSum
		rRed.mat = measurement(mean, 1/np.sqrt(wSum))
	else:
		rRed.mat = measurement(0.0,0.0)

	#print(rRed.mat)
	#return mat, dpPct
	return rRed

def normalization_reduced(rRed,rBreaks,cfg):
	# Functional form of normalizing with reduced runs
	#-------------------------------------------------------------------
	# Double check runBreak boundaries
	if rRed[0].run < rBreaks[0]:
		if cfg.vb:
			print("Warning: First element of runList is less than the first runBreak!")
			print("       I'll try to run this but something might be un-normalized!")
		rBreaks.append(rRed[0].run)
		rBreaks.sort()
	if rRed[-1].run > rBreaks[-1]:
		if cfg.vb:
			print("Warning: Last element of runList is more than the last runBreak!")
			print("        I'll try to run this but something might be un-normalized!")
		rBreaks.append(rRed[-1].run)
	
	# Create list of runs to normalize to
	runList = []
	indList = [] # For putting indices in the right places
	# cfg.hold can be set to 0 to normalize to all
	if cfg.hold > 0:
		# First get any runs with the right holding time
		holdList = []
		totalRuns = []
		totalInds = []
		totalHolds = []
		for i,r in enumerate(rRed):
			# Normalization list, +/- 1s slop:
			totalRuns.append(r.run)
			totalInds.append(i)
			totalHolds.append(int(round(r.hold)))
			if cfg.hold - 1 < r.hold < cfg.hold + 1:
				runList.append(r.run)
				indList.append(i)
			else:
				if int(round(r.hold)) not in holdList:
					holdList.append(int(round(r.hold)))
		totalRuns = np.array(totalRuns)
		totalInds = np.array(totalInds)
		totalHolds = np.array(totalHolds)
		holdList.sort() # sort minimum extra holds
		runList = np.array(runList)
		# If there are runs without the right holding time, deal with that now
		for b in range(len(rBreaks)): # Where are our runbreaks?
			if b == len(rBreaks) - 1: # End on last runBreak
				continue
			bCond = (rBreaks[b] <= runList)*(runList < rBreaks[b+1])
			rL = runList[bCond] # Which subset of runs is in the break?
			if len(rL) == 0: # OK, now we need to add additional runs.
			#if len(rL) < 2: # OK, now we need to add additional runs.	
				for h in holdList:
					#print("Adding additional Runs!")
					hCond = (totalHolds==h)
					bCondTmp = (rBreaks[b] <= totalRuns)*(totalRuns < rBreaks[b+1])
					newRuns = totalRuns[bCondTmp*hCond]
					newInds = totalInds[bCondTmp*hCond]
					if len(newRuns) >= 3:
						runList = np.array([])
						indList = np.array([])
						np.append(runList,newRuns)
						np.append(indList,newInds)
						break
					#else:
					#	print("Not enough to add!")
	else:
		for i,r in enumerate(rRed):
			# if cfg.hold is 0, just put everything in the list.
			runList.append(r.run)
			indList.append(i)
	runList = np.array(runList) # Cast to numpy arrays
	indList = np.array(indList)
	print(len(runList),len(indList),rBreaks)
	# Now that we've initialized counters, let's go and do things.
	prevAlpha = rRed[0].normalize_guess(cfg) # Initial guess for correlation
	prevBeta = measurement(0.0,0.0) # Set initial spectral correction at zero
	
	for i, r in enumerate(runList):
		wCond = (r - cfg.w <= runList)*(runList <= r+cfg.w) # Window condition
		for b in range(len(rBreaks)): # Where are our runbreaks?
			if b == len(rBreaks) - 1: # End on last runBreak
				continue
			if not (rBreaks[b] <= r < rBreaks[b+1]):
				continue
			bCond = (rBreaks[b] <= runList)*(runList < rBreaks[b+1])
			rL = runList[bCond*wCond] # Which subset of runs is in the break?
			iL = indList[bCond*wCond]
			if len(iL) > 2: # Did we get multiple runs to fit?
				m1 = [] # List of counts in monitor 1
				m2 = [] # List of counts in monitor 2
				unl  = [] # Unload counts
				unlE = [] 
				#runDeb = []
				for j in range(len(iL)): # Look at index 
				#	runDeb.append(rRed[iL[j]].run)
					m1.append(float(rRed[iL[j]].mon[0]))
					m2.append(float(rRed[iL[j]].mon[1]))
					unlM  = measurement(0.0,0.0)
					for d in cfg.normDips: # Normalization dips
						unlM += rRed[iL[j]].eff_cts(cfg)*rRed[iL[j]].pcts[d]
						#unlM += rRed[iL[j]].total_cts(cfg)*rRed[iL[j]].pcts[d].val
					unl.append(float(unlM))
					unlE.append(unlM.err)
				if float(prevAlpha) < 0:
					prevAlpha = measurement(0.,np.inf)
				#print(m1,m2,unl,unlE)
				co, cov = curve_fit(spectral_norm_meas_inv2, (m1,m2), unl,\
									sigma=unlE,absolute_sigma=True, \
									p0=(float(prevAlpha),float(prevBeta)), \
									bounds=([0.0,-np.inf],[np.inf,np.inf]))
				chisq = sum(np.power((spectral_norm_meas_inv2(np.array([m1,m2]),co[0],co[1]) -np.array(unl)),2) \
							/spectral_norm_meas_inv2(np.array([m1,m2]),co[0],co[1]))
				#			/(np.array(unlE)),2))
				chisq /= len(iL) - 2 # Chi-squared for scaling up errors
				rRed[indList[i]].alpha = co[0]
				rRed[indList[i]].beta  = co[1]
				rRed[indList[i]].nTime = rRed[indList[i]].hold
				if cfg.vb:
					if abs(cov[1][0] - cov[0][1]) > 1e-7: # Check on 10^-8 precision
						print("WARNING, run %d has poorly defined covariance %f" % \
								(rRed[i].run,cov[1][0]-cov[0][1]))
				rRed[indList[i]].alphaE = np.sqrt(cov[0][0]/chisq) # This is more complicated than it
				rRed[indList[i]].betaE  = np.sqrt(cov[1][1]/chisq) # needs to be: absolute_sigma=False			
				rRed[indList[i]].cov = cov[1][0]/chisq			   # would give us the same thing.
				# Return prevAlpha and prevBeta as a guess for later:
				prevAlpha = measurement(co[0],np.sqrt(cov[0][0]/chisq))
				prevBeta  = measurement(co[1],np.sqrt(cov[1][1]/chisq))
			elif len(iL) == 2: # Can't curve_fit sparse matrices
				if cfg.vb:
					print("Unconstrained curve_fit for run %d!" % r)
				unl1 = measurement(0.0,0.0)
				unl2 = measurement(0.0,0.0)
				m1 = [rRed[iL[0]].mon[0],rRed[iL[1]].mon[0]] # Monitor
				#m2 = [rRed[iL[0]].mon[1]*rRed[iL[0]].mon[1] / rRed[iL[0]].mon[0], \
				#	  rRed[iL[1]].mon[1]*rRed[iL[1]].mon[1] / rRed[iL[1]].mon[0]] # Spectrum
				m2 = [rRed[iL[0]].mon[0]*rRed[iL[0]].mon[0] / rRed[iL[0]].mon[1], \
					  rRed[iL[1]].mon[0]*rRed[iL[1]].mon[0] / rRed[iL[1]].mon[1]] # Spectrum
				for d in cfg.normDips: # Sum up the unloads by dips
					#unl1 += rRed[iL[0]].total_cts(cfg)*rRed[iL[0]].pcts[d].val
					#unl2 += rRed[iL[1]].total_cts(cfg)*rRed[iL[1]].pcts[d].val
					unl1 += rRed[iL[0]].total_cts(cfg)*rRed[iL[0]].pcts[d]
					unl2 += rRed[iL[1]].total_cts(cfg)*rRed[iL[1]].pcts[d]
				# With 2 unloads, the assumption is just linear fit
				alpha = (unl1 - unl2 * (m2[0]/m2[1])) / \
						(m1[0] - m1[1] * (m2[0]/m2[1]))
				beta  = (unl1 - alpha*m1[0])/m2[0]
				#print(alpha,beta)
				rRed[indList[i]].alpha = alpha.val
				rRed[indList[i]].beta  = beta.val
				rRed[indList[i]].alphaE = alpha.err
				rRed[indList[i]].betaE  = beta.err
				rRed[indList[i]].cov    = 0.0 # No covariance for a line
				# Return prevAlpha and prevBeta as a guess for later
				prevAlpha = alpha
				prevBeta  = beta
			elif len(iL) == 1: # We're assuming that there's one index
				if cfg.vb: 
					print("Only one run in window for run %d!" % rRed[i].run)
				unl1 = measurement(0.0,0.0)
				m1 = rRed[i].mon[0]
				m2 = rRed[i].mon[0]*rRed[i].mon[0]/rRed[i].mon[1]
				for d in cfg.normDips:
					unl1 += rRed[i].total_cts(cfg)*rRed[i].pcts[d]
					#unl1 += rRed[i].eff_cts(cfg)*rRed[i].pcts[d].val
				alpha = (unl1 - rRed[i].mon[1] * prevBeta) / rRed[i].mon[0]
				rRed[indList[i]].alpha  = alpha.val
				rRed[indList[i]].alphaE = alpha.err
				rRed[indList[i]].beta   = prevBeta.val
				rRed[indList[i]].betaE  = prevBeta.err
				rRed[indList[i]].cov = 0.0
			else:
				if cfg.vb:
					print("No runs in window for run %d!" % rRed[i].run)
				rRed[indList[i]].alpha  = prevAlpha.val
				rRed[indList[i]].alphaE = np.inf
				rRed[indList[i]].beta   = prevBeta.val
				rRed[indList[i]].betaE  = np.inf
				rRed[indList[i]].cov = 0.0
				
	return rRed

	
# #-----------------------------------------------------------------------
# # TODO Extracted Value Functions:
# def peak_subtraction(cts):
	# # If we want to extend a dip exponentially outwards
	# # Don't try and do this if we're moving counts later though.
	
	# # Think before doing this...
	# cts_corr = cts
	
	
	
	# return cts_corr

# #-----------------------------------------------------------------------
# # Old calculation subroutines
# def analyze_single_norm_condensed(runList,cts,nMon,nMad,runBreaks = [],nDet1 = 4, nDet2 = 8,dips = range(0,3),pmt1=True,pmt2=True): # Main function here
	# expoNorm = True
	# geomNorm = False

	# if (runBreaks == []):
		# runBreaks = [runList[0],max(runList)]
	
	# normedRun, timeL,timeM, ctsL, monL, dC,dtCts,bSubVec,_N,_N,_N= extract_values(runList, cts, nMon, nDet1, nDet2, 20, pmt1, pmt2,dips,runBreaks)
	
	# nCoeff,coErr = normalization_params(normedRun,monL,ctsL,runBreaks,9999)
	# nFac1, nFac2,normRuns,nCorr = normalize_counts_by_nvec(runList, cts, nMon, nDet1, nDet2, normedRun,nCoeff,coErr,runBreaks)
	
	# runNum, holdVec, meanArr, rawCts, rawMon, dC,dtCts,bSubVec,_N,_N,pctDip = extract_values(normRuns, cts, nMon, nDet1, nDet2, 0, pmt1, pmt2,dips,runBreaks)
	
	# nCtsVec = []
	# normVec = []
	# for i, c in enumerate(rawCts):
		# nCtsVec.append(c / (nCorr[i]))
		# normVec.append(nCorr[i])
	
	# nPhiVec = []
	# pct2 = []
	# pct3 = []
	# for dip in pctDip:
		# pct2.append(dip[1])
		# pct3.append(dip[2])
	# sig2noi = []
	# #write_summed_counts(runNum,holdVec,rawCts,rawMon)
	# print("Managed to successfully normalize "+str(len(runNum))+" runs! This is a ratio of: "+str(float(len(runNum))/float(len(runList))))
	# print(" ")
	# if len(runNum) < 8:
		# sys.exit("Error! Unable to normalize an entire octet's worth of runs! Exiting...")
	# # lol i'm just returning a bunch of stuff
	# return runNum,holdVec,nCtsVec,meanArr,rawCts,bSubVec,normVec,nFac1,nFac2,nPhiVec,pct2,pct3,dC

# def normalization_params(rList = [], mon = [], det = [], breaks = [], wt = 3, vb = True):
	# #-----------------------------------------------------------------------
	# # Functional form of normalizing, given 2 weighted monitor
	# #-----------------------------------------------------------------------
	# # Input parameters are vectors (except "window" w):
	# # rList is the list of normalization runs (for troubleshooting)
	# # mon is a 2xN list of weighted monitor signals -- can be "measurement" or double
	# # det is a list of the counts in our given detector -- can be "measurement" or double
	# # breaks is our runbreaks list  
	# #
	# # np.linalg.lstsq(a,b) solves the equation a x = b by computing a vector x
	# # that minimizes the equation ||b-ax||^2.
	# #
	# # Returns solution[0], resiudual[1], rank[2], and min value of a[3]
	# #
	# #-----------------------------------------------------------------------

	# # Initialize counters and outputs
	# nC = [] # Coeffecients (output)
	# err = [] # Residuals (also output) [element 2 is the covariance]
	
	# bC = 0 # Break Counter
	# m = 0 # Window counter (low)
	# n = 0 # Window counter (high)
	
	# # Error checking----------------------------------------------------
	# # Check that we have the right number of inputs
	# if len(rList) != len(det) or len(rList) != len(mon):
		# print("Error: Mismatched normalization size! Returning...")
		# print (str(len(rList))+str(len(det))+str(len(mon)))
		# return nC, err
	# if len(breaks) < 2:
		# print("Error: Number of runBreaks must be 2 or more! (At least first and last)! Returning...")
		# return nC, err
	
	# # Double check runBreak boundaries
	# if rList[0] < breaks[0] and vb:
		# print("Warning: First element of runList is less than the first runBreak!")
		# print("       I'll try to run this but something might be un-normalized!")
		# breaks.append(rList[0])
		# breaks.sort()
	# if max(rList) > max(breaks) and vb:
		# print("Warning: Last element of runList is more than the last runBreak!")
		# print("        I'll try to run this but something might be un-normalized!")
		# breaks.append(max(rList))
			
	# if type(det[0]) == type(mon[0][0]):
		# if isinstance(det[0],measurement): # Combined for calculating errors via black box or explicitly
			# meas = True
		# else:
			# meas = False
	# else:
		# print("Error! Monitor and Detector counts not same datatype!")
		# return nC, err
	# #-------------------------------------------------------------------
		
	# if meas: # Previous value of alpha (counting) and beta (spectral correction term)
		# prevAlpha = det[0]/mon[0][0]
		# prevBeta = measurement(0.0,0.0) # Set initial spectral correction at zero
	# else:
		# prevAlpha = float(det[0]/mon[0][0])
		# prevBeta = 0.0 
					
	# # Loop through the normalization runlist
	# for i, r in enumerate(rList):
		
		# # First, figure out which runBreaks space we're in and adjust bC.
		# if 0 <= bC < len(breaks)-1: 
			# bmi = breaks[bC]
			# bma = breaks[bC + 1]
			
			# # Check if our present run is greater than the next runBreak
			# if (r >= bma):
				# # Start a loop to increase bC
				# while (r >= bma):
					# # Make sure we don't increment too far!
					# if bC == len(breaks) - 2:
						# if vb:
							# print("This is the maximum runBreak, proceeding!")
						# bmi = breaks[bC]
						# bma = breaks[bC + 1]
						# break
					# bC += 1
					# bmi = breaks[bC]
					# bma = breaks[bC + 1]
			# if not (bmi <= r < bma):
				# if r == bma and bC == len(breaks) - 2: # Last possible break
					# alpha = (det[i] - prevBeta*mon[i][1]) / mon[i][0]
					# if meas:
						# nC.append([alpha.val,prevBeta.val])
						# err.append([alpha.err,prevBeta.err,0.0])
					# else:
						# nC.append([float(alpha),float(prevBeta)])
						# if mon[i][0]*det[i] > 0: # Error calculation -- make sure it's real
							# alpha_err = float(np.sqrt(mon[i][0]+det[i])/np.sqrt(mon[i][0]*det[i]))
						# else:
							# alpha_err = np.inf
						# if mon[i][0]*mon[i][1] > 0: # Require counts in both monitors
							# beta_err = float(np.sqrt(mon[i][0]+mon[i][1])/np.sqrt(mon[i][0]*mon[i][1]))
						# else:
							# beta_err = np.inf
						# err.append([alpha_err, beta_err,0.0])
									
					# bC -=1
					# prevAlpha = alpha
					# continue					
				# else:
					# if vb:
						# print("Unable to normalize run", r, "with rolling normalization!")
						# print(bmi,bma)
					# alpha = (det[i] - prevBeta*mon[i][1]) / mon[i][0]
										
					# if meas:
						# nC.append([alpha.val,prevBeta.val])
						# err.append([alpha.err,prevBeta.err,0.0])
					# else:
						# nC.append([float(alpha),float(prevBeta)])
						# if mon[i][0]*det[i] > 0: # Error calculation -- make sure it's real
							# alpha_err = float(np.sqrt(mon[i][0]+det[i])/np.sqrt(mon[i][0]*det[i]))
						# else:
							# alpha_err = np.inf
						# if mon[i][0]*mon[i][1] > 0:
							# beta_err = float(np.sqrt(mon[i][0]+mon[i][1])/np.sqrt(mon[i][0]*mon[i][1]))
						# else:
							# beta_err = np.inf
						# err.append([alpha_err, beta_err,0.0])
					# prevAlpha = alpha
					# continue
					
		# # Make sure max run [i+wt] is callable i.e. [i+len(rList) - (i+1)] = [len(rList)-1]
		# if i + wt < len(rList): 
			# n = wt
		# else:
			# n = len(rList) - (1 + i)
		# # Make sure min run [i-wt] is callable, i.e. [i-i] = [0]
		# if i >= wt: 
			# m = wt
		# else:
			# m = i
			
		# # Get the upper bound
		# if rList[i+n] >= bma:
			# while (rList[i+n] >= bma):
				# n-=1
			# if n < 0:
				# if vb:
					# print("Somehow have a negative window on run "+str(r))
					# print(i,n,bmi,bma)
				# while breaks[bC+1] <= rList[i+1]: 
					# bC += 1
				# rList.pop(i)
				# mon.pop(i)
				# det.pop(i)
		# # Get the lower bound
		# if rList[i-m] < bmi:
			# while (rList[i-m] < bmi):
				# m-=1
			# if m < 0:
				# if vb:
					# print("Somehow have a negative minimum on run "+str(r))
					# print(i,n,bmi,bma)
				# rList.pop(i)
				# mon.pop(i)
				# det.pop(i)
		
		# # case of just one normalization run in bounds
		# if m == 0 and n == 0:
		
			# alpha = (det[i] - mon[i][1]*prevBeta) / mon[i][0]
			# if meas:
				# nC.append([alpha.val,prevBeta.val])
				# err.append([alpha.err,prevBeta.err,0.0])
			# else:
				# nC.append([float(alpha),float(prevBeta)])
				
				# if mon[i][0]*det[i] > 0: # Error calculation -- make sure it's real
					# alpha_err = float(np.sqrt(mon[i][0]+det[i])/np.sqrt(mon[i][0]*det[i]))
				# else:
					# alpha_err = np.inf
				# if mon[i][0]*mon[i][1] > 0:
					# beta_err = float(np.sqrt(mon[i][0]+mon[i][1])/np.sqrt(mon[i][0]*mon[i][1]))
				# else:
					# beta_err = np.inf
				# err.append([alpha_err, beta_err,0.0])
			# prevAlpha = alpha		
		# # shift window to account for edge effects
		# elif m == 0 or n == 0:
			# if m == 0:
				# # If we can call n+w, do it. Otherwise just use max n
				# if i + (n+wt) < len(rList): 
					# n += wt
				# else:
					# n = len(rList) - (1 + i)
				# if rList[i+n] >= bma:
					# while (rList[i+n] >= bma):
						# n-=1
				# if n < 0:
					# if vb:
						# print("Somehow have a negative window on run "+str(r))
						# print(i,n,bmi,bma)
					# while breaks[bC+1] <= rList[i+1]: 
						# bC += 1
					# rList.pop(i)
					# mon.pop(i)
					# det.pop(i)
			# elif n == 0:	
				# # If we can call m+w, do it. Otherwise use max m
				# if i >= (m+wt): 
					# m += wt
				# else:
					# m = i
				# if rList[i-m] < bmi:
					# while (rList[i-m] < bmi):
						# m-=1
				# if m < 0:
					# if vb:
						# print("Somehow have a negative minimum on run "+str(r))
						# print(i,n,bmi,bma)
					# rList.pop(i)
					# mon.pop(i)
					# det.pop(i)

		# # Now calc. rolling norm
		# if m > 0 or n > 0:
			
			# m1 = []
			# m2 = []
			# val = []
			# verr= [] # This remains empty if no meas
			
			# if m + n + 1 > 2: # Curve fit doesn't like sparse matrices
				# for x in range(i-m,i+n+1):
					# #m1.append(float(mon[x][0]))
					# #m2.append(float(mon[x][1]))
					# if meas: # I guess I have to do a full divide-by-zero safety check here...
						# if mon[x][0].val > 0: # Make sure we're weighting by real values
							# rE1 = (mon[x][0].err / mon[x][0].val)**2
						# else: # If there's no counts in the monitor, we can skip that monitor for weighting
							# rE1 = 0.0
						# if mon[x][1].val > 0:
							# rE2 = (mon[x][1].err / mon[x][1].val)**2
						# else: 
							# rE2 = 0.0
						# if rE1 + rE2 > 0: # Now we should have monitor counts and errors
							# weight = 1.0/(rE1+rE2)
						# else: # If not, set weighting at zero since we don't know what's happening.
							# weight = 0.0 
					
					# else:
						# weight = 1.0
					# if weight == 0: 
						# print("Warning! Weighting is Set To Zero!")
					# if weight > 0:
						# weight = 1.0
						# m1.append(float(mon[x][0])*weight)
						# m2.append(float(mon[x][1]/mon[x][0])*weight)
						# if meas:
							# val.append(float(det[x].val)*weight)
							# #verr.append(float(det[x].err)*weight)
							# verr.append(1.0)#float(det[x].err)*weight)
						# else:
							# val.append(float(det[x]))
					
				# #-------------------------------------------------------
				# if float(mon[i][0]) > 0.0: # Make sure we have a rational starting spot
					# if meas:
						# try:
							# co, cov = curve_fit(spectral_norm_meas, (m1,m2), val,sigma=verr,absolute_sigma=True, p0=(float(det[i]/mon[i][0]),0.0),bounds=([0.0,-np.inf],[np.inf,np.inf]))
							# #co, cov = curve_fit(spectral_norm_meas, (m1,m2), val, p0=(float(det[i]/mon[i][0]),0.0),bounds=([0.0,-np.inf],[np.inf,np.inf]))
						# except ValueError:
							# co = [float(det[i]/mon[i][0]),0.0]
							# cov = [[np.inf,np.inf],[np.inf,np.inf]]
							# #print (str(m1)+str(m2)+str(val)+str(verr))
						# #co, cov = curve_fit(spectral_norm_meas, (m1,m2), val,sigma=verr,absolute_sigma=False, p0=(float(prevAlpha),float(prevBeta)),bounds=([0.0,-np.inf],[np.inf,np.inf]))
					# else:
						# co, cov = curve_fit(spectral_norm, (m1,m2), val, p0=(float(prevAlpha),float(prevBeta)),bounds=([0.0,-np.inf],[np.inf,np.inf]))
				# elif float(mon[i][1]) > 0.0: # Try using the secondary detector
					# if meas:
						# co, cov = curve_fit(spectral_norm_meas, (m1,m2), val,sigma=verr,absolute_sigma=False, p0=(0.0,float((det[i]/mon[i][1]))),bounds=([0.0,-np.inf],[np.inf,np.inf]))
					# else:
						# co, cov = curve_fit(spectral_norm, (m1,m2), val, p0=(0.0,float(det[i]/mon[i][1])),bounds=([0.0,-np.inf],[np.inf,np.inf]))
				# else: # Uh, just skip it and move on?
					# print("Error: Can't normalize to an empty bin!")
					# break
				# if not np.isfinite(np.diag(cov)[0]):
					# print("Error! Covariance matrix is infinite for run", r,"!")
					# #print(m1,m2,val)
				
				# #print(co,cov)
				# nC.append(co)
				# #print(co)
				# # Chi squared correlated fit:
				# #print spectral_norm_meas([m1,m2],co[0],co[1])
				# #print val
				# chisq = sum(((spectral_norm_meas(np.array([m1,m2]),co[0],co[1]) -np.array(val))/np.array(verr))**2)
				# chisq /= m+n-1
				# if meas:
					# prevAlpha = measurement(co[0],np.sqrt(np.diag(cov)[0])/chisq)
					# prevBeta = measurement(co[1],np.sqrt(np.diag(cov)[1])/chisq)
				# else:
					# prevAlpha = co[0]
					# prevBeta = co[1]
				# if abs(cov[1][0] - cov[0][1]) > 1e-8: # Arbitrarily choosing 10^-8 precision. Python seems to be only sometimes doing double bit precision (10^-16). 
					# print("Warning, poorly defined covariance! "+str(cov[1][0] - cov[0][1]))
							
				# if (np.diag(cov)[0] >= 0 and np.diag(cov)[1] >= 0) and not (m+n+1) == 0:
					# #err.append(np.sqrt(np.diag(cov))/chisq)#(m+n+1.0))
					# err.append([np.sqrt(cov[0][0]/chisq), np.sqrt(cov[1][1]/chisq), cov[1][0]/chisq])
				# else:
					# err.append([np.inf,np.inf,0.0])
			# else: # Two equations, two unknowns
				# #print "Unconstrained curve_fit on run", r
				# prevAlpha  = ((det[i+n] - det[i-m] * (mon[i+n][1]/mon[i-m][1])) 
							# / (mon[i+n][0] - mon[i-m][0] * (mon[i+n][1]/mon[i-m][1])))
				# prevBeta   = (det[i-m] - prevAlpha*mon[i-m][0])/mon[i-m][1]
				# nC.append([prevAlpha.val, prevBeta.val])
				# err.append([prevAlpha.err,prevBeta.err,0.0])
			# #nC.append(np.linalg.lstsq(mon[i-m:i+n],det[i-m:i+n])[0])
			# #err.append(r)
			# #singV.append(sing)
	# #print ""
	# #print nC1
	# #print ""
	# #print err
	# #print singV
	# #plt.figure(1)
	# #print len(rList), len(nC),len(err)
	
	# #-------------------------------------------------------------------
	# #c1 = []
	# #e1 = []
	# #c2 = []
	# #e2 = []
	# #-------------------------------------------------------------------
	# #c1 = []
	# #c2 = []
	# #-------------------------------------------------------------------
	# #plt.figure(777)
	# #for i, coeff in enumerate(nC):
		# #print coeff[0],coeff[1]
		# #---------------------------------------------------------------
	# #	c1.append(coeff[0])
	# #	e1.append(err[i][0])
	# #	c2.append(coeff[1])
	# #	e2.append(err[i][0])
		# #---------------------------------------------------------------
	# #	c1.append(coeff[0])
	# #	c2.append(coeff[1])
		# #---------------------------------------------------------------
		
		# #c1.extend(coeff[0])
		# #c2.extend(coeff[1])
	# #	plt.plot(rList[i],coeff[0],'b.')
	# #	plt.plot(rList[i],coeff[1],'r.')
		# #print coeff[0],coeff[1]
		
	# #-------------------------------------------------------------------
	# #plt.plot(rList,c1,color='b')
	# #plt.plot(rList,c2,color='r')
	# #plt.errorbar(rList,c1,yerr=e1,color='b')
	# #plt.errorbar(rList,c2,yerr=e2,color='r')
	# #-------------------------------------------------------------------
	# # ~ plt.plot(rList,c1,color='b')
	# #plt.plot(rList,c2,color='r')
	# #-------------------------------------------------------------------
	
	
		
	# #for xl in breaks:
# #		plt.axvline(x=xl)
	# #plt.show()	
	# #plt.figure(2)
	# #residuals=[]
	# #for i, rS in enumerate(err):
	# #	residuals.extend(rS)
	# #	if not len(rS) == 1:
	# #		residuals.extend([np.sqrt(det[i])])
	# #plt.plot(rList,residuals,color='g')
	# #for xl in breaks:
	# #	plt.axvline(x=xl)
	# #for i, coeff in enumerate(nC):
		
	# #for xl in breaks:
		
	# #	plt.errorbar(rList[i],coeff[1], yerr=err[i], fmt='b.')
	# #plt.show()
	# return nC, err

# def normalize_counts_by_nvec(rL, cts, nMon, m1=4, m2=8, nList=[],nC=[],nCE=[],runB=[],vB = True):
	# # This runs through and normalizes our counts on a run-to-run basis

	# # Set up initializations as error catchers
	# if nList == []: # If we don't actually do a spectral correction, assume cts/m1 for all.
		# nList = [min(runList),max(runList)] # Both default to 2
	# if nC == []:
	# #nC = []
		# for n in nList:
			# nC.append([1.0,0.0])
	# if nCE == []:
	# #nCE = []
		# for n in nList:
			# nCE.append([0.0,0.0,0.0])
	# if runB == []:
		# runB  = [min(rL),max(rL)]
			
	# # Load Norm Monitor names
	# m1S  = ('mon'+str(m1))
	# m1ES = ('mon'+str(m1)+'E')
	# m2S  = ('mon'+str(m2))
	# m2ES = ('mon'+str(m2)+'E')
			
	# # Indices for calculating normalizations
	# nInd = 0
	# bInd = 0
		
	# # Output normalization values
	# nFac1 = []
	# nFac2 = []
	# normRuns = []
	# nFacCorr = [] # Normalization, assuming correlation
	# #rTest1 = []
	# #rTest2 = []
	# # Loop through and determine normalization/spectrum dependence for each run.
	# for i,run in enumerate(rL):
		# #if run in nList:
		# #	continue	
		# # If we're past the next bCount we want to skip this 
		# if nInd < len(nList)-1 and bInd < len(runB)-1: # Make sure we're inside the right bands
			
			# if run >= runB[bInd+1]: # If we passed a run break, find the next applicable indices
				# while run >= runB[bInd+1]:
					# bInd+=1 # find the next applicable break indices
					# if bInd == len(runB)-1: # Make sure we're not on the last run
						# break
			# if run >= nList[nInd+1]: # We're planning on looking between 2 nLists
				# while run >= nList[nInd+1]: # We also would've passed to another nList member
					# nInd+=1
					# if nInd == len(nList)-1: # Make sure we're not on the last run
						# break
			# if nList[nInd] < runB[bInd]: # And make sure we're inside the right runB
				# while nList[nInd] < runB[bInd]:
					# nInd+=1
					# if nInd == len(nList) - 1:
						# break
		# if nInd < len(nList)-1 and bInd < len(runB)-1: # Make sure we're inside the right bands
			# if (abs(run - nList[nInd]) > abs(run - nList[nInd+1]) # are we closer to another index?
					# and nList[nInd+1] < runB[bInd+1]): # is this closer index not across the break?
				# while (abs(run - nList[nInd]) > abs(run - nList[nInd+1])
						# and nList[nInd+1] <= runB[bInd+1]):
					# if i > 0:
						# if (abs(run - nList[nInd+1]) > 8 and (run - rL[i-1]) < 4): 
							# break # If we're an octet away from normalization, but still well inside an octet, don't bother
					# nInd += 1
					# if nInd == len(nList) - 1: # Catch to make sure we get out on time
						# break			
	
		# if nInd > len(nC)-1: # Error catching
			# while nInd > len(nC) - 1: 
				# nInd -= 1
		# if bInd > len(runB)-1:	
			# bInd -= 1
		# #rTest1.append(run)
		# #rTest2.append(nList[nInd])
		# # Calculate normalization factor
		# nMonRaw = nMon[nMon['run']==run]
		# if len(nMonRaw) != 1:
			# if vb:
				# print("Warning! Unable to properly normalize run "+str(run)+" due to double counting!")
			# nFac1.append(measurement(nMonRaw[m1S][0],nMonRaw[m1ES][0])*measurement(nC[nInd][0],nCE[nInd][0]))
			# #nFac2.append(measurement(nMonRaw[m2S][0],nMonRaw[m2ES][0])*measurement(nC[nInd][0],nCE[nInd][0])*measurement(nC[nInd][1],nCE[nInd][1]))
			# nFac2.append(measurement(nMonRaw[m2S][0],nMonRaw[m2ES][0])*measurement(nC[nInd][1],nCE[nInd][1]))
		
		# if nCE[nInd][0] == np.inf or nCE[nInd][1] == np.inf:
			# print("Run "+str(run)+" has infinite monitor error!")
		# nFac1.append(measurement(nMonRaw[m1S],nMonRaw[m1ES])*measurement(nC[nInd][0],nCE[nInd][0]))
		# #nFac2.append(measurement(nMonRaw[m2S],nMonRaw[m2ES])*measurement(nC[nInd][0],nCE[nInd][0])*measurement(nC[nInd][1],nCE[nInd][1]))
		# #nFac2.append(measurement(nMonRaw[m2S],nMonRaw[m2ES])
		# #			/measurement(nMonRaw[m1S],nMonRaw[m1ES])*measurement(nC[nInd][1],nCE[nInd][1]))
		# nFac2.append(measurement(nMonRaw[m2S],nMonRaw[m2ES])*measurement(nC[nInd][1],nCE[nInd][1]))
		# # Correlated values
		# #nVal = ( measurement(nMonRaw[m1S],nMonRaw[m1ES]) * nC[nInd][0] # a*mon1
		# #		+ measurement(nMonRaw[m2S],nMonRaw[m2ES]) * nC[nInd][1] ) # b*mon2
		# nVal = ( measurement(nMonRaw[m1S],nMonRaw[m1ES]) * nC[nInd][0] # a*mon1
				# + measurement(nMonRaw[m2S],nMonRaw[m2ES])/measurement(nMonRaw[m1S],nMonRaw[m1ES]) * nC[nInd][1] ) # b*mon2
				
		# #nVal = ( measurement(nMonRaw[m1S],0.0) * nC[nInd][0] # a*mon1
		# #		+ measurement(nMonRaw[m2S],0.0) * nC[nInd][1] ) # b*mon2
		# nErr = np.sqrt((nVal.err*nVal.err) # Error from monitors
				# +(nMonRaw[m1S]*nMonRaw[m1S]*nCE[nInd][0]*nCE[nInd][0]
		# #		+ nMonRaw[m2S]*nMonRaw[m2S]*nCE[nInd][1]*nCE[nInd][1] # Error from uncorrelated
				# + (nMonRaw[m2S]*nMonRaw[m2S])/(nMonRaw[m1S]*nMonRaw[m1S])*nCE[nInd][1]*nCE[nInd][1] # Error from uncorrelated
		# #	+ 2*nMonRaw[m1S]*nMonRaw[m2S]*nCE[nInd][2])) # Correlation between monitors error
				# + 2*nMonRaw[m2S]*nCE[nInd][2])) # Correlation between monitors error
		# nFacCorr.append(measurement(nVal.val,nErr))
		# normRuns.append(run)

	# return nFac1, nFac2, normRuns, nFacCorr

# # Multivariate stuff that should be moved to a different file:

# #-----------------------------------------------------------------------
# # This quick ratio thing was sorta useful, but not really.

# # def take_ratio_of_counts(monIn):
	# # # This will shift normalization_params to be ratio of counts instead 
	# # # of a linear combination.
	
	# # monOut = []
	# # if isinstance(monIn[0][0],measurement):
		# # for inp in monIn:
			# # if inp[0].val > 0:
				# # monOut.append([inp[0],inp[1]/inp[0]])
			# # else:
				# # monOut.append([inp[0],measurement(0.0,0.0)])
	# # else:
		# # for inp in monIn:
			# # if inp[0].val > 0:
				# # monOut.append([inp[0],inp[1]/inp[0]])
			# # else:
				# # monOut.append([inp[0],measurement(0.0,0.0)])
			
	# # return monOut
# #-----------------------------------------------------------------------

# #-----------------------------------------------------------------------
# # Below this line is trash
# #-----------------------------------------------------------------------
# # def sum_all_dips(ct, bkgs, aNorm = False,maxUnl = 100):
	# # # Sum all the dips that we plan on using, making corrections for backgrounds
	# # # Input a cts vector separated by dips/slices, a background vector, and a pseudo-bool
	# # # Returns measurements [ctsSum, dtSum, bkgSum]
	
	# # #-------------------------------------------------------------------
	# # # Remove tese things later
	# # dip1 = True
	# # dip2 = True
	# # dip3 = True
	# # pmt1 = False
	# # pmt2 = False
	# # singLT = False
	# # coincLT = True
	# # usePosBkgs = True
	# # #-------------------------------------------------------------------
	
	# # # Load output sums
	# # ctsSum = measurement(0.0, 0.0) # Raw counts
	# # dtSum  = measurement(0.0, 0.0) # Deadtime Correction
	# # bkgSum = measurement(0.0, 0.0) # Background sum
	
	# # # Figure out the dagger height
	# # nDips = max(ct['dip']) # Note that this is zero indexed
	# # hMap = {1:[10.0], 3:[380.0, 250.0, 10.0], 9:[380.0, 250.0, 180.0, 140.0, 110.0, 80.0, 60.0, 40.0, 10.0]}
	# # if ((nDips == 0) or (nDips == 2) or (nDips == 8)):
		# # hgt = hMap[nDips+1]
	# # #else:
	# # #	print "Run is not a normal (production) 3-dip run!"
	# # #	return [ctsSum, dtSum, bkgSum]	
	
	# # # Make sure we're turning on the right number of dips
	# # if (nDips != 2):
		# # print("Run is not a normal (production) 3-dip run!")
		# # return [ctsSum, dtSum, bkgSum]
		
	# # # Add all counts from the different dips.
	# # for i in range(0,nDips+1):
		
		# # # Separate out runs by dip. Forcing 3-dip toggles.
		# # if (nDips == 2) and not aNorm:
			# # if (i == 0) and (dip1 == False):
				# # continue
			# # if (i == 1) and (dip2 == False):
				# # continue
			# # if (i == 2) and (dip3 == False):
				# # continue
		# # ctsByDip = ct[ct['dip']==i]
		
		# # # Now step through the dip slices
		# # for row in ctsByDip:
			# # # Slicing data to give a maximum position
			# # if (row['te'] - ctsByDip[0]['ts'] > maxUnl + 1):
				# # break
			
			# # sT = (row['ts'] + row['te']) / 2.0 # mean step time
			# # # Find the yield and background for the given dip for each PMT
			
			# # # Singles
			# # if (pmt1 == True):
				# # cts1 = row['d1'] # Counts for dip
				# # dt1  = row['d1DT'] # Deadtime correction for step
			# # else:
				# # cts1    = 0.0
				# # dt1     = 0.0
			# # if (pmt2 == True):
				# # cts2 = row['d2']
				# # dt2  = row['d2DT']
			# # else:
				# # cts2 = 0.0
				# # dt2  = 0.0
			# # # Coincidence
			# # if (coincLT == True):
				# # ctsC = row['coinc']
				# # dtC  = row['dt']
			# # else:
				# # ctsC = 0.0
				# # dtC  = 0.0

			# # # Background (either position dependent or not). 
			# # # If a PMT is off, the bkg[i] should all be 0
			# # if usePosBkgs: 
				# # bkgFcn1 = bkgFunc(bkgs[0], row['run'], 1, hgt[i], 10.0, sT, bkgs[3])
				# # bkgFcn2 = bkgFunc(bkgs[1], row['run'], 2, hgt[i], 10.0, sT, bkgs[3])
				# # bkgFcnC = bkgFunc(bkgs[2], row['run'], 0, hgt[i], 10.0, sT, bkgs[3])
								
				# # #bkgFcn1 = dBKG1(hgt[i], sT, bkgs[0], row['run'])
				# # #bkgFcn2 = dBKG2(hgt[i], sT, bkgs[1], row['run'])
				# # #bkgFcnC = dBKGc(hgt[i], sT, bkgs[2], row['run'])
			# # else:
				# # if bkg[0] > 0.0:
					# # bkgFcn1 = measurement(bkgs[0],sqrt(bkgs[0]))
				# # else:
					# # bkgFcn1 = measurement(0.0,0.0) 
				# # if bkg[1] > 0.0:
					# # bkgFcn2 = measurement(bkgs[1],sqrt(bkgs[1]))
				# # else:
					# # bkgFcn2 = measurement(0.0,0.0)
				# # if bkg[2] > 0.0:
					# # bkgFcnC = measurement(bkgs[2],sqrt(bkgs[2]))
				# # else:
					# # bkgFcnC = measurement(0.0,0.0)
			# # # Backgrounds
			# # if singLT == True:
				# # #if ((cts1 + dt1) + (cts2 + dt2)) > (bkgFcn1 + bkgFcn2).val * (row['te']-row['ts']):
				# # ctsSum += measurement(cts1 + cts2, sqrt((cts1 + cts2) / 15.0))
				# # dtSum  += measurement((dt1 + dt2), sqrt(dt1+dt2))
				# # bkgSum += (bkgFcn1 + bkgFcn2) * measurement(row['te']-row['ts'],0.0)
			# # elif coincLT == True:
				# # #if (ctsC + dtC) > bkgFcnC.val * (row['te']-row['ts']):
				# # ctsSum += measurement(ctsC,sqrt(ctsC))
				# # dtSum  += measurement(dtC, sqrt(dtC))
				# # bkgSum += (bkgFcnC) * measurement(row['te']-row['ts'],0.0)
		
	# # # Return sums
	# # return [ctsSum, dtSum, bkgSum]
	
	
# # def analyze_single_norm(runList,cts,nMon,nMad,runBreaks = [],nDet1 = 3, nDet2 = 5): # Main function here
	
	# # if (runBreaks == []):
		# # runBreaks = [runList[0],max(runList)]
	
	# # # Counting vectors for our normalization
	# # normedRun= []
	# # unldCts  = []
	
	# # # To be changed later-----------------------------------------------
	# # expoNorm = True
	# # useBkgCorr = True
	# # useDTCorr = True
	# # norm2All = True
	# # geomNorm = False
	# # w = 5
	# # holdSel = 20
	# # maxUnl = 100
	# # #-------------------------------------------------------------------
	
	# # if expoNorm == True:
		# # normExp = []
		# # wgtExp  = []
	# # if geomNorm == True:
		# # normPhi = []
		# # wgtPhi  = []
	
	# # # Load Norm Monitor name
	# # monStr1  = ('mon'+str(nDet1))
	# # monEStr1 = ('mon'+str(nDet1)+'E')
	# # monStr2  = ('mon'+str(nDet2))
	# # monEStr2 = ('mon'+str(nDet2)+'E')
	
	
	
	# # #pmts = [pmt1,pmt2]
	# # #-----------------------------------------------------------------------
	# # # Sum all counts from each holdSel length run for normalization analysis
	# # #-----------------------------------------------------------------------
	# # for run in runList:
		
		# # # Load the raw data from each run -- the file is separated by dip already.
		# # ctsRaw  = cts[cts['run']==run]
		# # nMonRaw = nMon[nMon['run']==run]
		# # if len(nMad) > 0:
			# # nMadRaw = nMad[nMad['run']==run]
		
		# # # For each run define nDips and holdT
		# # holdT = (nMonRaw['ts'] - nMonRaw['td']) - 50.0 # include cleaning time
	# # #	nDips = max(ctsRaw['dip']) # Note that this is zero indexed
		
		# # # Select only one holding time for normalization (+- 1s)
		# # if ((holdT < holdSel-1.0) or (holdT > holdSel+1.0)):
			# # continue
			
		# # # Figure out the dagger height
	# # #	if (nDips == 2):
	# # #		height = heightsDict[nDips+1]
	# # #	else:
	# # #		print "Run number", run, "is not a normal (production) 3-dip run!"
	# # #		continue
				
		# # # Normalize background. Separated into multiple PMTs
		# # if useBkgCorr:
			# # [normBKG1, normBKG2, normBKG, bkgTime] = extract_background(nMonRaw)
		# # else:
			# # [normBKG1, normBKG2, normBKG, bkgTime] = [0.0, 0.0, 0.0, 0.0]
		# # bkgRaw = [normBKG1, normBKG2, normBKG, bkgTime]
		# # #print bkgRaw
		# # [ctsSum, dtSum, bkgSum] = sum_all_dips(ctsRaw, bkgRaw,norm2All)
		# # if ctsSum.val == 0.0 and dtSum.val == 0.0 and bkgSum.val == 0.0:
			# # print("Run "+str(run)+" is not a normal (production) 3-dip run!")
	
		# # if useDTCorr == False:
			# # dtSum = measurement(0.0, 0.0)
		
		# # # Write out normalization monitors	
		# # normedRun.append(run)
		# # if expoNorm == True:
			# # normExp.append([nMonRaw[monStr1][0], nMonRaw[monStr2][0]])
			# # #normExp.append([measurement(nMonRaw[monStr1][0],nMonRaw[monEStr1][0]), measurement(nMonRaw[monStr2][0],nMonRaw[monEStr2][0])])
			# # wgtExp.append(1.0/((nMonRaw[monEStr1][0]/nMonRaw[monStr1][0])**2 + (nMonRaw[monEStr2][0]/nMonRaw[monStr2][0])**2))# + (bkgSum/ctsSum).val)))
		# # if geomNorm == True:
			# # normPhi.append([nMadRaw[nMadRaw['det']==nDet1]['mon'][0], nMadRaw[nMadRaw['det']==nDet2]['mon'][0]])
			# # #wgtPhi.append(1.0/((nMadRaw[nMadRaw['det']==nDet1]['monE'][0]/nMadRaw[nMadRaw['det']==nDet1]['mon'][0])**2 + (nMadRaw[nMadRaw['det']==nDet2]['monE'][0]/nMadRaw[nMadRaw['det']==nDet2]['mon'][0])**2 + (bkgSum/ctsSum).val))
		
		# # unldCts.append((ctsSum + dtSum - bkgSum).val)	
		# # #unldCts.append((ctsSum + dtSum - bkgSum))	
	# # #-----------------------------------------------------------------------
	# # #-----------------------------------------------------------------------
	# # # Normalization through normalization_params (with possible weighting)
	# # #print normExp, wgtExp
	# # if len(normedRun) > 0:
		# # if expoNorm == True:
			# # #nCoeff, coErr = normalization_params(normedRun,np.dot(np.diag(wgtExp),normExp),np.dot(np.diag(wgtExp),unldCts),runBreaks,w)
			# # nCoeff,coErr = normalization_params(normedRun,normExp,unldCts,runBreaks,w) # Do unweighted here
			# # #nCoeff,coErr = normalization_params_meas(normedRun,normExp,unldCts,runBreaks,w)
			# # print("Using "+str(len(nCoeff))+" of "+str(len(normedRun))+" unique normalization coefficients! (Expo. weighting)")
		# # if geomNorm == True:
			# # #nCoeffPhi = normalization_params(normedRun,np.dot(np.diag(wgtPhi),normPhi),np.dot(np.diag(wgtPhi),unldCts),runBreaks,w) 
			# # nCoeffPhi = normalization_params(normedRun,normPhi,unldCts,runBreaks,w) 
			# # print("Using "+str(len(nCoeffPhi))+" of "+str(len(normedRun))+" unique normalization coefficients! (Geom. weighting)")
	# # #-----------------------------------------------------------------------
	# # else:
		# # nCoeff = [1.0,0.0]
		# # coErr  = [0.0,0.0]
	
	
	# # #for i, co in enumerate(nCoeff):
	# # #	print co, coErr[i]
	# # extractObs = False # for Dan
	# # #if extractObs:
	
	# # #nMonOut = []
	# # #timeOut = []
		
	
	# # # Unload parameters
	# # runNum  = []
	# # holdVec = []
	# # nCtsVec = []
	# # #if (useMeanArr == True) or (plotPSE == True):
	# # meanArr = []
	# # #if plotRaw == True:
	# # ctsVec  = []
	# # #if plotBSub == True:
	# # bSubVec = []
	# # #if plotNCts == True or plotNHists:
	# # #	if expoNorm == True:
	# # nCtsVec = []
	# # #	if geomNorm == True:
	# # nPhiVec = []	
	# # #if plotNorm == True:
	# # normVec1 = []
	# # normVec2 = []
	# # #	if expoNorm == True:
	# # normVec = []
	# # #	if geomNorm == True:
	# # nPhiVec = []
	# # #if plotPSE == True:
	# # pct2 = []
	# # pct3 = []
	# # #if plotSig == True:
	# # sig2noi = []
	
	
	# # normBuff1 = []
	# # normBuff2 = []
	
	# # # Indices for calculating normalizations
	# # normIndex = 0
	# # bCount = 0
	
	# # normVar = measurement(0.0,0.0)
	# # # Loop through and determine normalization/spectrum dependence for each run.
	# # for run in runList:
		
		# # # Data buffer for a given run
		# # ctsRaw = cts[cts['run']==run]
		# # nMonRaw = nMon[nMon['run']==run]
		# # if len(nMad) > 0:
			# # nMadRaw = nMad[nMad['run']==run]
		
		# # # For each run define nDips and holdT
		# # holdT = (nMonRaw['ts'] - nMonRaw['td']) - 50.0 # include cleaning
	
		# # #-------------------------------------------------------------------
		# # # Apply normalization
		# # #-------------------------------------------------------------------
		# # # If we're past the next bCount we want to skip this 
		# # if normIndex < len(normedRun)-1 and bCount < len(runBreaks)-1:
			
			# # # Two reasons to increment normIndex:
			# # # (1) we're using the next normalization run
			# # # (2) something changed in runBreaks but we haven't gone on to next normedRun.
			# # # Increment bCount first.
			# # if run >= runBreaks[bCount+1]:
				# # while run >= runBreaks[bCount+1]:
					# # bCount+=1
					# # if bCount >= len(runBreaks)-1:
						# # break
				# # while run >= normedRun[normIndex+1]:
					# # normIndex+=1
					# # if normIndex >= len(normedRun)-1:
						# # break
				# # if normedRun[normIndex] < runBreaks[bCount]:
					# # normIndex+=1
					
			# # # Each run in norm we increment
			# # elif abs(run - normedRun[normIndex]) > abs(run - normedRun[normIndex+1]) and normedRun[normIndex+1] < runBreaks[bCount+1]:
				# # normIndex+=1
		
		# # if normIndex < len(normedRun)-1:
			# # while run >= normedRun[normIndex+1]:
				# # normIndex+=1		
				# # if normIndex == len(normedRun) - 1:
					# # break
		# # #elif normIndex < len(normedRun)-1:
		# # #	if run >= normedRun[normIndex+1]:
		# # #		normIndex+=1		
	
		# # if expoNorm == True:
			# # while normIndex > len(nCoeff)-1:
				# # normIndex-=1
		# # elif geomNorm == True:
			# # while normIndex > len(nCoeffPhi)-1:
				# # normIndex-=1
		
		# # if bCount > len(runBreaks)-1:	
			# # bCount-=1
		
		# # if extractObs:
			# # timeOut.append(holdT)
			# # nMonOut.append([measurement(nMonRaw[monStr1],nMonRaw[monEStr1]),measurement(nMonRaw[monStr2],nMonRaw[monEStr2])])
		
		# # # Calculate normalization from monitors
		# # if expoNorm == True:
			# # if len(normedRun) > 0:
				# # norm1 = measurement(nMonRaw[monStr1],nMonRaw[monEStr1])*measurement(nCoeff[normIndex][0],coErr[normIndex][0])
				# # norm2 = measurement(nMonRaw[monStr2],nMonRaw[monEStr2])*measurement(nCoeff[normIndex][1],coErr[normIndex][1])
			# # #norm1 = measurement(nMonRaw[monStr1]*nCoeff[normIndex][0], nMonRaw[monEStr1]*nCoeff[normIndex][0])
			# # #norm2 = measurement(nMonRaw[monStr2]*nCoeff[normIndex][1], nMonRaw[monEStr2]*nCoeff[normIndex][1])
				
			# # else:
				# # norm1 = measurement(nMonRaw[monStr1],nMonRaw[monEStr1])*measurement(nCoeff[0],coErr[0])
				# # norm2 = measurement(nMonRaw[monStr2],nMonRaw[monEStr2])*measurement(nCoeff[1],coErr[1])
			# # norm = norm1 + norm2
			# # #print norm.err/norm.val
			# # #norm = measurement(norm1.val + norm2.val, 0.0)
		# # else:
			# # norm1 = []
			# # norm2 = []
			# # norm = []
		# # # Calculate normalization from phi values
		# # if geomNorm == True:
			# # normP1 = measurement(nMadRaw[nMadRaw['det']==nDet1]['mon']*nCoeffPhi[normIndex][0],nMadRaw[nMadRaw['det']==nDet1]['monE']*nCoeffPhi[normIndex][0])
			# # normP2 = measurement(nMadRaw[nMadRaw['det']==nDet2]['mon']*nCoeffPhi[normIndex][1],nMadRaw[nMadRaw['det']==nDet2]['monE']*nCoeffPhi[normIndex][1])
			# # normPhiM = normP1 + normP2
		# # else:
			# # normP1 = []
			# # normP2 = []
			# # normPhiM = []
		# # #-------------------------------------------------------------------
			
		# # # Extract Background Rate
		# # if useBkgCorr:
			# # [normBKG1, normBKG2, normBKG, bkgTime] = extract_background(nMonRaw)
		# # else:
			# # [normBKG1, normBKG2, normBKG, bkgTime] = [0.0, 0.0, 0.0, 0.0]
		# # bkgRaw = [normBKG1, normBKG2, normBKG, bkgTime]
		
		# # # Now do our main meat
		# # [ctsSum, dtSum, bkgSum] = sum_all_dips(ctsRaw, bkgRaw)
		# # if useDTCorr == False:
			# # dtSum = measurement(0.0, 0.0)
		
		# # dagDown, pctDips = find_mean_arrival(ctsRaw, [ctsSum,dtSum,bkgSum], bkgRaw)	
		# # dagDown = measurement(dagDown.val - (nMonRaw['td']+50.0),dagDown.err)
		
		# # # Check if signal to background is too low
		# # if useBkgCorr == True and bkgSum.val > 0.0:
			# # if ctsSum.val/(bkgSum.val) < 1.0:
				# # continue
		# # elif bkgSum.val <= 0.0:
			# # continue
		# # #ctsSum.err = 0.0
		# # #dtSum.err = 0.0
		# # #bkgSum.err = 0.0	
		# # if (dagDown.val > 0):# and (norm.val > norm.err) and (((ctsSum+dtSum-bkgSum)/norm).val >((ctsSum+dtSum-bkgSum)/norm).err):
			
			# # if 19.0 < holdT < 21.0:
				# # normVar += (((ctsSum + dtSum - bkgSum)/norm) -measurement(1.0,0.0))*(((ctsSum + dtSum - bkgSum)/norm) - measurement(1.0,0.0)) 
			
				# # if not nCoeff[normIndex][0] == 0 and not nMonRaw[monEStr1] == 0:
					# # #if abs((nMonRaw[monStr1] - ((ctsSum+dtSum-bkgSum).val - nCoeff[normIndex][1]*nMonRaw[monStr2])/nCoeff[normIndex][0]) / nMonRaw[monEStr1]) < 100:
								
						# # normBuff1.append((nMonRaw[monStr1] - ((ctsSum+dtSum-bkgSum).val - nCoeff[normIndex][1]*nMonRaw[monStr2])/nCoeff[normIndex][0]) / nMonRaw[monEStr1])
					# # #else: 
					# # #	print run
				# # #else:
				# # #	print run
				# # if not nCoeff[normIndex][1] == 0 and not nMonRaw[monEStr2] == 0:
					# # #if abs((nMonRaw[monStr2] - ((ctsSum+dtSum-bkgSum).val - nCoeff[normIndex][0]*nMonRaw[monStr1])/nCoeff[normIndex][1]) / nMonRaw[monEStr2]) < 100:
						# # normBuff2.append((nMonRaw[monStr2] - ((ctsSum+dtSum-bkgSum).val - nCoeff[normIndex][0]*nMonRaw[monStr1])/nCoeff[normIndex][1]) / nMonRaw[monEStr2])
					# # #else:
					# # #	print run
				# # #else:
				# # #	print run
				
				# # if nCoeff[normIndex][0] == 0 and nMonRaw[monEStr1] == 0 and nCoeff[normIndex][1] == 0 and nMonRaw[monEStr2] == 0:
					# # print (str(run)+" has no normallization counts!")
				
			
			# # # VECTORS:
			# # runNum.append(run)
			# # holdVec.append(holdT)
			
			# # if expoNorm: # using expoNorm first
				# # nCtsVec.append((ctsSum + dtSum - bkgSum)/norm)
			# # elif geomNorm:
				# # nCtsVec.append((ctsSum + dtSum - bkgSum)/normPhiM)
			# # #else:
			# # #	nCtsVec.append(ctsSum + dtSum - bkgSum)
			
			# # # OPTIONAL VECTORS
			# # #if (useMeanArr == True) or (plotPSE == True):
			# # meanArr.append(dagDown)
			# # #if plotRaw == True:
			# # ctsVec.append(ctsSum)
			# # #if plotBSub == True or extractObs == True:
			# # bSubVec.append(ctsSum + dtSum - bkgSum)		
			# # #if plotNorm == True:
			# # #	if expoNorm == True:
			# # normVec.append(norm)
			# # if expoNorm:
				# # normVec1.append(measurement(nMonRaw[monStr1], nMonRaw[monEStr1]))
				# # normVec2.append(measurement(nMonRaw[monStr2], nMonRaw[monEStr2]))
			# # else:
				# # normVec1.append(normP1)
				# # normVec2.append(normP2)
			# # #	if geomNorm == True:
			# # nPhiVec.append(normPhiM)		
			# # #	if not expoNorm: # For now just plot expoNorm vectors
			# # #			normVec1.append(normP1)
			# # #			normVec2.append(normP2)
			# # #if plotPSE == True:
			# # pct2.append(pctDips[1])
			# # pct3.append(pctDips[2])
			# # # if plotSig == True:
			# # sig2noi.append(ctsRaw[2]['te'] - ctsRaw[0]['ts'])
		# # else:
			# # print(str(run)+str(dagDown))
	
	# # print (str(normVar)+str(np.sqrt(normVar.val) / np.sqrt(len(normedRun))))
	# # if extractObs:
		# # write_summed_counts(runNum,timeOut,bSubVec,nMonOut)
			
	# # print("Managed to successfully normalize "+str(len(runNum))+" runs! This is a ratio of: "+str(float(len(runNum))/float(len(runList))))
	# # print(" ")
	
	# # if len(runNum) < 8:
		# # sys.exit("Error! Unable to normalize an entire octet's worth of runs! Exiting...")
	# # #range1 = np.linspace(float(min(normBuff1)),float(max(normBuff1)),20)
	# # #print normBuff1
	# # #print normBuff2
	# # #histout1, bins1 = np.histogram(normBuff1, bins=range1)
	# # #plt.hist(histout1, bins=bins1)
	# # #plt.show()
	
	# # # lol i'm just returning a bunch of stuff
	# # return runNum,holdVec,nCtsVec,meanArr,ctsVec,bSubVec,normVec,normVec1,normVec2,nPhiVec,pct2,pct3,sig2noi

# # def extract_values(rL, cts, nMon, m1 = 3, m2 = 4, hold = 0, pmt1=True, pmt2=True,dips = range(0,3),runBreaks=[]):
	# # # extract_values will convert your lists into a usable thing for normalization.
	
	# # #-------------------------------------------------------------------
	# # maxUnl = 100
	# # dBkg   = True
	# # useBkgCorr = True
	# # useMeanArr = True
	# # useDTCorr  = True
	# # nDipsReq= 3
	# # #-------------------------------------------------------------------
	
	# # # Counting vectors for our normalization
	# # goodR = [] # List of runs we've created
	# # timeL = [] # List of "holding times"
	# # timeM = [] # List of "Mean Arrival Times"
	# # ctsL  = [] # N summed counts values
	# # monL  = [] # 2xN monitor values
	# # monEL = [] # 2xN monitor errors
	# # bSub  = [] # Background we're taking out
	# # bSumTimeCorr=[]
	# # try: # Hacky way to check singles vs. coincidence
		# # c = cts['coinc'][0]
		# # coin = True
	# # except ValueError:
		# # coin = False
	# # except TypeError:
		# # coin = False
		# # print("Error! Run has no associated datatypes!")
	# # try:
		# # c = cts['d1'][0]
		# # sing = True
	# # except ValueError:
		# # sing = False 
	# # except TypeError:
		# # sing = False
		# # print("Error! Run has no associated datatypes!")
	# # if (not sing) and (not coin):
		# # print("Error! Unable to find either singles OR coincidence counts!")
		# # return [],[],[],[],[],[],[],[],[]
	
	# # # Load Norm Monitor names
	# # monStr1  = ('mon'+str(m1))
	# # monEStr1 = ('mon'+str(m1)+'E')
	# # monStr2  = ('mon'+str(m2))
	# # monEStr2 = ('mon'+str(m2)+'E')
	# # bkgTimes = []
	# # bkgRates = []
	# # ctsF     = []
	# # tH       = []
	# # # Averages for background
	# # rBInd = 0
	# # while runBreaks[rBInd+1] >= min(rL):
		# # rBInd += 1
		# # if rBInd == len(runBreaks) - 1:
			# # rBInd -= 1
			# # break
	# # bkgUC = 0
	# # correction = []
	# # pctDip = []
	# # for run in rL:
		
		# # # Initialize sums
		# # ctsSum = measurement(0.0,0.0)
		# # dtSum  = measurement(0.0,0.0)
		# # bkgSum = measurement(0.0,0.0)
		# # tCSum  = measurement(0.0,0.0)		
		# # # Load the raw data from each run -- the file is separated by dip already.
		# # ctsRaw  = cts[cts['run']==run]
		# # nMonRaw = nMon[nMon['run']==run]
		# # # For each run define nDips and holdT
		# # holdT = (nMonRaw['ts'] - nMonRaw['td']) - 50.0 # include cleaning time
		# # if (holdT < 0.0):
			# # #print("Hold Time is negative for run "+str(run))
			# # continue
			
		# # # Select only one holding time for normalization (+- 1s)
		# # if (hold > 0):
			# # if ((holdT < hold-1.0) or (holdT > hold+1.0)):
				# # continue
		
		# # # Average background inside this run_break
		# # if bkgUC == 0 or run >= runBreaks[rBInd+1]:
			# # [bkgU1, bkgU2, bkgUC,_t] = extract_average_background(run,cts,nMon,runBreaks,9999,100,pmt1,pmt2,sing,rL)
			# # while rBInd >= len(runBreaks)-1:
				# # rBInd -= 1
			# # while run >= runBreaks[rBInd+1]:
				# # rBInd += 1
				# # if rBInd == len(runBreaks) - 1:
					# # rBInd -=1
					# # break
		# # # Extract Background Rate
		# # if useBkgCorr:
			# # [normBKG1,normBKG2,normBKG,bkgTime] = extract_average_background(run,cts,nMon,runBreaks,0,100,pmt1,pmt2,sing,rL)
			# # #[normBKG1, normBKG2, normBKG, bkgTime] = extract_background(nMonRaw,pmt1,pmt2)
			# # #[normBKG1_U, normBKG2_U, normBKG_U, bkgTime_U, dt_U] = extract_background_unload(ctsRaw,pmt1,pmt2,maxUnl,sing)
		# # else:
			# # [normBKG1, normBKG2, normBKG, bkgTime] = [0.0, 0.0, 0.0, 0.0]
		
		# # # And "smooth" the coincidence backgrounds 
		# # normBKG = normBKG #* (1 + ((normBKG1 - bkgU1) + (normBKG2 - bkgU2))/(normBKG1+normBKG) )
				
		# # nDips = max(ctsRaw['dip']) # Note that this is zero indexed
		# # if nDips != nDipsReq - 1: # Check that we're doing the right number of dips
			# # print("Run "+str(run)+" is not a proper "+str(nDipsReq)+" dip run")
			# # continue
				
		# # if coin: # Now do our main meat calculations
			# # [ctsSumL, dtSumL, bkgSumL] = sum_coinc_counts(ctsRaw, normBKG,bkgTime, maxUnl, dBkg)
			# # [d1c,d2c] = [0.5,0.5] # Efficiency equalized between the two PMTs
			# # tCSumL = np.zeros(len(bkgSumL)) # Time dependence for coincidences is presently zero.
		# # elif sing:
			# # [ctsSumL, dtSumL, bkgSumL],[d1c,d2c],tCSumL = sum_singles_counts(ctsRaw, normBKG1, normBKG2, bkgTime, maxUnl, pmt1,pmt2,dBkg,normBKG)
				
		# # for i in dips:
			# # if 11275 <= run < 11285:
				# # print(ctsSumL[i].err/ctsSumL[i].val,dtSumL[i].err/dtSumL[i].val,bkgSumL[i].err/bkgSumL[i].val)
			# # ctsSum  += ctsSumL[i]
			# # dtSum   += dtSumL[i]
			# # bkgSum  += bkgSumL[i]
			# # tCSum   += tCSumL[i]
			
		# # #print(bkgSum)	
		# # bkgTimes.append(nMonRaw['bkgE'] - nMonRaw['bkgS'])
		# # bkgRates.append(bkgSum + tCSum)	
		# # if useDTCorr == False:
			# # dtSum = measurement(0.0, 0.0)
		
		# # if useMeanArr:
			# # dagDown, pctDips = find_mean_arrival(ctsRaw, [ctsSum,dtSum,bkgSum], [normBKG1, normBKG2, normBKG, bkgTime],dips,pmt1,pmt2,sing,dBkg,maxUnl)
			# # dagDown = measurement(dagDown.val - (nMonRaw['td']+50.0),dagDown.err)
		# # else:
			# # dagDown = measurement(holdT,0.0)
			# # pctDips = []

		# # #print(dagDown)
		# # # Write out normalization monitors	
		# # goodR.append(run)
		# # ctsL.append(ctsSum + dtSum - bkgSum)
		# # timeL.append(holdT)
		# # timeM.append(dagDown)
		# # #timeL.append(dagDown)
		# # #monL.append([nMonRaw[monStr1][0], nMonRaw[monStr2][0]])
		# # monL.append([measurement(nMonRaw[monStr1][0],nMonRaw[monEStr1][0]), measurement(nMonRaw[monStr2][0],nMonRaw[monEStr2][0])])
		# # pctDip.append(pctDips)
		# # #monL.append([measurement(nMonRaw[monStr1][0],0.0), measurement(nMonRaw[monStr2][0],0.0)])
		# # monEL.append([d1c,d2c])
		# # #if nMonRaw[monEStr1][0] > 0 and nMonRaw[monEStr2][0] > 0:
		# # #	if nMonRaw[monStr1][0] > 0 and nMonRaw[monStr2][0] > 0:
		# # #		monEL.append(1.0/((nMonRaw[monEStr1][0]/nMonRaw[monStr1][0])**2 + (nMonRaw[monEStr2][0]/nMonRaw[monStr2][0])**2)) # Weighting by 1/
		# # #	else:
		# # #		monEL.append(0.0) # If no counts, weight by 0
		# # #else:
		# # #	monEL.append(np.inf) # if no error, weignt infinitely
		# # ctsF.append(float(ctsSum+dtSum))
		# # bSub.append(float(bkgSum-tCSum)) # For use in likelihood fitting
		# # bSumTimeCorr.append(float(tCSum))
		# # #bSub.append(float(dtSum))
		# # tH.append(int(round(float(dagDown))))
	# # #plt.hist(correction,bins=50)
	# # #plt.show()
	# # #plt.plot(bkgTimes, bkgRates, 'b.')
	# # #plt.show()
	# # return goodR, timeL,timeM, ctsL, monL, monEL,ctsF,bSub,tH,bSumTimeCorr,pctDip
