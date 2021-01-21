#!/usr/local/bin/python3
import os
import sys
import pdb
import csv
from math import *
import numpy as np
#from statsmodels.stats.weightstats import DescrStatsW
from scipy import stats, special
from scipy.odr import *
from scipy.optimize import curve_fit, nnls
from datetime import datetime
import matplotlib.pyplot as plt

from PythonAnalyzer.extract import *
from PythonAnalyzer.backgrounds import *
from PythonAnalyzer.boolconversion import *
from PythonAnalyzer.pairingandlifetimes import *
from PythonAnalyzer.functions import *
from PythonAnalyzer.classes import *
from PythonAnalyzer.plotting import *
from PythonAnalyzer.plotting_monitors import *
from PythonAnalyzer.plotting_observables import *
from PythonAnalyzer.dataIO import *
from PythonAnalyzer.runFilters import *
from PythonAnalyzer.writeOut import *
from PythonAnalyzer.likelihood_lifetime import *

#-----------------------------------------------------------------------
# This analyzer code normalizes the monitor detector counts
# and provides a yield from our short runs. This needs 
# monitor detector counts from our data output.
# 
# Information about more runs:
#  5707 (Foil monitor changes)
#  6754 (Foil monitor changes back)
#  7439 (Tagbits don't work!)
#  8595 Roundhouse installation
#
# Information about runs -- Looks like 9960 is when the RH det came online
#  10645-10680 have a crazy high rate
#  10942 (Something happens -- RH detector height change?)
#  10995 (RH detector height change?)
#  11086 (RH detector height change?)
#  11305 (Change in stability)
#  13200 (PMT changes)
#
# 13240-13290 only have 1 PMT -- excluded for now...
#
# I've made it uncomfortably functionally programmed. This is because I
# like C++ and treat Python the same way.

if __name__ == "__main__":
	
#	try:
#		os.environ["OMP_NUM_THREADS"] = "1" # This is the master MPI check
#		pool = MPIPool()
#		if not pool.is_master(): # All the pools will get called after the master loads
#			pool.wait() # Wait if you're not the master
#			sys.exit(0)
#	except ValueError:
#		print("Running without MPI Pool!")
#		pool=[]
	pool = []
	
	# Initial loading function for our output.
	if(len(sys.argv) < 3):
		sys.exit("Error! Usage: python analyzer.py counts_file_name monitor_file_name mad_file_name badRunsList useTheseRuns")
	else:
		print(" ")
		print("Thank you for using LifetimeAnalyzer!")
		print(" ")
	
	#nList,bData,bNorm,bPlot,bWrite = initialize_global_booleans()
	vb,loadS,anlz, outS = initialization_script()
	
	#initialization_script(nList,bData,bNorm,bPlot,bWrite)
	#fileList = [''] * 20 # If you need more scale this up
	fileListV = []
	for i, f in enumerate(sys.argv):
		#fileList[i] = f # If you're breaking here, fileList[i] is too short. Though that actually doesn't matter...
		if i ==0:
			continue
		fileListV.append(f)
			
	
	#runListAll, cts,  nMon, nMad, bData = load_counts_lists(bData,nList,fileList[1],fileList[2],fileList[3],fileList[4],fileList[5])
	runListAll, ctsSing, ctsCoinc,  nMon, nMad, bkgDep,bkgTime = load_flex_lists(fileListV,loadS,vb)
	runListS, detS = sort_counts_by_year(runListAll,anlz,vb) # Parse runs by year
	
	# Figure out the files we've loaded (e.g. coinc vs singles vs etc.)
	try: # Multiple thresholds?
		if len(nMon[1]) > 1:
			multiScan = True
			if len(ctsSing) != len(nMon) and len(ctsCoinc) != len(nMon):
				sys.exit("ERROR! You can't mix thresholds across singles/coincidences!")
	except IndexError:
		multiScan = False

	ctsTrunc = []
	# Now we're checking the configurations of counts
	if (len(ctsSing) != 0 and len(ctsCoinc) != 0): # Loaded both sing. and coinc.
		oneUnload = False
	elif (len(ctsSing) == 0 and len(ctsCoinc) != 0): # One Singles
		ctsTrunc   = ctsCoinc
		oneUnload  = True # Flag for just singles/coinc
		anlz.sing = False # Set analyzer singles or coincidence
		saveName = "outputCoinc.txt" # Eh, not so needed for 1D scan.
	elif (len(ctsSing) != 0 and len(ctsCoinc) == 0):
		ctsTrunc = ctsSing
		oneUnload  = True
		anlz.sing = True
		saveName = "outputSing.txt"
	else:
		sys.exit("ERROR! No counts list loaded!")	
	# Figure out if we're comparing High/Low Thresh things:
	
	# Temporary hardcode ----------------------------------------------------------------------------
	pmtScan = False
	# pmtScan does not work with singles/coincidence comparisons
	#------------------------------------------------------------------------------------------------	
	#if not oneUnload:
	#	pmtScan = False
	reducedRunS = [] # Our reduced run vector (Singles)
	reducedRunC = [] # Our reduced run vector (Coincidence)
	
	if pmtScan: # Make a bunch of lists
		print("Scanning multiple PMTs on/off!!")
		pmtList = [0,1,2] # PMTs set as such
		if multiScan:
			print("Doing Multi Scan with PMT Scan!")
		for pmt in range(len(pmtList)):
			if multiScan:
				threshList = []
				for ind in range(len(nMon)): # Should be 2?
					reducedRunS.append([])
					reducedRunC.append([])
					threshList.append(ind)
			else:
				reducedRunS.append([])
				reducedRunC.append([])
	elif multiScan: # If we're doing a multiscan I'm putting in more lists
		print("Doing Multiscan")
		threshList = []
		for ind in range(len(nMon)): # List of empty lists
			reducedRunS.append([])
			reducedRunC.append([])
			threshList.append(ind)
			
	runBreaks = []
	for i, runList in enumerate(runListS): # Sorting data by year!
		if len(runList) == 0:
			continue
		# nDet from config flags:
		if 4200 <= runList[0] < 9600: # 2017
			anlz.year = 2017
		elif 9600 <= runList[0] < 14728: # 2018
			anlz.year = 2018
		else:
			if anlz.vb:
				print("ERROR: This data isn't from 2017-2018!")	
			continue
		
		rBTmp = [] # Establish runbreaks list
		if loadS.loadBreaks: # Generate runbreaks automatically
			if ((anlz.year == 2017 and 0 not in anlz.det17) or \
				(anlz.year == 2018 and 0 not in anlz.det18)):
				pctBreak = 0.8
				if anlz.year == 2017:
					rfilter_easy = filter_runs_by_mon(runList, nMon, anlz.det17[0],anlz.det17[1]) # Easy is a hard-cutoff
					rfilter,RBG = filter_runs_by_error(rfilter_easy,nMon,anlz.det17[0],anlz.det17[1],3)
					rBTmp, plot_gen = load_run_breaks_rolling(rfilter, nMon, anlz.det17[0],anlz.det17[1],2,pctBreak, 50)
				if anlz.year == 2018:	
					rfilter_easy = filter_runs_by_mon(runList, nMon, anlz.det18[0],anlz.det18[1]) # Easy is a hard-cutoff
					rfilter,RBG = filter_runs_by_error(rfilter_easy,nMon,anlz.det18[0],anlz.det18[1],3)
					rBTmp, plot_gen = load_run_breaks_rolling(rfilter, nMon, anlz.det18[0],anlz.det18[1],2,pctBreak, 50)
		if len(rBTmp) == 0: # This is more commonly used.			
			rMin = min(runList)
			rMax = max(runList)+1
			# Hardcoded runBreaks
			
			# runBreaks_all    = [4298,4373,4465,4487,4585,4620,4673,4744,4747,4767,\
								# 4785,4849,4899,4931,4980,5010,5044,5127,5195,5316,\
								# 5341,5441,5462,5576,5578,5607,5634,5668,5713,5719,\
								# 5841,5884,5923,5955,5964,5990,6047,6092,6123,6180,\
								# 6327,6363,6378,6424,6452,6653,6711,6731,6776,6882,\
								# 6931,6972,7106,7327,7356,7484,7526,7571,7612,7618,\
								# 7810,7822,7858,7902,7983,8118,8178,8221,8271,8303,\
								# 8340,8372,8450,8550,8593,9764,9908,9946,9959,10032,\
								# 10061,10163,10232,10271,10351,10354,10410,10505,10573,\
								# 10607,10723,10815,10879,10988,11045,11083,11102,11135,\
								# 11175,11202,11311,11669,11835,11902,11927,11982,12009,\
								# 12102,12186,12216,12260,12289,12341,12424,12481,12505,\
								# 12595,12645,12696,12748,12803,12833,12867,12885,12909,\
								# 13043,13113,13162,13172,13293,13475,13539,13738,13787,\
								# 13841,13908,13919,13954,13998,14159,14238,14384,14516]

			
			runBreaks_all=[4230,4391,4415,4711,5453,5475,5713,5955,6126,6429,6754,6930,7326,7490,\
							9767,9960,10936,10988,11085,11669,12516,13209,13307,14509]
							#9767,9960,10936,10988,11085,11669,12516,13209,14509]
			for i in range(0,len(runBreaks_all)-1): # reduce the number of runBreaks
				if runBreaks_all[i] <= rMin < runBreaks_all[i+1]: 
					rBTmp.append(runBreaks_all[i]) # find first
					for j in range(i+1,len(runBreaks_all)):
						if runBreaks_all[j] <= rMax:
							rBTmp.append(runBreaks_all[j]) # append through the last break
						else:
							break
					break
			if rBTmp[-1] < rMax: # And possibly add the last run.
				rBTmp.append(rMax)
			rBTmp.sort()
		print(rBTmp)
		# Now we scan stuff through a shitton of loops and if statements
		# Could optimize this for readability?
		if oneUnload: # Just one type
			if not pmtScan:
				if not multiScan: # One Threshold		
					redRun = analyze_single_norm(runList,ctsTrunc,nMon,bkgDep,bkgTime,anlz,rBTmp)
					reducedRunS.extend(redRun)
				else: # Multi-threshold, need to scan
					for j in [0,1]: # Hardcode
						anlz.thresh_scan(j)
						redRun = analyze_single_norm(runList,ctsTrunc[j],nMon[j],bkgDep[j],bkgTime[j],anlz,rBTmp)
						reducedRunS[j].extend(redRun) # Singles is fine
			else: # pmtScan 
				ctr = 0
				for p in pmtList:
					anlz.pmt_scan(p)
					if not multiScan: # One Threshold		
						redRun = analyze_single_norm(runList,ctsTrunc,nMon,bkgDep,bkgTime,anlz,rBTmp)
						reducedRunS[ctr].extend(redRun)
						ctr += 1
					else: # Multi-threshold, need to scan
						for j in [0,1]:
							anlz.thresh_scan(j)
							redRun = analyze_single_norm(runList,ctsTrunc[j],nMon[j],bkgDep[j],bkgTime[i],anlz,rBTmp)
							reducedRunS[ctr].extend(redRun) # Singles is fine
							ctr += 1
		else: # Singles and coincidences
			if not pmtScan: # No scanning PMTs
				# First take care of singles then do coincidences
				if not multiScan: # One Threshold
					anlz.sing = True
					redRunS = analyze_single_norm(runList,ctsSing, nMon,bkgDep,bkgTime,anlz,rBTmp)
					reducedRunS.extend(redRunS) # Separating out singles 
					anlz.sing = False
					redRunC = analyze_single_norm(runList,ctsCoinc,nMon,bkgDep,bkgTime,anlz,rBTmp)
					reducedRunC.extend(redRunC) # and coincidence
				elif multiScan: # Multi-Threshold, need to scan
					for j in [0,1]: # Hardcode 2 thresholds
						anlz.thresh_scan(j)
						anlz.sing = True
						redRunS = analyze_single_norm(runList,ctsSing[j],nMon[j],bkgDep[j],bkgTime[j],anlz,rBTmp)
						reducedRunS[j].extend(redRunS) # Separating singles
						anlz.sing = False
						redRunC = analyze_single_norm(runList,ctsCoinc[j],nMon[j],bkgDep[j],bkgTime[j],anlz,rBTmp)
						reducedRunC[j].extend(redRunC) # and coincidence
			else:
				# Do singles then tack coincidences onto the end.
				ctr = 0 # PMT information
				anlz.sing = True
				for p in pmtList:
					anlz.pmt_scan(p)
					if not multiScan: # One Threshold		
						redRunS = analyze_single_norm(runList,ctsSing,nMon,bkgDep,bkgTime,anlz,rBTmp)
						reducedRunS[ctr].extend(redRunS)
						ctr += 1
					else: # Multi-threshold, need to scan
						for j in [0,1]:
							anlz.thresh_scan(j)
							redRunS = analyze_single_norm(runList,ctsSing[j],nMon[j],bkgDep[j],bkgTime[j],anlz,rBTmp)
							reducedRunS[ctr].extend(redRunS) # Singles is fine
							ctr += 1
				ctr = 0
				anlz.sing = False
				if not multiScan:
					redRunC = analyze_single_norm(runList,ctsCoinc,nMon,bkgDep,anlz,rBTmp)
					reducedRunC.extend(redRunC) # and coincidence
				else:
					for j in [0,1]:
						anlz.thresh_scan(j)
						redRunC = analyze_single_norm(runList,ctsCoinc[j],nMon[j],bkgDep[j],bkgTime[j],anlz,rBTmp)
						reducedRunC[j].extend(redRunC)
				
		runBreaks.extend(rBTmp) # And extend the runBreaks
		
	#-------------------------------------------------------------------
	# Now that the loop is done, we should be able to calculate lifetime
	# and plot everything.
	#-------------------------------------------------------------------
	# Plotting and Output
	#-------------------------------------------------------------------	
	# Plan of Attack: 
	#   1) Single, easy plots (singles and coincidence the same)
	#   2) Then go and do singles vs coincidences
	#   3) Then go and do singles vs singles
	#-------------------------------------------------------------------
	ltMeas_Hardcoded = measurement(887.0954855525279,0.2638994427897106)
	# Hardcoding my High Threshold Coincidence Lifetime.
	# Later take this out.
	plot_gen = 1 # Acute code-readers will note there's a plot_gen earlier.
	
	
	if not multiScan:
		if oneUnload: # Just one type	
			# Assuming one unload reducedRunS:
			fmtBlobs      = []
			lifetimeBlobs = []
			lifetimeOuts  = []
			lifeYearBlobs = []
			ltVec, runPair2,holdPair,ltVec2 = pair_runs(runBreaks,reducedRunS,anlz)
			ltVecRaw, runPair,holdPair,ltVec2 = pair_runs(runBreaks,reducedRunS,anlz)
			#ltVec,runPair,redRunS_2 = pair_runs_from_file(reducedRunS,anlz,'/home/frank/FUCKED_MCA_Analysis/EricCompare/emf_one_pairs.csv')
			#ltVec,runPair2,redRunS_2 = pair_runs_from_file(reducedRunS,anlz,'/home/frank/Downloads/emf_pairs.csv')
			#lifetimeExp  = calc_lifetime_exp(reducedRunS,anlz)
			#lifetimePair = calc_lifetime_paired(ltVecRaw,runPair2)
			lifetimePair = calc_lifetime_paired(ltVec2,runPair)
			lifetimePairUnw = calc_lifetime_unw_paired(ltVecRaw,runPair)	
			#plot_lifetime_paired(runPair,ltVec2,[],lifetimePair,holdPair)
			#lifetimeSec,lifeYears  = calc_lifetime_paired_sections(ltVec,runPair2)
			lifetimeSec,lifeYears  = calc_lifetime_paired_sections(ltVec2,runPair)
			lifetimeBlobs.append(lifetimeSec)
			lifetimeOuts.append(lifetimePair)
			lifeYearBlobs.append(lifeYears)
			lts, rates, bkgs, hlds,norms = extract_paired_runs(runPair,reducedRunS,anlz)
			#plot_lifetime_scatter(lifetimePair,lts,rates,bkgs,hlds,norms,plot_gen)
			#plt.show()
			
			if not outS.plotBreaks: # If we don't want to plot runbreaks,
				runBreaks = []      # clear the buffer.
			if outS.plotRaw: # Raw Unload Counts
				plot_gen += plot_raw_yields(reducedRunS,plot_gen,runBreaks)
			if outS.plotNCts: # Normalized Unload Count
				plot_gen += plot_normalized_counts(reducedRunS,plot_gen,runBreaks)
			if outS.plotNHists: # Normalized Unload Histograms
				plot_gen += histogram_n_counts(reducedRunS,plot_gen,[20,50,100,200,1550])
			if outS.plotNorm:
				plot_gen += plot_normalization_monitor(reducedRunS,plot_gen,runBreaks)
			if outS.plotBSub:
				plot_gen += plot_background_subtracted(reducedRunS,plot_gen,runBreaks)
			if outS.plotPSE:
				plot_gen += plot_phasespace_evolution(reducedRunS,plot_gen,runBreaks,anlz)
				plot_gen += plot_dip_percents(reducedRunS,plot_gen,runBreaks,anlz)
				plot_gen += histogram_phasespace(reducedRunS,plot_gen,[20,1550])
				plot_gen += histogram_mean_arr(reducedRunS,plot_gen,[20,1550])
			if outS.plotSig:
				plot_gen += plot_signal2noise(reducedRunS,plot_gen,runBreaks)
				
			if outS.plotLTPair: # All the paired lifetimes
				plot_lifetime_paired(runPair,ltVec,[],lifetimePair,holdPair)
				histogram_lifetime_paired(runPair,ltVec,lifetimePair,holdPair)
				#plot_lifetime_paired_by_week(runPair,ltVec)
			if outS.plotLTExp: # Exponential lifetimes
				try:
					plot_lifetime_exponential(runNum,nCtsVec,holdT, lifetimeExp,scaleExp)
				except NameError:
					print("Plot not defined!")
					#plot_lifetime_exponential(runNum,nCtsVec,holdT,lifetimeExp)
		#else: # One Threshold Singles/Coincidence comparison
	else: # Multi-Threshold
		if oneUnload: # So either singles OR coincidence
			
			fmtBlobs      = []
			lifetimeBlobs = []
			anlz_tmp = anlz
			for k,rRun in enumerate(reducedRunS):
				if len(rRun) == 0:
					continue
				# Formatting for writeout:
				anlz_tmp.load_format(rRun[0])
				anlz_tmp.dips = anlz.dips
				fmtBlobs.append(anlz_tmp) # For formatting stuff
				if anlz.vb:
					print("\nComparing Different Thresholds!\n")
					if rRun[0].thresh:
						print("High Threshold:")
					else:
						print("Low Threshold:")
					if rRun[0].sing:
						if rRun[0].pmt1 and rRun[0].pmt2:
							print("   Singles (Combined)\n")
						elif rRun[0].pmt1:
							print("   PMT 1")
						else:
							print("   PMT 2")
					else:
						print("   Coincidence\n")
				ltVec, runPair,holdPair,ltVec2 = pair_runs(runBreaks,rRun,anlz)
				#lifetimeExp  = calc_lifetime_exp(rRun,anlz)
				#lifetimePair = calc_lifetime_paired(ltVec,runPair)
				lifetimePair = calc_lifetime_paired(ltVec2,runPair)
				#lifetimePairUnw = calc_lifetime_unw_paired(ltVec,runPair)
				lifetimeSec,lifeYear  = calc_lifetime_paired_sections(ltVec2,runPair)
				lifetimeBlobs.append(lifetimeSec)
				# For each of these we'll want to make the same plots.
				# Only do the first though (LowT Total).
				if not k == 0: # We should be able to utilize the comp. 
					continue   # plots with the general overview plots
				
				ltMeas0 = lifetimePair # Save LowT Total for subtracting
				if not outS.plotBreaks: # If we don't want to plot runbreaks,
					runBreaks = []      # clear the buffer.
				if outS.plotRaw: # Raw Unload Counts
					plot_gen += plot_raw_yields(rRun,plot_gen,runBreaks)
				if outS.plotNCts: # Normalized Unload Count
					plot_gen += plot_normalized_counts(rRun,plot_gen,runBreaks)
				if outS.plotNHists: # Normalized Unload Histograms
					plot_gen += histogram_n_counts(rRun,plot_gen,[20,50,100,200,1550])
				if outS.plotNorm:
					plot_gen += plot_normalization_monitor(rRun,plot_gen,runBreaks)
				if outS.plotBSub:
					plot_gen += plot_background_subtracted(rRun,plot_gen,runBreaks)
				if outS.plotPSE:
					plot_gen += plot_phasespace_evolution(rRun,plot_gen,runBreaks,anlz)
					plot_gen += plot_dip_percents(rRun,plot_gen,runBreaks,anlz)
					plot_gen += histogram_phasespace(rRun,plot_gen,[20,50,100,200,1550])
					plot_gen += histogram_mean_arr(rRun,plot_gen,[20,50,100,200,1550])
					plot_gen += histogram_phasespace_SL(rRun,plot_gen)
					plot_gen += histogram_mean_arr_SL(rRun,plot_gen)
				if outS.plotSig:
					plot_gen += plot_signal2noise(rRun,plot_gen,runBreaks)
			
			# Comparison between PMT1 and PMT2
			# if outS.plotRaw:
				# plot_gen += plot_PMT_comp(reducedRunS,plot_gen,runBreaks)
				# plot_gen += plot_PMT_cts_diff(reducedRunS,plot_gen,runBreaks)
			# if outS.plotBSub:
				# plot_gen += plot_PMT_bkg_comp(reducedRunS,plot_gen,runBreaks)
				# plot_gen += plot_PMT_bkg_diff(reducedRunS,plot_gen,runBreaks)
			# if outS.plotNCts:
				# plot_gen += plot_PMT_n_cts(reducedRunS,plot_gen,runBreaks)
			
			plot_lifetime_sections(lifetimeBlobs,ltMeas_Hardcoded)
		else: # Singles + coincidence + multiscan (so like everything)
			# OK now I make blobs to save
			anlz_tmp = anlz
			fmtBlobs      = []
			lifetimeBlobs = []
			lifetimeOuts  = []
			lifeYearBlobs = []
			for k,rRun in enumerate(reducedRunS):
				if len(rRun) == 0:
					continue
				# Formatting for writeout:
				anlz_tmp.load_format(rRun[0])
				anlz_tmp.thresh = rRun[0].thresh
				anlz_tmp.sing   = rRun[0].sing
				anlz_tmp.pmt1   = rRun[0].pmt1
				anlz_tmp.pmt2   = rRun[0].pmt2
				fmtBlobs.append(rRun[0]) # For formatting stuff
				if anlz.vb:
					print("\nComparing Different Thresholds!\n")
					if rRun[0].thresh:
						print("High Threshold:")
					else:
						print("Low Threshold:")
					if rRun[0].sing:
						if rRun[0].pmt1 and rRun[0].pmt2:
							print("   Singles (Combined)\n")
						elif rRun[0].pmt1:
							print("   PMT 1")
						else:
							print("   PMT 2")
					else:
						print("   Coincidence\n")
				ltVec, runPair,holdPair,ltVec2 = pair_runs(runBreaks,rRun,anlz)
				#lifetimeExp  = calc_lifetime_exp(rRun,anlz)
				#lifetimePair = calc_lifetime_paired(ltVec,runPair)
				lifetimePair = calc_lifetime_paired(ltVec2,runPair)
				#lifetimePairUnw = calc_lifetime_unw_paired(ltVec,runPair)
				lifetimeSec,lifeYears  = calc_lifetime_paired_sections(ltVec2,runPair)
				lifetimeBlobs.append(lifetimeSec)
				lifetimeOuts.append(lifetimePair)
				lifeYearBlobs.append(lifeYears)
			for k,rRun in enumerate(reducedRunC):
				if len(rRun) == 0:
					continue
				# Formatting for writeout:
				anlz_tmp.load_format(rRun[0])
				anlz_tmp.thresh = rRun[0].thresh
				anlz_tmp.sing   = rRun[0].sing
				anlz_tmp.pmt1   = rRun[0].pmt1
				anlz_tmp.pmt2   = rRun[0].pmt2
				fmtBlobs.append(rRun[0]) # For formatting stuff
				if anlz.vb:
					print("\nComparing Different Thresholds!\n")
					if rRun[0].thresh:
						print("High Threshold:")
					else:
						print("Low Threshold:")
					if rRun[0].sing:
						if rRun[0].pmt1 and rRun[0].pmt2:
							print("   Singles (Combined)\n")
						elif rRun[0].pmt1:
							print("   PMT 1")
						else:
							print("   PMT 2")
					else:
						print("   Coincidence\n")
				ltVec, runPair,holdPair,ltVec2 = pair_runs(runBreaks,rRun,anlz)
				#lifetimeExp  = calc_lifetime_exp(rRun,anlz)
				#lifetimePair = calc_lifetime_paired(ltVec,runPair)
				lifetimePair = calc_lifetime_paired(ltVec2,runPair)
				#lifetimePairUnw = calc_lifetime_unw_paired(ltVec,runPair)
				lifetimeSec,lifeYears  = calc_lifetime_paired_sections(ltVec2,runPair)
				lifetimeBlobs.append(lifetimeSec)
				lifetimeOuts.append(lifetimePair)
				lifeYearBlobs.append(lifeYears)
				if k==0:
					ltMeas0 = lifetimePair # Save LowT Total for subtracting
			plot_lifetime_sections(lifetimeBlobs,ltMeas_Hardcoded)
			#plot_lifetime_sections(lifetimeBlobs,ltMeas0)
		
			#pmt1V = []
			#pmt2V = []
			#singV = []
			#for i1,i2 in sig2noi:
			#	pmt1V.append(i1)
			#	pmt2V.append(i2)
			#	singV.append(i1+i2)
			#plot_gen = plot_signal2noise(runNum, holdVec, sig2noi, plot_gen)
			#plot_gen = plot_background_by_PMT(runNum, holdVec, singV,pmt1V,pmt2V,plot_gen)
			
			#plot_gen = plot_PMT_yields(runNum,holdVec,pmt1V,pmt2V,ctsVec,plot_gen)
			#plot_gen = histogram_n_counts(runNum,holdVec,ctsVec, plot_gen,[20,50,100,200,1550])
	#-------------------------------------------------------------------
	
	#-------------------------------------------------------------------
	# Lifetime plots don't need plot_gen (they're hardcoded)
	#
	#-------------------------------------------------------------------
	#write_adam_lifetimes(lifetimeOuts,lifeYearBlobs,fmtBlobs,anlz)
	write_lifetime_telapsed(lifetimeOuts,lifeYearBlobs,anlz)
	# Writeout
	if outS.writeLTPairs == True:
		write_lifetime_pairs(runPair,ltVec)
	if outS.writeAllRuns == True:
		write_all_runs(reducedRunS)
	if outS.writeLongY == True:
		write_long_runs(runNum,holdVec,nCtsVec)
		
	#write_mean_arr(reducedRunS)
	#write_backgrounds_out(reducedRunS)
	#write_extracted_rRed_3(reducedRunS)
	#write_extracted_pairs(reducedRunS,runPair2)
	#plot_gen = plot_expected_values(runVec,ctsVec,nCorr,holdVec,runBreaks = [], nPlt=1,normT=[],nSig = 3.0)
	#plot_gen = plot_expected(runNum,holdVec,ctsVec,normVec,plot_gen)
	
	try:
		sig = 0.
		back = 0.
		for r in reducedRunS:
			if r.hold > 21:
				continue
			if r.run > 7321:
				continue
			sig += r.ctsSum.val * r.pcts[1].val
			back += r.bkgSum.val *20. / 160.
		print(sig/back)
	except:
		sig = 0.
		back = 0.
	
	plt.show()
	sys.exit("Thank you for using the LifetimeAnalyzer!")
	
	
	
	# Sorted output data:
	# runNum = []
	# holdVec = []
	# nCtsVec = []
	# nCtsVec2 = [] # If trying two analyses simultaneously
	# meanArr = []
	# meanArr2 = []
	# ctsVec = []
	# ctsVec2 = []
	# bkgSubVec = []
	# bkgSubC = []
	# normVec = []
	# nVec1 = []
	# nVec2 = []
	# nPhiVec = []
	# pct2 = []
	# pct3 = []
	# sig2noi = []
	# runBreaks = []
	# bkgSub1 = []
	# bkgSub2 = []
	# pmt1V = []
	# pmt2V = []
	# #_s,_s = optimized_likelihood_fit(runListAll,ctsCoinc,nMon,saveName,pool)
	# #chen_yu_likelihood_fit(runListAll,ctsTrunc,nMon,[0,1,2])
	# #sys.exit()
	# #-------------------------------------------------------------------
	# # Modification of code 
	# # want to use config flags and just reduced_run
	# #-------------------------------------------------------------------
	
	# for i, runList in enumerate(runListS): # Sorting data by year!
		
		# if len(runList) == 0:
			# continue
		
		# #print cts
		# nDet1 = detS[i][0]
		# nDet2 = detS[i][1]
		
		# if loadS.loadBreaks and nDet1 > 0 and nDet2 > 0:
			# #runBreaks = load_run_breaks(runList, nMad,nList)
			# #pctBreak = 1/1.125
			# pctBreak = 0.8
			# #nRBreaks = convert_mad_to_mon(nMon,nMad)
			# rfilter_easy = filter_runs_by_mon(runList, nMon, nDet1, nDet2) # Easy is a hard-cutoff
			# #rfilter = filter_runs_by_asym(rfilter_easy, nMon, nDet1, nDet2, 4)
			# rfilter,RBG = filter_runs_by_error(rfilter_easy,nMon,nDet1,nDet2,3)
			# #runBreaks = load_run_breaks(rfilter,nMon,nDet1,nDet2)
			# runBreaksTmp, plot_gen = load_run_breaks_rolling(rfilter, nMon, nDet1,nDet2, 2,pctBreak, plot_gen)
		# else:
			# #runBreaks = [runList[0],13306,max(runList)] # Run Breaks
			# #runBreaks = [runList[0],15800,max(runList)] # 2019 hard-code
			# rfilter = runList
			# #runBreaksTmp = [min(runList),7327,8600,12514,13217,max(runList)]
			# runBreaksTmp=[min(runList),4230,4391,4415,4711,5453,5475,5713,5955,6126,6429,6754,6930,7326,7490,\
# #			runBreaksTmp=[min(runList),4230,4672,5713,5955,6125,6429,6753,6960,7326,7490,\
							# 9767,9960,10936,10988,11085,11669,12516,13209,13307,14509,max(runList)+1]
			# # Old had				# 10936, 11217
			# # runBreaks2017Monitor = 4223
			# #runBreaksTmp=[min(runList),4227,4672,5713,5955,6125,6429,6753,6960,7326,7490,9768,9960,10940,10988,11085,11217,12514,13209,14508,max(runList)+1] #,13307
			# #runBreaksTmp = [min(runList),4230,4672,5713,6753,6960,7326,9768,9960,10940,10988,11085,11217,11898,12514,13209,14508,max(runList)+1]
			# runBreaksTmp.sort()
	
		# # This is the normalization algorithm here
		# # Might be easier to read if there's two steps in here? Separate out e.g. normalization and PSE?
		# dips = [0,1,2]
		
		# #chen_yu_likelihood_fit(runList,cts,nMon,dips)
		# #sys.exit()
		
		# # Singles Tmp.
		# test = False
		# if not test:
			# #try:
			# #rNT,hVT,nCT_S,mAT_S,ctsT_S,bST_S,nVT,nV1T,nV2T,nPT,p2T,p3T,s2nT = analyze_single_norm_condensed(runList,ctsTrunc,nMon,nMad,runBreaksTmp,nDet1,nDet2,dips,True,True)
				# #_N,_N,_N,_N,_N,bST1,_N,_N,_N,_N,_N,_N,_N = analyze_single_norm_condensed(runList,ctsSing,nMon,nMad,runBreaksTmp,nDet1,nDet2,dips,True,False)
				# #_N,_N,_N,_N,_N,bST2,_N,_N,_N,_N,_N,_N,_N = analyze_single_norm_condensed(runList,ctsSing,nMon,nMad,runBreaksTmp,nDet1,nDet2,dips,False,True)
			# try:
				# rNT,hVT,nCT_S,mAT_S,ctsT_S,bST_S,nVT,nV1T,nV2T,nPT,p2T,p3T,s2nT = analyze_single_norm_condensed(runList,ctsTrunc,nMon,nMad,runBreaksTmp,nDet1,nDet2,dips,True,True)
			# except:
				# nCT_S = []
				# ctsT_S = []
				# bST_S = []
				# bST1 = []
				# bST2 = []
				# pmt1S = []
				# pmt2S = []
				# for j in range(0,len(ctsSing)):
					# rNT,hVT,nCT_S_tmp,mAT_S,ctsT_S_tmp,bST_S_tmp,nVT,nV1T,nV2T,nPT,p2T,p3T,s2nT = analyze_single_norm_condensed(runList,ctsSing[j],nMon[j],nMad,runBreaksTmp,nDet1,nDet2,dips,False,True)
					# #_N,_N,_N,_N,pmt1_tmp,bST1_tmp,_N,_N,_N,_N,_N,_N,_N = analyze_single_norm_condensed(runList,ctsSing[j],nMon[j],nMad,runBreaksTmp,nDet1,nDet2,dips,True,False)
					# #_N,_N,_N,_N,pmt2_tmp,bST2_tmp,_N,_N,_N,_N,_N,_N,_N = analyze_single_norm_condensed(runList,ctsSing[j],nMon[j],nMad,runBreaksTmp,nDet1,nDet2,dips,False,True)
					# nCT_S.append(nCT_S_tmp)
					# ctsT_S.append(ctsT_S_tmp)
					# bST_S.append(bST_S_tmp)
					# #bST1.append(bST1_tmp)
					# #bST2.append(bST2_tmp)
					# #pmt1S.append(pmt1_tmp)
					# #pmt2S.append(pmt2_tmp)
				
				
				
			# # Coinc tmp.
			# #rNT,hVT,nCT,mAT,ctsT,bST,nVT,nV1T,nV2T,nPT,p2T,p3T,s2nT = analyze_single_norm_condensed(runList,ctsCoinc,nMon,nMad,runBreaksTmp,nDet1,nDet2,dips,True,True)
			# #_N,_N,nCT_C,mAT_C,ctsT_C,bST_C,_N,_N,_N,_N,_N,_N,s2nT_C = analyze_single_norm_condensed(runList,ctsCoinc,nMon,nMad,runBreaksTmp,nDet1,nDet2,dips,True,True)
		# #	else:
			
		# #rNT,hVT,nCT_S,mAT_S,ctsT_S,bST_S,nVT,nV1T,nV2T,nPT,p2T,p3T,s2nT = analyze_multivar(runList,ctsCoinc,nMon,nMad,runBreaksTmp,nDet1,nDet2)
		
		# # Parse by year
		# runNum.extend(rNT)
		# holdVec.extend(hVT)
		# #nCtsVec.extend(np.array(nCT_S[0])/np.array(nCT_S[1]))
		# #meanArr.extend(np.array(mAT_S)/np.array(mAT_C))
		# #ctsVec.extend(np.array(ctsT_S[0])/np.array(ctsT_S[1]))
		# #ctsVec.extend(np.array(ctsT_S[0])/np.array(ctsT_S[1]))
		# #bkgSubVec.extend(np.array(bST_S[0])/np.array(bST_S[1]))
		# #pmt1V.extend(np.array(pmt1S[0])/np.array(pmt1S[1]))
		# #pmt2V.extend(np.array(pmt2S[0])/np.array(pmt2S[1]))
		# #nCtsVec.extend(np.array(nCT_S)/np.array(nCT_C))
		# #meanArr.extend(np.array(mAT_S)/np.array(mAT_C))
		# #ctsVec.extend(np.array(ctsT_S)/np.array(ctsT_C))
		# #bkgSubVec.extend(np.array(bST_S)/np.array(bST_C))
		# nCtsVec.extend(np.array(nCT_S))
		# meanArr.extend(np.array(mAT_S))
		# ctsVec.extend(np.array(ctsT_S))
		# bkgSubVec.extend(np.array(bST_S))
		# try:
			# bkgSub1.extend(np.array(bST1))
			# bkgSub2.extend(np.array(bST2))
			# #bkgSub1.extend(np.array(bST1[0])/np.array(bST1[1]))
			# #bkgSub2.extend(np.array(bST2[0])/np.array(bST2[1]))
		# except NameError:
			# bkgSub1 = []
			# bkgSub2 = []
		# normVec.extend(nVT)
		# nVec1.extend(nV1T)
		# nVec2.extend(nV2T)
		# nPhiVec.extend(nPT)
		# pct2.extend(p2T)
		# pct3.extend(p3T)
		# sig2noi.extend(s2nT)
		# runBreaks.extend(runBreaksTmp)
	# #print out
	# holdT = []
	# timeV = [] # Need to convert from holdT vector to a measurement vector
	# timeE = []
	# if useMeanArr:
		# for x in meanArr: # Mean arrival time has a break
			# holdT.append(x)
			# timeV.append(float(x.val))
			# timeE.append(float(x.err))
	# else:
		# for x in holdVec:
			# holdT.append(measurement(float(x), 0.0))
			# timeV.append(float(x))
			# timeE.append(0.0)
	
	# tmpR = []
	# tmpY = []
	# tmpH = []
	#for i, run in enumerate(runNum):
		#if run==11984:
		#	print holdT[i], nCtsVec[i],normVec[i]
		#	print nCtsVec[i].err/nCtsVec[i].val
		
		#if 11985 <= run <11993:
		#if 11993 <= run <12001:
		#if 11985 <= run <12001:
		#	tmpR.append(run)
		#	tmpY.append(nCtsVec[i])
		#	tmpH.append(holdT[i])
			#print run, ctsVec[i]
		#	print run,nCtsVec[i].err/nCtsVec[i].val,nCtsVec[i]
	#ratio = (np.array(ctsVec)-np.array(normVec))/np.array(normVec)
	
	#plt.figure(50)
	#plt.plot(holdT,ratio,'b.')
	#plt.show()
	
	#plot_gen,filter1 = plot_expected_values(runNum, ctsVec,normVec,holdT,runBreaks, plot_gen,sig2noi)
	#plot_gen,filter2 = plot_expected_values(filter1,ctsVec,normVec,holdT,runBreaks, plot_gen,sig2noi)
	#plt.show()
	#write_all_runs(filter1)
	#sys.exit("Quitting for now")
	#calc_lifetime_globalChi2(ctsVec,normVec,holdT,sig2noi)
	#ltVec, runPair = pair_runs_from_file(runNum, nCtsVec, holdT)
	
#	ltVec, runPair,holdPair,ltVec2 = pair_runs(runBreaks, runNum, nCtsVec, holdT,normVec,0.9,sig2noi)
#	lifetimeExp  = calc_lifetime_exp(nCtsVec,holdT)
#	lifetimePair = calc_lifetime_paired(ltVec2,runPair)
#	lifetimePairUnw = calc_lifetime_unw_paired(ltVec,runPair)


	#ltVec, runPair = pair_runs(runBreaks,tmpR, tmpY, tmpH)
	#ltVec2, runPair2 = pair_runs_summed(runBreaks,runNum,ctsVec,holdT,normVec)
	#lifetimePair, runPair, ltVec = calc_lifetime_paired(runBreaks,runNum,holdT,nCtsVec)
	#lifetimeExp, scaleExp  = calc_lifetime_exp(nCtsVec,holdT)
	#lifetimeExpMore = calc_lifetime_ODR_meanFit(nCtsVec,holdT)

# #-----------------------------------------------------------------------
# # Controls for turning on/off things in initialize_global_booleans()
# def initialize_global_booleans():
	# # Numbers:
	# global holdSel
	# global w
	# global nDet1
	# global nDet2
	# global maxUnl
	# holdSel = 20.0 # Holding time for normalization (code adds 50s for hold later)
	# w = 5		   # num. of runs before/after to add. ("window" = 2w+1, so e.g. w=2 gives 5)
	# nDet1 = 3	   # normalization monitor 1 (3 = GV, 4 = RHC/Bare)
	# nDet2 = 5	   # normalization monitor 2 (5 = SP, 8 = RH/Foil)
	# maxUnl = 100.0 # Maximum amount of time to count in a single unload step
		
	# # Use various lifetime corrections
	# global coincLT
	# global singLT
	# global useMeanArr
	# global useDTCorr
	# global useBkgCorr
	# global usePosBkgs
	# global breaksOn
	# global use2017
	# global use2018
	# global sepRuns
	# global useBlock
	# global useRHC
	# coincLT    = True  # Calculate coincidence
	# singLT     = False  # Calculate singles
	# useMeanArr = True  # Mean arrival time vs. holding time
	# useDTCorr  = True  # Account for PMT deadtime
	# useBkgCorr = True  # Account for backgrounds
	# usePosBkgs = True  # Account for position sensitive backgrounds
	# breaksOn   = False  # Automatically generate runBreaks
	# use2017    = True  # Data set for 2017
	# use2018    = True  # Data set for 2018
	# sepRuns    = False  # Turn on "Separate Runs" to break data into groups
	# useBlock   = False  # Aluminum Block (for 2017 data only!)
	# useRHC     = False  # Round house cleaner moving period (for 2018 data only!)
		
	# # Turn on/off varying data sets
	# global pmt1
	# global pmt2
	# global dip1
	# global dip2
	# global dip3
	# global norm2All
	# global expoNorm
	# global geomNorm
	# pmt1       = True  # Turn on/off PMT1 -- Only for Singles
	# pmt2       = True  # Turn on/off PMT2 -- Only for Singles
	# dip1       = True  # Turn on/off dip1
	# dip2       = True  # Turn on/off dip2
	# dip3       = True  # Turn on/off dip3
	# norm2All   = True  # Use whole unload for normalization instead of individual dips
	# expoNorm   = True  # Use exponential method of normalization 
	# geomNorm   = False # Use geometric correction for normalization instead of exponential
		
	# # Turn on/off plots
	# global plotBreaks
	# global plotRaw
	# global plotNCts
	# global plotNHists
	# global plotBSub
	# global plotNorm
	# global plotPSE
	# global plotSig
	# global plotLTPair
	# global plotLTExp
	# plotBreaks = True
	# plotRaw    = False
	# plotNCts   = True
	# plotNHists = False
	# plotBSub   = False
	# plotNorm   = False
	# plotPSE    = False
	# plotSig    = False
	# plotLTPair = True
	# plotLTExp  = True
		
	# # Write out files
	# global writeAllRuns
	# global writeLTPairs
	# global writeLongY
	# writeAllRuns = False
	# writeLTPairs = False
	# writeLongY   = False
	
	# bData  = bool_ind_to_list_data(coincLT,singLT,use2017,use2018,sepRuns,useBlock,useRHC,breaksOn)
	# bNorm  = bool_ind_to_list_norm(useMeanArr,useDTCorr,useBkgCorr,usePosBkgs,pmt1,pmt2,dip1,dip2,dip3,norm2All,expoNorm,geomNorm)
	# bPlot  = bool_ind_to_list_plot(plotBreaks,plotRaw,plotNCts,plotNHists,plotBSub,plotNorm,plotPSE,plotSig,plotLTPair,plotLTExp)
	# bWrite = bool_ind_to_list_write(writeAllRuns,writeLTPairs,writeLongY)
	# nList  = number_list_params_ind(holdSel,w,nDet1,nDet2,maxUnl)
	
	# return nList,bData,bNorm,bPlot,bWrite
