#!/usr/local/bin/python3

#-----------------------------------------------------------------------
# These are because I fucked up and wrote a bunch of booleans to analyze
# but they were initially globals and I wanted to get away from one 
# 2500 line file because it was getting to be too unwieldy, so
# these functions convert a list to local bools and v.v.
#-----------------------------------------------------------------------
def bool_list_to_ind_data(bl):
	# Convert a list (for easy codes) to individual indices
	# This is for loading/creating data

	coincLT    = bl[0]  # Calculate coincidence
	singLT     = bl[1]  # Calculate singles
	use2017    = bl[2]  # Data set for 2017
	use2018    = bl[3]  # Data set for 2018
	sepRuns    = bl[4]  # Turn on "Separate Runs" to break data into groups
	useBlock   = bl[5]  # Aluminum Block (for 2017 data only!)
	useRHC     = bl[6]  # Round house cleaner moving period (for 2018 data only!)
	breaksOn   = bl[7]  # Automatically generate runBreaks
	
	return coincLT,singLT,use2017,use2018,sepRuns,useBlock,useRHC,breaksOn
def bool_ind_to_list_data(coincLT,singLT,use2017,use2018,sepRuns,useBlock,useRHC,breaksOn):
	# Convert a bunch of bools to a single object
	# This is for loading data
	
	bl = []
	bl.append(coincLT)
	bl.append(singLT)
	bl.append(use2017)
	bl.append(use2018)
	bl.append(sepRuns)
	bl.append(useBlock)
	bl.append(useRHC)
	bl.append(breaksOn)
	
	return bl	
	
def bool_list_to_ind_norm(bl):
	# Convert a list (for easy codes) to individual indices
	# This is for actual analysis stuff
	
	useMeanArr = bl[0]  # Mean arrival time vs. holding time
	useDTCorr  = bl[1]  # Account for PMT deadtime
	useBkgCorr = bl[2]  # Account for backgrounds
	usePosBkgs = bl[3]  # Account for position sensitive backgrounds
	pmt1       = bl[4]  # Turn on/off PMT1 -- Only for Singles
	pmt2       = bl[5]  # Turn on/off PMT2 -- Only for Singles
	dip1       = bl[6]  # Turn on/off dip1
	dip2       = bl[7]  # Turn on/off dip2
	dip3       = bl[8]  # Turn on/off dip3
	norm2All   = bl[9]  # Use whole unload for normalization instead of individual dips
	expoNorm   = bl[10] # Use exponential method of normalization 
	geomNorm   = bl[11] # Use geometric correction for normalization instead of exponential
	
	return useMeanArr,useDTCorr,useBkgCorr,usePosBkgs,pmt1,pmt2,dip1,dip2,dip3,norm2All,expoNorm,geomNorm
def bool_ind_to_list_norm(useMeanArr,useDTCorr,useBkgCorr,usePosBkgs,pmt1,pmt2,dip1,dip2,dip3,norm2All,expoNorm,geomNorm):
	# Convert a bunch of bools to a single object
	# This is for normalization
	
	bl = []
	bl.append(useMeanArr)
	bl.append(useDTCorr)
	bl.append(useBkgCorr)
	bl.append(usePosBkgs)
	bl.append(pmt1)
	bl.append(pmt2)
	bl.append(dip1)
	bl.append(dip2)
	bl.append(dip3)
	bl.append(norm2All)
	bl.append(expoNorm)
	bl.append(geomNorm)

	return useMeanArr,useDTCorr,useBkgCorr,usePosBkgs,pmt1,pmt2,dip1,dip2,dip3,norm2All,expoNorm,geomNorm

def bool_list_to_ind_plot(bl):
	# Convert a list (for easy codes) to individual indices
	# This is for plotting
	
	plotBreaks = bl[0] # Plot run breaks
	plotRaw    = bl[1] 
	plotNCts   = bl[2]
	plotNHists = bl[3]
	plotBSub   = bl[4]
	plotNorm   = bl[5]
	plotPSE    = bl[6]
	plotSig    = bl[7]
	plotLTPair = bl[8]
	plotLTExp  = bl[9]
	
	return plotBreaks,plotRaw,plotNCts,plotNHists,plotBSub,plotNorm,plotPSE,plotSig,plotLTPair,plotLTExp
def bool_ind_to_list_plot(plotBreaks,plotRaw,plotNCts,plotNHists,plotBSub,plotNorm,plotPSE,plotSig,plotLTPair,plotLTExp):
	# Convert a bunch of bools to a single object
	# This is for plotting
	
	bl = []
	bl.append(plotBreaks)
	bl.append(plotRaw)
	bl.append(plotNCts)
	bl.append(plotNHists)
	bl.append(plotBSub)
	bl.append(plotNorm)
	bl.append(plotPSE)
	bl.append(plotSig)
	bl.append(plotLTPair)
	bl.append(plotLTExp)
	
	return bl

def bool_list_to_ind_write(bl):
	# Convert a list (for easy codes) to individual indices
	# This is for writing out files

	writeAllRuns = bl[0]
	writeLTPairs = bl[1]
	writeLongY   = bl[2]
	
	return writeAllRuns,writeLTPairs,writeLongY
def bool_ind_to_list_write(writeAllRuns,writeLTPairs,writeLongY):
	# Convert a list (for easy codes) to individual indices
	# This is for writing out files

	bl = []
	bl.append(writeAllRuns)
	bl.append(writeLTPairs)
	bl.append(writeLongY)
	
	return bl

def	number_list_params_list(nl):
	# Convert a list (for easy codes) to individual indices
	# This is for numbers (not bools!)
	
	holdSel = nl[0] # Holding time for normalization (code adds 50s for hold later)
	w       = nl[1] # num. of runs before/after to add. ("window" = 2w+1, so e.g. w=2 gives 5)
	nDet1   = nl[2] # normalization monitor 1 (3 = GV, 4 = RHC/Bare)
	nDet2   = nl[3] # normalization monitor 2 (5 = SP, 8 = RH/Foil)
	maxUnl  = nl[4] # Maximum amount of time to count in a single unload step

	return holdSel,w,nDet1,nDet2,maxUnl	
def number_list_params_ind(holdSel,w,nDet1,nDet2,maxUnl):
	
	nl = []
	nl.append(holdSel)
	nl.append(w)
	nl.append(nDet1)
	nl.append(nDet2)
	nl.append(maxUnl)
	
	return nl
