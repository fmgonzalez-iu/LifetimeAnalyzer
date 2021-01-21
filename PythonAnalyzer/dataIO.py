#!/usr/local/bin/python
import os 			# Loading environmental variables
import configparser # Loading .conf files
import sys          # argv and 
import csv			# .csv file for runlists
import numpy as np	# numpy lists are useful!

import datetime		# For load_runs_to_datetime()
#import pdb

#from math import *

#from statsmodels.stats.weightstats import DescrStatsW
#from scipy import stats, special
#from scipy.odr import *
#from scipy.optimize import curve_fit, nnls
#from datetime import datetime
#import matplotlib.pyplot as plt

#from PythonAnalyzer.boolconversion import *
from PythonAnalyzer.classes import *
from PythonAnalyzer.backgrounds import bkgHgtDep,bkgTimeDep
#from PythonAnalyzer.plotting import *
#from PythonAnalyzer.runFilters import * 
#from PythonAnalyzer.pairingandlifetimes import *



def load_flex_lists(fileList,loadS,vb):
	# load_flex_lists loads whatever the lists are found in 
	
	# flexible file lists -- I've decided to increase flexibility in 
	# the number of files we can load.
	#
	# At some point I should clean the LifetimeAnalyzer.py software since it's a mess.
		
	#coincLT,singLT,use2017,use2018,sepRuns,useBlock,useRHC,breaksOn = bool_list_to_ind_data(bData)
	#_z,_z,_z,_z,sepRuns,useBlock,useRHC,_z = bool_list_to_ind_data(bData)
	if vb:
		print("Initializing loading process!")
		
	# Initializations (Lists for loading multiples at the same time
	runList = []
	badList = []
	ctsSing = []
	ctsCoin = []
	nMon    = []
	nMonL   = []
	nMad    = []
	nMadL   = []
	bkgsH   = []
	bkgsT   = []
	for f in fileList:
		# Order of loading:
		# Singles -> Coincidence -> Detector -> MAD -> RunLists
		# You don't need to actually put your files in this order!
		
		ctsSing_tmp, singLT   = load_counts_data_sing(f)
		ctsCoinc_tmp,coincLT  = load_counts_data_coinc(f)
		nMon_tmp,nMonList_tmp = load_det_data(f) # This is the only necessary one.
		nMad_tmp,nMadList_tmp = load_mad_data(f)
		runList_tmp           = load_run_list(f)
		bkgs_tmp              = load_bkg_data(f)
		bkgT_tmp              = load_tdep_data(f)
		# Now that we've tried all lists, let's go and concatenate 
		# things together.
		if singLT: # Coinc data can be loaded in a singles file?
			ctsSing.append(ctsSing_tmp)
		elif coincLT:
			ctsCoin.append(ctsCoinc_tmp)
		if len(nMonList_tmp) > 0:
			nMon.append(nMon_tmp)
			nMonL.extend(nMonList_tmp)
		if len(nMadList_tmp) > 0:
			nMad = nMad_tmp # OK so we can only load one MAD file. That's fine...
			nMadL.extend(nMadList_tmp)
		if len(runList_tmp) > 0:
			if len(badList) == 0:
				badList.extend(runList_tmp)
			else:
				runList.extend(runList_tmp)
		# Always have to load hdep first
		if len(bkgs_tmp) > 0 and len(bkgsH) <= len(bkgsT):
			bkgsH.append(bkgs_tmp)
		elif len(bkgT_tmp) > 0 and len(bkgsH) > len(bkgsT):
			bkgsT.append(bkgT_tmp)
		
	if len(nMonL) == 0:
		sys.exit("ERROR! At least one file must be a *-det.csv file!")
	if len(ctsSing) == 0 and len(ctsCoin) == 0:
		sys.exit("ERROR! At least one file must be a *-dag.csv file!")
	else:
		# Yo dawg I heard you like lists so I put lists in your lists so you can list while you list
		ctsList_cat = []
		for s in ctsSing:
			ctsListS_tmp = gen_cts_list(s)
			for c in ctsListS_tmp:
				ctsList_cat.append(c)
		for s in ctsCoin:
			ctsListC_tmp = gen_cts_list(s)
			for c in ctsListC_tmp:
				ctsList_cat.append(c)
		ctsList = np.unique(ctsList_cat)
		if len(ctsList) == 0:
			sys.exit("ERROR! Unable to load counts list!")
		
	nMad_out = []
	if len(nMadL) > 0:
		for n in nMon:
			nMad_out.extend(convert_mad_to_mon(n,nMad))
	else:
		nMad_out = []
	
	# Must always include unblinded lists
	if 'UNBLINDED_LIST' in os.environ:
		unblindedList = load_run_list(os.environ['UNBLINDED_LIST'])
	else:
		if vb:
			print("Defaulting to UNBLINDED list one folder out!")
		unblindedList = load_run_list("../UNBLINDED.csv")
	if len(unblindedList) > 0 and len(badList) > 0:
		badList.extend(unblindedList)
	elif len(unblindedList) > 0 and len(badList) == 0:
		badList = unblindedList	
	
	# background height dependence hardcoded in os
	if len(bkgsH) == 0:
		if 'BACKGROUND_HDEP' in os.environ:
			bkgsH = load_bkg_data(os.environ['BACKGROUND_HDEP'])
		else:
			if vb:
				print("Defaulting to hardcoded background height dependence!")
			bkgsH = load_bkg_data("/home/frank/FUCKED_MCA_Analysis/bkgHeightDep.csv")
	if len(bkgsT) == 0:
		if 'BACKGROUND_TDEP' in os.environ:
			bkgsT = load_tdep_data(os.environ['BACKGROUND_TDEP'])
		else:
			if vb:
				print("Defaulting to hardcoded background time dependence!")
			bkgsT = load_bkg_data("/home/frank/FUCKED_MCA_Analysis/bkgTimeDep.csv")
	#print(bkgs,len(bkgs))
	#runListParse = parse_run_lists(ctsList,nMonL,nMadL,badList,sepRuns,useRHC,useBlock)
	runListParse = parse_run_lists(ctsList,nMonL,nMadL,loadS,badList)
	if len(runList) == 0: # Either an external file or parsed through what we have
		runList = runListParse	
	else:
		runListBoth = []
		for r in runListParse:
			if r in runList:
				runListBoth.append(r)
		runList = runListBoth
					
	print("Found "+str(len(runList))+" runs with good monitor values.")
	print("   This is a ratio of: "+ \
			str(float(len(runList))/float(len(np.unique(ctsList)))))

	if len(ctsSing) == 1: # If we only have 1 file of type loaded, convert back to array
		ctsSing = np.array(ctsSing)
	if len(ctsCoin) == 1:
		ctsCoin = np.array(ctsCoin)
	if len(nMon) == 1:
		nMon = np.array(nMon)
	if len(nMad_out) == 1:
		nMad_out = np.array(nMad_out)
	#try:
	#	for b in bkgs:
	#		if b.is_run(run):
	#			bkg_out = b
	#except AtttributeError:
	#	bkgs = bkgs[0]
	if len(bkgsH) == 1:
		bkgsH = np.array(bkgsH[0])
	if len(bkgsT) == 1:
		bkgsT = np.array(bkgsT[0])
		
	return runList,ctsSing,ctsCoin,nMon,nMad_out,bkgsH,bkgsT

def load_configuration_file():
	# Load our configuration file.
	# We've initialized the structures in the classes.py subset so
	# if there's no config parser we'll just default to that.
	load = loading_cfg()
	anlz = analyzer_cfg()
	out  = output_cfg()
	
	vb = [False, True]
	config = configparser.ConfigParser()
	if 'ANALYSIS_CONFIG' in os.environ:
		config.read(os.environ['ANALYSIS_CONFIG'])
		# Verbose
		vb = config['Debug'].getboolean('Verbose_Loader')
		anlz.vb = config['Debug'].getboolean('Verbose_Analyzer')
		
		# Loading
		load.loadBreaks = config['Normalization'].getboolean('GenerateBreaks')
		load.minRun = int(config['Runs']['MinRun'])
		load.maxRun = int(config['Runs']['MaxRun'])
		load.preBlock  = config['Runs'].getboolean('PreBlock_2017')
		load.alBlock   = config['Runs'].getboolean('AlBlock_2017')
		load.postBlock = config['Runs'].getboolean('PostBlock_2017')
		load.rhc       = config['Runs'].getboolean('RHC_2018')
		load.mid2018   = config['Runs'].getboolean('Mid_2018')
		load.badDag    = config['Runs'].getboolean('BadDagger_2018')
		load.badBackground=config['Runs'].getboolean('BadBackground')
		load.lightLeaks  = config['Runs'].getboolean('LightLeaks')
		load.badTiming  = config['Runs'].getboolean('BadTiming')
		load.notProduction = config['Runs'].getboolean('NotProduction')	
	
		# Analyzer
		anlz.hold     = float(config['Normalization']['HoldSel'])
		anlz.w        = int(config['Normalization']['Window'])
		anlz.bkgWin   = int(config['Normalization']['BkgWindow'])
		anlz.det17    = [int(config['Normalization']['NDet1_2017']),\
						 int(config['Normalization']['NDet2_2017'])]
		anlz.det18    = [int(config['Normalization']['NDet1_2018']),\
						 int(config['Normalization']['NDet2_2018'])]
		anlz.expoNorm = config['Normalization'].getboolean('ExpoNorm')
		anlz.geomNorm = config['Normalization'].getboolean('GeomNorm')		
		anlz.useLong = config['Normalization'].getboolean('UseXtraLong')
		anlz.pmt1  = config['Systematics'].getboolean('PMT1')
		anlz.pmt2  = config['Systematics'].getboolean('PMT2')
		anlz.maxUnl = float(config['Systematics']['MaxUnl'])
		anlz.useMeanArr = config['Systematics'].getboolean('UseMeanArr')
		anlz.useDTCorr  = config['Systematics'].getboolean('UseDT')
		anlz.useBkgs    = config['Systematics'].getboolean('UseBkg')
		anlz.usePosBkgs = config['Systematics'].getboolean('UsePosBkgs')
		anlz.useTimeBkgs= config['Systematics'].getboolean('UseTimeBkgs')
		anlz.useMoving  = config['Systematics'].getboolean('UseMoving')
		anlz.scaleSing  = config['Systematics'].getboolean('ScaleSing')
		anlz.ndips  = int(config['Dips']['NDipsReq'])
		#anlz.thresh = True # Hard coding?
		if anlz.ndips == 3: # only doing single dip data for 3 dips for now.
			anlz.dips = []
			if config['Dips'].getboolean('Dip1'):
				anlz.dips.append(0)
			if config['Dips'].getboolean('Dip2'):
				anlz.dips.append(1)
			if config['Dips'].getboolean('Dip3'):
				anlz.dips.append(2)	
		else:
			anlz.dips = range(anlz.ndips)
		if config['Dips'].getboolean('Norm2All'):
			anlz.normDips = range(anlz.ndips)
		else:
			anlz.normDips = anlz.dips
		# Output 
		out.plotBreaks = config['Plotting'].getboolean('PlotBreaks')
		out.plotRaw    = config['Plotting'].getboolean('PlotRaw')
		out.plotNCts   = config['Plotting'].getboolean('PlotNCts')
		out.plotNHists = config['Plotting'].getboolean('PlotNHists')
		out.plotBSub   = config['Plotting'].getboolean('PlotBSub')
		out.plotNorm   = config['Plotting'].getboolean('PlotNorm')
		out.plotPSE    = config['Plotting'].getboolean('PlotPSE')
		out.plotSig    = config['Plotting'].getboolean('PlotSig')
		out.plotLTPair = config['Plotting'].getboolean('PlotLTPair')
		out.plotLTExp  = config['Plotting'].getboolean('PlotLTExp')
		out.writeAllRuns = config['Output'].getboolean('WriteAllRuns')
		out.writeLTPairs = config['Output'].getboolean('WriteLTPairs')
		out.writeLongY = config['Output'].getboolean('WriteLongY')
		
	return vb,load,anlz,out

def initialization_script():
	# Start by loading configuration file
	vb, load,anlz, out =  load_configuration_file()
	
	# Be nice to your user! They're probably super confused right now.
	# This is what we expect to read out of our program
	if vb:
		print("Beginning Analysis!")
		print("   If this is a singles file...")
		if anlz.pmt1 == True and anlz.pmt2 == True:
			print("    Using both PMTs for Singles Analysis!")
		elif anlz.pmt1 == True and anlz.pmt2 == False:
			print("    Using only PMT1 for Singles Analysis!")
		elif anlz.pmt2 == True and anlz.pmt1 == False:
			print("    Using only PMT2 for Singles Analysis!")
		else:
			print("    No PMT is presently on! Hopefully this is a coincidence set!")
			
		print("This program is presently looking at the lifetime in peak:")
		for d in anlz.dips:
			print("   %d" % (d+1))
		
		if len(anlz.dips) == len(anlz.normDips):
			print("Presently using all peaks for normalization!")
		else:
			print("Presently normalizing to just lifetime peaks!")
		print(" ")
		
		if load.loadBreaks == True:
			print("Note that runBreak auto-finder is presently ON!")
		else:
			print("Note that runBreak auto-finder is presently OFF!")
		
		print("This program is configured to make plots for:")
		if out.plotRaw:
			print("   Raw Singles")
		if out.plotNCts:
			print("   Normalized Signal")
		if out.plotNHists:
			print("   Normalized counts hist")
		if out.plotBSub:
			print("   Normalized Signal (Background Subtracted)")
		if out.plotNorm:
			print("   Normalization Factors")
		if out.plotPSE:
			print("   Phase Space Evolution")
		if out.plotLTPair:
			print("   Lifetime (Paired)")
		if out.plotLTExp:
			print("   Lifetime (Exponential)")
		if not (out.plotRaw or out.plotNCts or out.plotBSub or \
				out.plotNorm or out.plotPSE or out.plotLTPair or \
				out.plotLTExp):
			print("   Nothing!")
			
		# Write out files
		print("This program will write out runlists for:")
		if out.writeAllRuns:
			print("   All normalized runs")
		if out.writeLTPairs:
			print("   All paired runs")
		if out.writeLongY:
			print("   All long runs")
		if not (out.writeAllRuns or out.writeLTPairs or out.writeLongY):
			print("   Nothing!")
		print(" ")
	
	# And return...
	return vb,load,anlz, out

def gen_cts_list(cts = [],vb = True):
	# Generate the list of runs from our data
	
	ctsList = []
	if len(cts) > 0:
		try:
			ctsList = np.unique(cts['run']) # Arbitrary "Run" data
		except TypeError: # Maybe we have a list of lists?
			ctsList_tmp = []
			for c in cts:
				ctsList_tmp.extend(np.unique(c['run']))
			ctsList	= np.unique(ctsList_tmp)
		if vb:
			print("    List Contains Data From "+str(len(np.unique(ctsList)))+" Runs!")

	return ctsList	
	
def sort_counts_by_year(runList = [], conf = [], vb = True, ):
	# This will tell you which run set you're using
	
	use2017 = False
	use2018 = False
	use2019 = False
		
	if len(runList) > 0:
		
		# Set the year of the data set
		if min(runList) < 9600: # Hardcode year
			use2017 = True
		if max(runList) > 9600 and min(runList) <= 14999: 
			use2018 = True
		if max(runList) > 14999:
			use2019 = True
		if vb:
			print("This includes runs from")
			if use2017:
				print("   2017 data")
			if use2018:
				print("   2018 data")
			if use2019:
				print("   2019 data")
	
	# Sort the counts apart based on years:
	runSort = []
	ctsSort = []
	monSort = []
	detSort = []
	
	bounds  = [[4120,9600],[9600,14999],[14999,17400]] # Hardcoding run lists
	detHard = [conf.det17,conf.det18,conf.det18] # These are the detectors by year
	#detHard = [[3,5],[8,4],[8,4]] # These are the detectors we plan to use by year
	#detHard = [[3,5],[8,5],[8,5]] # These are the detectors we plan to use by year
	#detHard = [[3,0],[3,0],[3,0]]
	for i in range(0,3):
		runTmp = []
		for run in runList:
			if bounds[i][0] <= run < bounds[i][1]:
				runTmp.append(run)
		if len(runTmp) > 0:
			runSort.append(runTmp)
			detSort.append(detHard[i])
			
	return runSort,detSort
	
def load_counts_data_coinc(file_1 = [], vb = False): 
	# Load our lists of runs that we plan on using later.
	#-----------------------------------------------------------------------
	# Loading data from files here. Some of these are tough to find, so this
	# is a translation from old file to plain english -- useful when later
	# modifying c++ AnalyzerForeach code:
	# CTS: (Determines the counts data for each independent run)
	#	run  = Run #
	#	dip	 = Present dip
	#	ts	 = Start Time (of counting)
	#	te	 = End Time
	#   These next three can have i = 1, 2, or C depending on PMTs and Coincidences
	#	mi	 = Step Mean (just m for coinc)
	#   ctsi = Number of counts in our given detector (coinc for ctsC)
	#	dti  = Dead time counts -- Should be zero for all coinc at right values (just dt for coinc)
	#-----------------------------------------------------------------------
	coincLT = False	
	cts = []
	try:
		test = np.loadtxt(file_1,delimiter=',')
		if (np.size(test)/len(test)) == 7:
			cts = np.loadtxt(file_1, delimiter=",", dtype=[('run', 'i4'), ('dip', 'i4'), ('ts', 'f8'), ('te', 'f8'),('m','f8'),('coinc','f8'),('dt','f8')])
			print("   Loaded Coincidence Data!")
			coincLT = True
		else:
			if vb:
				print("   Data File Not Coincidence Format!")
	except IOError: # IOError means there was no file
		if vb:
			print("   WARNING! Improper file path!")
	except IndexError: # IndexError means there were more than len(dtype) entries in a line
		if vb:
			print("   Data File Not Coincidence Format!")
	return cts, coincLT

def load_counts_data_sing(file_1 = [],vb = False):
	# Load our lists of runs that we plan on using later.
	#-----------------------------------------------------------------------
	# Loading data from files here. Some of these are tough to find, so this
	# is a translation from old file to plain english -- useful when later
	# modifying c++ AnalyzerForeach code:
	# CTS: (Determines the counts data for each independent run)
	#	run  = Run #
	#	dip	 = Present dip
	#	ts	 = Start Time (of counting)
	#	te	 = End Time
	#   These next three can have i = 1, 2, or C depending on PMTs and Coincidences
	#	mi	 = Step Mean (just m for coinc)
	#   ctsi = Number of counts in our given detector (coinc for ctsC)
	#	dti  = Dead time counts -- Should be zero for all coinc at right values (just dt for coinc)
	#   diC  = Average number of counts in PMT for a coincidence
	#-----------------------------------------------------------------------

	singLT = False
	cts = []
	try: # This is a hacky way to check the type of run
		test = np.loadtxt(file_1,delimiter=',') 
		if (np.size(test)/len(test)) == 12:
			dtype_new = [('run', 'i4'),  ('dip', 'i4'),
						 ('ts', 'f8'),   ('te', 'f8'),
						 ('m1', 'f8'),   ('d1', 'i4'),
						 ('d1DT', 'f8'), ('d1C','f8'),
						 ('m2', 'f8'),   ('d2', 'i4'),
						 ('d2DT', 'f8'), ('d2C','f8')]
			cts = np.loadtxt(file_1, delimiter=",", dtype=dtype_new)
			print("   Loaded Singles Data!")
			singLT = True
		elif (np.size(test)/len(test)) == 10: # Old dtype
			dtype_old = [('run', 'i4'),('dip', 'i4'),
						 ('ts', 'f8'), ('te', 'f8'),
						 ('m1', 'f8'), ('d1', 'f8'), ('d1DT', 'f8'), 
						 ('m2', 'f8'), ('d2', 'f8'), ('d2DT', 'f8')]
			cts = np.loadtxt(file_1, delimiter=",", dtype=dtype_old)
			print("   Loaded Singles Data!")
			singLT = True
		else:
			if vb:
				print("   Data file not singles (old or new)!")
	except IOError: # IOError means there was no file
		if vb:
			print("  WARNING! Improper File Path!")
	except IndexError: # IndexError means there were more than 
		if vb:         # len(dtype) wntries in a line
			print("   Data file not singles (new format)! Attempting old singles format...")
		try:
			dtype_old = [('run', 'i4'),('dip', 'i4'),
						 ('ts', 'f8'), ('te', 'f8'),
						 ('m1', 'f8'), ('d1', 'f8'), ('d1DT', 'f8'),
						 ('m2', 'f8'), ('d2', 'f8'), ('d2DT', 'f8')]
			cts = np.loadtxt(file_1, delimiter=",", dtype=dtype_old)
			singLT = True
		except IndexError:
			if vb:
				print("   Data file not a singles (old OR new)!")
	
	return cts, singLT

def load_bkg_data(file_1 = [],vb = False):
	# Load our lists of backgrounds that we want to use
	
	# Blob together our background
	datatype = [('rmin','i4'),('rmax','i4'),
				('r1','f8'),('r1E','f8'),('r2','f8'),('r2E','f8'),('rC','f8'),('rCE','f8'),
				('h21','f8'),('h21E','f8'),('h31','f8'),('h31E','f8'),('h41','f8'),('h41E','f8'),
				('h22','f8'),('h22E','f8'),('h32','f8'),('h32E','f8'),('h42','f8'),('h42E','f8'),
				('h2C','f8'),('h2CE','f8'),('h3C','f8'),('h3CE','f8'),('h4C','f8'),('h4CE','f8')]
	bkgs = []
	try:
		test = np.loadtxt(file_1,delimiter=',')
		if (np.size(test)/len(test)) == 26:
			bkgs = np.loadtxt(file_1,delimiter=",", dtype=datatype)
			print("   Loaded Background File!")
		
		#if len(nMon) > 0:
		#	nMonList = nMon['run']
		#	print "Loaded monitor data from", len(np.unique(nMonList)), "runs!"
	except IOError:
		if vb:
			print("   WARNING! Improper File Path!")
		#nMon = []
	except IndexError:
		if vb:
			print("   File Not a Background-Height-Dep-Type!")
	bHD = []
	if len(bkgs) > 0:
		for b in bkgs:
			# Generate a temporary height dependent object
			bTemp = bkgHgtDep(b['rmin'],b['rmax'])
			# And push back our rates
			for i in range(1,len(bTemp.hgts)):
				n1 = 'h'+str(i+1)+'1'
				bTemp.pmt1[i]  = b[n1]
				bTemp.pmt1E[i] = b[n1+'E']
				n2 = 'h'+str(i+1)+'1'
				bTemp.pmt2[i]  = b[n2]
				bTemp.pmt2E[i] = b[n2+'E']
				nC = 'h'+str(i+1)+'1'
				bTemp.coinc[i]  = b[nC]
				bTemp.coincE[i] = b[nC+'E']
			bHD.append(bTemp)
			
	return bHD

def load_tdep_data(file_1 = [], vb = False):
	# Load our background time dependence
	
	datatype = [('rmin','i4'),('rmax','i4'),
				('a1','f8'),('a1E','f8'),('t1_1','f8'),('t1_1E','f8'),
				('b1','f8'),('b1E','f8'),('t2_1','f8'),('t2_1E','f8'),
				('a2','f8'),('a2E','f8'),('t1_2','f8'),('t1_2E','f8'),
				('b2','f8'),('b2E','f8'),('t2_2','f8'),('t2_2E','f8'),
				('aC','f8'),('aCE','f8'),('t1_C','f8'),('t1_CE','f8'),
				('bC','f8'),('bCE','f8'),('t2_C','f8'),('t2_CE','f8')]
				
	bkgs = []
	try:
		test = np.loadtxt(file_1,delimiter=',')
		if (np.size(test)/len(test)) == 26:
			bkgs = np.loadtxt(file_1,delimiter=",", dtype=datatype)
			print("   Loaded Background File!")
		
		#if len(nMon) > 0:
		#	nMonList = nMon['run']
		#	print "Loaded monitor data from", len(np.unique(nMonList)), "runs!"
	except IOError:
		if vb:
			print("   WARNING! Improper File Path!")
		#nMon = []
	except IndexError:
		if vb:
			print("   File Not a Background-Height-Dep-Type!")
	bTD = []
	if len(bkgs) > 0:
		for b in bkgs:
			# Generate a temporary height dependent object
			bTemp = bkgTimeDep(b['rmin'],b['rmax'])
			
			
			# And push back our factors
			# If they're consistent with zero, just push back zero (there's a bug somewhere)
			bTemp.pmt1  = [measurement(b['a1'],b['a1E']),measurement(b['b1'],b['b1E'])]
			bTemp.pmt2  = [measurement(b['a2'],b['a2E']),measurement(b['b2'],b['b2E'])]
			bTemp.coinc = [measurement(b['aC'],b['aCE']),measurement(b['bC'],b['bCE'])]
			# and the times
			bTemp.pmt1T  = [measurement(b['t1_1'],b['t1_1E']),measurement(b['t2_1'],b['t2_1E'])]
			bTemp.pmt2T  = [measurement(b['t1_2'],b['t1_2E']),measurement(b['t2_2'],b['t2_2E'])]
			bTemp.coincT = [measurement(b['t1_C'],b['t1_CE']),measurement(b['t2_C'],b['t2_CE'])]
			bTD.append(bTemp)
			
	return bTD
		
def load_det_data(file_1 = [],vb = False):
	# Load our lists of runs that we plan on using later.
	#-----------------------------------------------------------------------
	# NORMMON: (Determines normalization data for each independent run)
	#	run	 = Run #
	#	td   = Trap door time
	#	ts	 = Start of counting
	#	bkg1 = Bkg counts PMT1
	#	bkg2 = Bkg counts PMT2
	#	bkgC = Bkg counts Coinc
	#	bkgS = Background start
	#	bkgE = Background end
	#	moni = Monitor (i) weighted counts [i goes from 0 to 10]
	#	moniE= Monitor (i) weighted uncertainty 
	#-----------------------------------------------------------------------
	
	# Load from our 'detector' file (which has background and norm. information)
	monL = []
	datatypeRaw = [('run','i4'), ('td','f8'),    ('ts','f8'),
				   ('bkg1','f8'), ('bkg2','f8'), ('bkgC','f8'),
				   ('bkgS','f8'), ('bkgE','f8')]
	for i in range(1,11):
		monStr  = ('mon'+str(i))
		monEStr = ('mon'+str(i)+'E')
		monL.append((monStr, 'f8'))
		monL.append((monEStr,'f8'))
	datatype = datatypeRaw
	datatype.extend(monL)

	nMon = []
	try:
		test = np.loadtxt(file_1,delimiter=',')
		if (np.size(test)/len(test)) == 28:
			nMon = np.loadtxt(file_1,delimiter=",", dtype=datatype)
			print("   Loaded Detector File!")
		
		#if len(nMon) > 0:
		#	nMonList = nMon['run']
		#	print "Loaded monitor data from", len(np.unique(nMonList)), "runs!"
	except IOError:
		if vb:
			print("   WARNING! Improper File Path!")
		#nMon = []
	except IndexError:
		if vb:
			print("   File Not a Detector-Type!")
		#nMon = []

	# For only normalizing to one thing at a time
	mon0Str  = ('mon0','f8')
	mon0EStr = ('mon0E','f8')
	mon0 = []
	mon0.append(mon0Str)
	mon0.append(mon0EStr)
	datatype.extend(mon0)
	dtypeNull = datatype
	
	# Cast nMon to nMonNull
	if len(nMon) > 0:
		nMon_out = np.empty(len(nMon),dtype=dtypeNull) # Manually fill an empty array
		for i,l in enumerate(nMon): 
			nMon_tmp = np.zeros(1,dtype=dtypeNull)
			for t in dtypeNull: # Extend to fill null space 
				try:
					nMon_tmp[t[0]] = nMon[t[0]][i]
				except ValueError:
					nMon_tmp[t[0]] = 0.0
			nMon_out[i] = nMon_tmp
	else:
		nMon_out = []
	nMonList = gen_cts_list(nMon_out)
	if len(nMonList) > 0:
		print("     Contains", len(nMonList), "Runs!")
	return nMon_out,nMonList
	
def load_mad_data(file_3 = [], vb = True):
	# Load our lists of runs that we plan on using later.
	#-----------------------------------------------------------------------
	# NORMMAD: (Determines normalization data for each independent run from MAD)
	#	run	 = Run #
	#	det	 = detector
	#	moni = Monitor (i) MAD weighted counts [i goes from 1 to 10]
	#	moniE= Monitor (i) MAD weighed uncertainty
	#-----------------------------------------------------------------------
	
	# MAD information for geometric weighting
	nMad = []
	try:
		test = np.loadtxt(file_3,delimiter=',')
		if (np.size(test)/len(test)) == 4:
			nMad = np.loadtxt(file_3,delimiter=",", dtype=[('run','i4'),('det','i4'),('mon','f8'),('monE','f8')])
			
	except IOError:
		if vb:
			print("Not loading MAD norm data")
		#nMad = []
	
	# Now we convert MAD to the "det" style
	if len(nMad) > 0:
		nMadList = nMad['run']
		if vb:
			print("   Loaded MAD Norm Data From "+str(len(np.unique(nMadList)))+" Runs!")
	else:
		nMadList = []
		
	return nMad, nMadList

def convert_mad_to_mon(nMon = [],nMad = [], vb = True):
	# This copies nMad to the nMon style, assuming nMon actually exists.
	# We require nMon and nMad, and it'll speed up calculations if we input lists
		
	monL = [] # datatype from our 'detector' file (which has background and norm. information)
	datatype = [('run','i4'), ('td','f8'),('ts','f8'),('bkg1','f8'),('bkg2','f8'),('bkgC','f8'),('bkgS','f8'),('bkgE','f8')]
	for i in range(0,11):
		monStr  = ('mon'+str(i))
		monEStr = ('mon'+str(i)+'E')
		monL.append((monStr, 'f8'))
		monL.append((monEStr,'f8'))
	datatype.extend(monL)
		
	madL = np.unique(gen_cts_list(nMad,vb)) # Create a list
	
	nMad_out = np.empty(len(madL),dtype=datatype) # Manually fill an empty array
	for i,l in enumerate(madL): 
		
		nMon_tmp = nMon[nMon['run']==l]  #tmp MAD/monitor events
		nMad_by_det = nMad[nMad['run']==l] 
		nMad_tmp = np.zeros(1,dtype=datatype) # Output initialized as zero.
		
		nMad_tmp['run'] = l # Run info		
		if np.size(nMon_tmp['td'])>0: # Assuming monitor has data, copy all this over
			nMad_tmp['td'] = nMon_tmp['td'] # Hold timings
			nMad_tmp['ts'] = nMon_tmp['ts']
			nMad_tmp['bkg1'] = nMon_tmp['bkg1'] # Backgrounds
			nMad_tmp['bkg2'] = nMon_tmp['bkg2']
			nMad_tmp['bkgC'] = nMon_tmp['bkgC']
			nMad_tmp['bkgS'] = nMon_tmp['bkgS']
			nMad_tmp['bkgE'] = nMon_tmp['bkgE']
				
		for d in nMad_by_det['det']: # Figure out which monitors we have data for
			monStr  = ('mon'+str(d)) # And parse these strings!
			monEStr = ('mon'+str(d)+'E')
			
			# If the monitor counts are empty, set 0
			if len(nMad_by_det[nMad_by_det['det']==d]['mon']) == 0:
				nMad_tmp[monStr] = 0.0
			else:
				nMad_tmp[monStr] = nMad_by_det[nMad_by_det['det']==d]['mon']
			if len(nMad_by_det[nMad_by_det['det']==d]['monE']) == 0:
				nMad_tmp[monEStr] = 0.0
			else:
				nMad_tmp[monEStr] = nMad_by_det[nMad_by_det['det']==d]['monE']
			if len(nMad_tmp[monStr]) > 1 and vb:
				print("ERROR! Run "+str(l)+" has multiple monitor values!")
			
				
		nMad_out[i] = nMad_tmp # push back data
		
	return nMad_out

def load_runs_to_datetime(runList = [], inFName='/home/frank/run_and_start.txt'):
	#-----------------------------------------------------------------------
	# This script is designed to read through run_and_start.txt files
	# Converts a list of runs into datetime values
	#-----------------------------------------------------------------------

	dateList  = []
	dateListL = []
	try:
		inF = open(inFName, 'r')
		#run_input = csv.DictReader(inF, fieldnames=['run_number','start_time'], delimiter=',')
		runs = np.loadtxt(inF, delimiter=',', dtype=[('run_number','i4'),('start_time','S20')])
	except IOError:
		print("ERROR: Unable to load datetime file!")
		return dateList
	inF.close()
	for rs, rl in runList:
		# Format of t is 'yyyy-mm-dd hh:mm:ss'
		t = np.array2string(runs[runs['run_number']==rs]['start_time']) # Direct conversion keeps brackets
		dt = datetime.datetime.strptime(t[3:-2],"%Y-%m-%d %H:%M:%S")
		dateList.append(dt)
		tL = np.array2string(runs[runs['run_number']==rl]['start_time']) # Direct conversion keeps brackets
		dtL = datetime.datetime.strptime(tL[3:-2],"%Y-%m-%d %H:%M:%S")
		dateListL.append(dtL)
		
	return dateList,dateListL
	
def load_run_list(file_N = [],vb = False): 
	# This is just a quick script to load a file list csv

	runList = []
	try:
		if(len(file_N) > 0):
			with open(file_N, 'r') as f: #TODO: This fails if there's a comma at the end
				runList = [list(map(int,rl)) for rl in csv.reader(f,delimiter=',')][0]
			if len(runList) < 2:
				runList = np.loadtxt(file_N)		
		print("   Loaded List of Runs! (",len(runList),")")
	except IOError:
		if vb:
			print("Unable to sucessfully load run!")
	except ValueError:
		if vb:
			print("Not a runList type file!")
	
	
	#rL = []
	#for r in runList:
	#	if r <13200:
	#		rL.append(r)
	#return rL
	return runList

def parse_run_lists(cL = [],monL = [],madL = [],load = [],badL = []):
	# This function will parse monitor lists, and has some hardcoded stuff too
	# Hardcoding is separating out runs and the like
	
	mon = False
	mad = False
	if not (len(cL) > 0 and (len(monL) > 0 or len(madL) > 0)):
		sys.exit("No counts/monitor lists loaded! Ending LifetimeAnalyzer...")
	else: # Error checking for monitor listings
		if len(monL) > 0: 
			mon = True
		if len(madL) > 0:
			mad = True
	
	rL = []
	if len(cL) > 0:
		for run in cL:
		
			if run in badL: # Hand-picked "Bad" runs -- from file
				continue
		
			# Start with the specific min/max presets
			if not (load.minRun <= run <= load.maxRun): 
				continue
			# Generic regions
			if not load.preBlock: 
				if 4200 <= run < 4711:
					continue
			if not load.alBlock:
				if 4711 <= run < 7326:
					continue
			if not load.postBlock:
				if 7326 <= run < 9600:
					continue
			if not load.rhc:
				if 9600 <= run < 11669:
					continue
			if not load.mid2018:
				if 11669 <= run < 13209:
					continue
			if not load.badDag:
				if 13209 <= run < 14516:
					continue
			
			if not load.badBackground:
				# These all have coincidence rates > 50 Hz. That seems wildly wrong.
				# My guess is actually some tagbit miscalculation.
				if 8600 <= run < 8733: # Might be bad tagbits honestly.
					continue
				if 8761 <= run < 8860: # More possible bad tagbits
					continue
				if 8150==run or 10722==run: # Single bad runs
					continue
				
				# These are runs that have a noise spike.
				# Coincidence -- 
				if run==4920 or run == 7230:
					continue
				if run==6414 or run == 7712:
					continue	
	
				# Singles -- 
				if run==4368:
					continue
				if run==4636:
					continue
				if run==6316:	
					continue
				if run==7427:
					continue
				if run==10994:
					continue
				if run==11278:
					continue
				if run==12300:
					continue
				if run==12970:
					continue
					
				# Difference between singles and coinc. --
				if run==5789:
					continue
				if run==6628:
					continue
			if not load.lightLeaks:
				# Light leaks ------------------------------------------
				if 10496 <= run < 10504: # Maybe less bad one
					continue
				if 10645 <= run < 10680: # Light leak or something -- also leads to PSE
					continue	
				if 11937 <= run < 11982: # Abnormal amounts of Phase Space Evolution
					continue
				if 12649 <= run < 12671: # Phase Space Evolution (light leak)
					continue
			if not load.badTiming:
				# These are runs that don't start at 0 -- this affects the fill and mean arrival time
				if run==11082: # This is actually a super-long run that's not normalizable 
					continue
				if run==11786:
					continue
				if run==12568:
					continue
				if run==12590:
					continue
				if run==12737:
					continue
					
			if not load.notProduction:
			# Hardcoded "Bad Runs" (these are non-production things)
				if run < 4200: # Start of production running (There's bad runs before here)
					continue
				if 4223 <= run < 4230: # spin flipper tuning
					continue
				#if 4391 <= run < 4406: # Rate is obnoxiously low
				#	continue
				if 4240 <= run < 4273: # spin flipper tuning
					continue
				if 8598 <= run < 8609: # RH filltime tuning
					continue
				if 8725 <= run < 8761: # Dark matter background runs (no UCN in trap)
					continue
				if 8595 == run: # Roundhouse but bad comment
					continue
			
				# # 2018 -----------------------------------------------
				if 9957 <= run < 9997: # Monitor weirdness
					continue
				if 10935 < run < 10948: # Spin flipper tuning, not running.
					continue
				if 11669 <= run < 11681: # Spin flipper tuning
					continue
				if 14711 <= run < 14723: # spin flipper tuning
					continue
				if 14050 <= run < 14059: # No cleaning! But somehow doesn't get tagged as such
					continue
			
				# 2019 -------------------------------------------------
				if 15285 < run < 15352:
					continue
				if 15426 < run < 15460:
					continue
				if 16205 < run < 16214:
					continue
				if 16177 == run:
					continue
				if 16175 < run < 16195:
					continue
		
			if mon and mad: # parsing 3 lists
				if (run in madL) and (run in monL): # If run is in both
					rL.append(run)
			elif mon: # Just exponential monitor counts
				if run in monL:
					rL.append(run)
			elif mad: # Just geometric monitor counts
				if run in madL:
					rL.append(run)
			else:
				print("Boo!")
				break
	print("Successfully loaded "+str(len(rL))+" runs!")
	return rL

# def load_counts_lists(bData,nList,file_1, file_2, file_3 = [], file_4 = [], file_5=[]):# Load Data From File
		
	# coincLT,singLT,use2017,use2018,sepRuns,useBlock,useRHC,breaksOn = bool_list_to_ind_data(bData)
	# holdSel,w,nDet1,nDet2,maxUnl =number_list_params_list(nList)
	
	# # These runs are unblinded and should always be excluded
	
	# unblindedList = load_run_list("../UNBLINDED.csv")
	
	# cts_sing, singLT  = load_counts_data_sing(file_1)
	# cts_coinc,coincLT = load_counts_data_coinc(file_1)
	
	# if singLT:
		# cts = cts_sing
	# elif coincLT:
		# cts = cts_coinc
	# else:
		# sys.exit("ERROR! Unable to load unload data!")
	# ctsList = gen_cts_list(cts)
	
	# nMon,nMonList = load_det_data(file_2)
	# nMad,nMadList = load_mad_data(file_3)
	
	# nMad_out = convert_mad_to_mon(nMon,nMad)
		
	# badRuns = load_run_list(file_4)	# Extra bad runs (where MAD doesn't show up or something)
	# if len(unblindedList) > 0 and len(badRuns) > 0:
		# badRuns.extend(unblindedList)
	# elif len(unblindedList) > 0 and len(badRuns) == 0:
		# badRuns = unblindedList	
	# runList = load_run_list(file_5) # Get the list of runs we plan on processing.
	
	# if not len(runList) > 0: # Either an external file or parsed through what we have
		# runList = parse_run_lists(ctsList,nMonList,nMadList,badRuns,sepRuns,useRHC,useBlock)
					
	# print("Found "+str(len(runList))+" runs with good monitor values. This is a ratio of: "+str(float(len(runList))/float(len(np.unique(ctsList)))))
	# bData = bool_ind_to_list_data(coincLT,singLT,use2017,use2018,sepRuns,useBlock,useRHC,breaksOn)

	# return runList, cts, nMon, nMad_out,bData
# def initialization_script(nList,bData,bNorm,bPlot,bWrite):
	
	# holdSel,w,nDet1,nDet2,maxUnl =number_list_params_list(nList)
	# coincLT,singLT,use2017,use2018,sepRuns,useBlock,useRHC,breaksOn = bool_list_to_ind_data(bData)
	# useMeanArr,useDTCorr,useBkgCorr,usePosBkgs,pmt1,pmt2,dip1,dip2,dip3,norm2All,expoNorm,geomNorm = bool_list_to_ind_norm(bNorm)
	# plotBreaks,plotRaw,plotNCts,plotNHists,plotBSub,plotNorm,plotPSE,plotSig,plotLTPair,plotLTExp = bool_list_to_ind_plot(bPlot)
	# writeAllRuns,writeLTPairs,writeLongY = bool_list_to_ind_write(bWrite)
	
	# # Be nice to your user! They're probably super confused right now.
	# # This is what we expect to read out of our program
	# if singLT == True:
		# if pmt1 == True and pmt2 == True:
			# print("Using both PMTs for Singles Analysis!")
		# elif pmt1 == True and pmt2 == False:
			# print("Using only PMT1 for Singles Analysis!")
		# elif pmt2 == True and pmt1 == False:
			# print("Using only PMT2 for Singles Analysis!")
		# else:
			# sys.exit("Error! No PMT is presently on! Turn on a PMT (variable pmt1 or pmt2) for analysis!")
	# elif coincLT == True:
		# print("Beginning Coincidence Analysis!")
	# else:
		# sys.exit("Error! Not doing either Singles or Coincidence Analysis! Turn on one of these!")
	
	# if singLT == True and coincLT == True:
		# sys.exit("Error! Both singles and coincidence lifetimes on right now! Only do one of them!")
	
	# print("This program is presently looking at the lifetime in peak:")
	# if dip1 == True:
		# print("   1")
	# if dip2 == True:
		# print("   2")
	# if dip3 == True:
		# print("   3")
	
	# if norm2All == True:
		# print("Presently using all peaks for normalization!")
	# else:
		# print("Presently normalizing to just lifetime peaks!")
	# print(" ")
	
	# if breaksOn == True:
		# print("Note that runBreaks is presently ON!")
	# else:
		# print("Note that runBreaks is presently OFF!")
	
	# print("This program is configured to make plots for:")
	# if plotRaw == True:
		# print("   Raw Singles")
	# if plotNCts == True:
		# print("   Normalized Signal")
	# if plotNHists == True:
		# print("   Normalized counts hist")
	# if plotBSub == True:
		# print("   Normalized Signal (Background Subtracted)")
	# if plotNorm == True:
		# print("   Normalization Factors")
	# if plotPSE == True:
		# print("   Phase Space Evolution")
	# if plotLTPair == True:
		# print("   Lifetime (Paired)")
	# if plotLTExp == True:
		# print("   Lifetime (Exponential)")
	# if not (plotRaw or plotNCts or plotBSub or plotNorm or plotPSE or plotLTPair or plotLTExp):
		# print("   Nothing!")
		
	# # Write out files
	# print("This program will write out runlists for:")
	# if writeAllRuns == True:
		# print("   All normalized runs")
	# if writeLTPairs == True:
		# print("   All paired runs")
	# if writeLongY == True:
		# print("   All long runs")
	# if not (writeAllRuns or writeLTPairs or writeLongY):
		# print("   Nothing!")
	# print(" ")
	# return 0
