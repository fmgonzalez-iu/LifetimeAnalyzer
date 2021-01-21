#!/usr/local/bin/python3

import sys
import pdb
from math import *
import numpy as np
from scipy import stats, special
from scipy.optimize import curve_fit
from datetime import datetime
import matplotlib.pyplot as plt

from PythonAnalyzer.classes import measurement
#-----------------------------------------------------------------------
# This background analyzer code is designed to separate time dependence
# from other possible sources of backgrounds
#
# Pseudocode -- What we want to do here:
# (a) Look at beam off backgrounds to find mean/stderr corr. for heights
# (b) Separate other backgrounds by heights and scale accordingly
# (c) Look at time dependence as a function of heights
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# Function and Class Definitions
#-----------------------------------------------------------------------
class background:
	def __init__(self,run,mt,pmt1,pmt2,coinc,temp1,temp2):
		self.run = run
		self.mt  = mt
		self.pmt1  = pmt1
		self.pmt2  = pmt2
		self.coinc = coinc
		self.temp1 = temp1
		self.temp2 = temp2

class bkgNorm:
	def __init__(self, bkg1, bkg2):
		self.run = bkg1.run
		self.t0  = bkg1.mt
		self.td  = bkg2.mt - bkg1.mt
		self.pmt1 = bkg2.pmt1/bkg1.pmt1
		self.pmt2 = bkg2.pmt2/bkg1.pmt2
		self.coinc = bkg2.coinc/bkg1.coinc
		
class bkgRed:
	def __init__(self,pmt1,pmt2,coinc):
		self.pmt1 = pmt1
		self.pmt2 = pmt2
		self.coinc = coinc
	
	def __add__(b1,b2):
		return bkgRed(b1.pmt1+b2.pmt1, b1.pmt2+b2.pmt2, b1.coinc+b2.coinc)
	
	def __sub__(b1,b2):
		return bkgRed(b1.pmt1-b2.pmt1, b1.pmt2-b2.pmt2, b1.coinc-b2.coinc)
	
	def __mul__(b1,b2):
		return bkgRed(b1.pmt1*b2.pmt1, b1.pmt2*b2.pmt2, b1.coinc*b2.coinc)
	def __div__(b1,i):
		if (i==0):
			return bkgRed(b1.pmt1,b1.pmt2,b1.coinc)
		else:
			return bkgRed(b1.pmt1/i, b1.pmt2/i, b1.coinc/i)
	def __truediv__(b1,i):
		if (i==0):
			return bkgRed(b1.pmt1,b1.pmt2,b1.coinc)
		else:
			return bkgRed(b1.pmt1/i, b1.pmt2/i, b1.coinc/i)
		
	def __pow__(b1,i):
		return bkgRed(b1.pmt1**i,b1.pmt2**i,b1.coinc**i)
	
#-----------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------
def goodBkg(bkg):
	if ((bkg.pmt1 > 10) & (bkg.pmt2 > 10) & (bkg.coinc > 0)):
		return True
	else:
		return False

def bkgRate(X, a, b, c):
	# Background rate function -- linear in T, quadratic in height.
	# I use this for plotting lines
	T, h = X
	return a*T + b*(0.001*h)**2 + c
	#return np.exp(T/a) + b*(0.001*h)**2 + c
	
def bkgTime(x,a,b):
	t, r = x
	return a*np.exp(-t/b) + r

def bkgTimeLinear(x,a,b):
	return a + b*x

global bkgExp
def doubleExp(x,a,t1,b,t2):
	global bkgExp
	if t1 < 0 or t2 < 0:
		# Can't have a negative time constant
		return np.inf
	return a*np.exp(-x/t1) + b*np.exp(-x/t2) + bkgExp
def doubleExp_bkg(x,a,t1,b,t2,bkg):
	if t1 < 0 or t2 < 0:
		# Can't have a negative time constant
		return np.inf
	return a*np.exp(-x/t1) + b*np.exp(-x/t2) + bkg
#-----------------------------------------------------------------------

# Initial loading function for our output.
if(len(sys.argv) != 3):
	sys.exit("Error! Usage: python background_general.py bkg_file_name bkg_tdep_folder")

#-----------------------------------------------------------------------
# Loading data from files here. 
# This is a translation from .csv to plain english.
# These files are generated through "functionality 3" on AnalyzerForeach
# CTS: (Determines the counts data for each independent run)
#	run  = "Run #"
#   hgt  = height -- hardcoded into run
#	ts	 = Start Time (of counting)
#	te	 = End Time
#   pmt1 = pmt1 counts
#   pmt2 = pmt2 counts
#	coinc= Number of Coincidence hits
#-----------------------------------------------------------------------

# "cts" contains all the data from our bkgRunByDip.csv.
cts = np.loadtxt(sys.argv[1], delimiter=",", dtype=[('run', 'i4'), ('hgt', 'i4'), ('ts', 'f8'), ('te', 'f8'), ('pmt1', 'f8'), ('pmt2', 'f8'), ('coinc', 'f8'), ('temp1','f8'), ('temp2','f8')])
	
# get the list of background runs
runList = cts['run']

# Search region -- switch for pre/post dagger runs
# Dagger runBreaks:
rBreakFull = [4200,7327,9600,13309,15000]

dag2017 = False
dag2018 = True
preDag = True
postDag= False
if dag2017 and not dag2018:
	if preDag and postDag:
		rBreak = [4600,9600]
	elif preDag and not postDag:
		rBreak = [4600,7327]
	elif postDag and not preDag:
		rBreak = [7327, 9600]
	else:
		rBreak = [0,9600]
	fillTime = 150.0
elif dag2018 and not dag2017:
	if preDag and postDag:
		rBreak = [9600, 15000]
	elif preDag and not postDag:
		rBreak = [9600, 13309]
	elif postDag and not preDag:
		rBreak = [13309, 15000]
	else:
		rBreak = [9600,99999]
	fillTime = 300.0
else:
	fillTime = 150.0
	rBreak = [0,99999]

hDepBkg = open("bkgHeightDep.csv","w") # Write a bkgHeightDep csv

# Shitty coding here
muCts = [] # List of average PMTs
muStd = []
hDCts = [] # List of 49cm cts
hDStd = []
for r in range(len(rBreakFull)-1):
	rBreakTmp = [rBreakFull[r],rBreakFull[r+1]]
	
	# Initialize lists for each height -- 1, 25, 38, 49 cm each (shortened to 1/2/3/4)
	# Beam off
	day1 = []
	day2 = []
	day3 = []
	day4 = []
	# Beam on
	bOn1 = []
	bOn2 = []
	bOn3 = []
	bOn4 = []
	
	badRun = 0
	# Extract data from each line of our input file
	for row in cts:
		
		# Extract data from cts
		runNo = row['run']
		# make sure we are in the right region -- hardcoding PMT changes above
		if not (rBreakTmp[0] <= runNo < rBreakTmp[1]):
			continue

		meanTime = (row['te'] + row['ts'] ) / 2
		t1 = row['temp1']
		t2 = row['temp2']
		
		# convert raw counts to rates
		pmt1 = row['pmt1'] / (row['te'] - row['ts'])
		pmt2 = row['pmt2'] / (row['te'] - row['ts'])
		coinc = row['coinc'] / (row['te'] - row['ts'])
				
		# check that we're using a good background (want a non-zero rate)
		if not((pmt1 > 0) & (pmt2 > 0) & (coinc > 0)):
			if runNo != badRun:
				print("Bad background for run ", runNo)
				badRun = runNo
			continue
		if (10720 <= runNo < 10740):
			# These runs are uncooled or something.
			continue
		# sort out beam off background runs -- should be 250 - 2*10 seconds on each step
		if (225.0 < (row['te'] - row['ts']) < 235.0):
			if (row['hgt'] == 10):
				day1.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
			elif (row['hgt'] == 250):
				day2.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
			elif (row['hgt'] == 380):
				day3.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
			elif (row['hgt'] == 490):
				day4.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
			else:
				print("Unknown height for run", runNo, "!!")
				
		# Beam-on background runs should all be less than 200s, but there's a timing glitch now to also fix.
		# The timings are 100s (immediate) and 40, 20, 150, 50 (or with glitch 150, 170, 210) (s/l pairs)
		else:
			if (row['hgt'] == 10):
				bOn1.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
			elif (row['hgt'] == 250):
				bOn2.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
			elif (row['hgt'] == 380):
				bOn3.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
			elif (row['hgt'] == 490):
				bOn4.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
			else:
				print("Unknown height for run", runNo, "!!")
	
	# Normalize each run to a height of 10
	n1 = []
	n2 = []
	n3 = []
	n4 = []
	
	# keep track of mean values for each height (using reduced backgrounds
	mu1 = bkgRed(0.0,0.0,0.0) # This is raw counts
	mu2 = bkgRed(1.0,1.0,1.0) # These will be scaling factors
	mu3 = bkgRed(1.0,1.0,1.0)
	mu4 = bkgRed(1.0,1.0,1.0)
	print(len(day1))
	# Match run numbers of beam-off backgrounds to get normalized data
	for r1 in day1:
			
		# Add to mean and increment counter
		mu1 += bkgRed(r1.pmt1,r1.pmt2,r1.coinc)
		n1.append(bkgRed(r1.pmt1,r1.pmt2,r1.coinc))
		
		# Match run numbers and add to means		
		for r2 in day2:
			if (r1.run == r2.run):
				n2.append(bkgNorm(r1, r2))
				mu2 += bkgNorm(r1,r2)
		for r3 in day3:
			if (r1.run == r3.run):
				n3.append(bkgNorm(r1, r3))
				mu3 += bkgNorm(r1,r3)
		for r4 in day4:
			if (r1.run == r4.run):
				n4.append(bkgNorm(r1, r4))
				mu4 += bkgNorm(r1,r4)
	
	# Divide out means and calculate std. error of the mean:
	if len(n1) > 0:
		mu1 = mu1 / len(n1)
		mu2 = mu2 / len(n2)
		mu3 = mu3 / len(n3)
		mu4 = mu4 / len(n4)
	
	sd1 = bkgRed(0.0,0.0,0.0)
	sd2 = bkgRed(0.0,0.0,0.0)
	sd3 = bkgRed(0.0,0.0,0.0)
	sd4 = bkgRed(0.0,0.0,0.0)
	# Note that due to our object format we have to do this (can't use np.std)
	for r1 in n1:
		diff = mu1 - r1
		sd1 += diff*diff
	if len(n1) > 0:
		sd1 = (sd1**0.5) / len(n1)
	
	for r2 in n2:
		diff = mu2 - r2
		sd2 += diff*diff
	if len(n2) > 0:
		sd2 = (sd2**0.5) / len(n2)
	
	for r3 in n3:
		diff = mu3 - r3
		sd3 += diff*diff
	if len(n3) > 0:
		sd3 = (sd3**0.5) / len(n3)
	
	for r4 in n4:
		diff = mu4 - r4
		sd4 += diff*diff
	if len(n4) > 0:
		sd4 = (sd4**0.5) / len(n4)

	print("\nAverage 1 cm Background Rates (runs between", rBreakTmp[0],"and", rBreakTmp[1],"):")
	print("   PMT 1:", mu1.pmt1, "+/-", sd1.pmt1)
	print("   PMT 2:", mu1.pmt2, "+/-", sd1.pmt2)
	print("   Coinc:", mu1.coinc, "+/-", sd1.coinc)
	
	print("\nPMT 1 Scaling:")
	print("   25cm:", mu2.pmt1, "+/-", sd2.pmt1)
	print("   38cm:", mu3.pmt1, "+/-", sd3.pmt1)
	print("   49cm:", mu4.pmt1, "+/-", sd4.pmt1)
	print("PMT 2 Scaling:")
	print("   25cm:", mu2.pmt2, "+/-", sd2.pmt2)
	print("   38cm:", mu3.pmt2, "+/-", sd3.pmt2)
	print("   49cm:", mu4.pmt2, "+/-", sd4.pmt2)
	print("Coinc Scaling:")
	print("   25cm:", mu2.coinc, "+/-", sd2.coinc)
	print("   38cm:", mu3.coinc, "+/-", sd3.coinc)
	print("   49cm:", mu4.coinc, "+/-", sd4.coinc)
	hDepBkg.write("%d,%d," % (rBreakTmp[0],rBreakTmp[1])) # Runs
	hDepBkg.write("%f,%f,%f,%f,%f,%f," % (mu1.pmt1,sd1.pmt1,mu1.pmt2,sd1.pmt2,mu1.coinc,sd1.coinc)) # Rates
	hDepBkg.write("%f,%f,%f,%f,%f,%f," % (mu2.pmt1,sd2.pmt1,mu3.pmt1,sd3.pmt1,mu4.pmt1,sd4.pmt1)) # PMT 1
	hDepBkg.write("%f,%f,%f,%f,%f,%f," % (mu2.pmt2,sd2.pmt2,mu3.pmt2,sd3.pmt2,mu4.pmt2,sd4.pmt2)) # PMT 2
	hDepBkg.write("%f,%f,%f,%f,%f,%f\n" % (mu2.coinc,sd2.coinc,mu3.coinc,sd3.coinc,mu4.coinc,sd4.coinc)) # Coinc.
	muCts.append(mu1) # Don't care about intermediate steps for time dependence
	muStd.append(sd1)
	hDCts.append(mu4)
	hDStd.append(sd4)
hDepBkg.close()

# Time Dependent histograms
tDepBkg = open("bkgTimeDep.csv","w")
for r in range(len(rBreakFull)-1):
	name = sys.argv[2]+'/holdingBkgtot.csv'+str(rBreakFull[r])
	tCts = np.loadtxt(name, delimiter=",", dtype=[('bin', 'i4'), ('pmt1', 'f8'),('pmt2','f8'),('coinc','f8')])
	#tCts = 0
	bins_raw = tCts['bin'][21:] * (len(tCts['bin'][21:]) / 1549.) # in case we went finer
	if tCts['pmt1'][0] > 0:
		tdPmt1_raw = tCts['pmt1'][21:] / tCts['pmt1'][0]
	else:
		tdPmt1_raw = tCts['pmt1'][21:]*0.
	global bkgExp
	bkgExp = muCts[r].pmt1 #s* hDCts[r].pmt1
	#tdPmt1 -= muCts[r].pmt1 * hDCts[r].pmt1 # Background Subtraction
	if tCts['pmt2'][0] > 0:
		tdPmt2_raw = tCts['pmt2'][21:] / tCts['pmt2'][0]
	else:
		tdPmt2_raw = tCts['pmt1'][21:]*0.
	#tdPmt2 -= muCts[r].pmt2 * hDCts[r].pmt2
	if tCts['coinc'][0] > 0:
		tdPmtC_raw = tCts['coinc'][21:] / tCts['coinc'][0]
	else:
		tdPmtC_raw = tCts['coinc'][21:]*0.

	# Extract the expected background rates
	pmt1_avg = measurement(muCts[r].pmt1/hDCts[r].pmt1,\
						   muCts[r].pmt1/hDCts[r].pmt1*np.sqrt((muStd[r].pmt1/muCts[r].pmt1)**2+(hDStd[r].pmt1/hDCts[r].pmt1)**2))
	pmt2_avg = measurement(muCts[r].pmt2/hDCts[r].pmt2,\
						   muCts[r].pmt2/hDCts[r].pmt2*np.sqrt((muStd[r].pmt2/muCts[r].pmt2)**2+(hDStd[r].pmt2/hDCts[r].pmt2)**2))
	pmtC_avg = measurement(muCts[r].coinc/hDCts[r].coinc,\
						   muCts[r].coinc/hDCts[r].coinc*np.sqrt((muStd[r].coinc/muCts[r].coinc)**2+(hDStd[r].coinc/hDCts[r].coinc)**2))					   
	avg_bin = 10
	bins   = np.zeros(int(len(bins_raw)/avg_bin))
	tdPmt1 = np.zeros(int(len(bins_raw)/avg_bin))
	tdPmt2 = np.zeros(int(len(bins_raw)/avg_bin))
	tdPmtC = np.zeros(int(len(bins_raw)/avg_bin))
	for i in range(int(len(bins_raw)/avg_bin)):
		bins[i] = np.mean(bins_raw[10*i:10*(i+1)-1])
		tdPmt1[i] = np.mean(tdPmt1_raw[10*i:10*(i+1)-1])
		tdPmt2[i] = np.mean(tdPmt2_raw[10*i:10*(i+1)-1])
		tdPmtC[i] = np.mean(tdPmtC_raw[10*i:10*(i+1)-1])
		
	#tdPmtC -= muCts[r].coinc * hDCts[r].coinc
	
	# Now we fit stuff
	if np.mean(tdPmt1) > 0:
		co1, cov1 = curve_fit(doubleExp_bkg, bins, tdPmt1,\
						  #p0=(0.1,100,0.1,1000,muCts[r].pmt1*hDCts[r].pmt1),\
						  p0=(1.,100,1.,1000,np.mean(tdPmt1)),maxfev=5000,\
						  #bounds=([0.0,10.0,0.0,300.0,0.0],[np.inf,1000.0,np.inf,np.inf,np.inf]))				  
						  bounds=([0.0,0.0,0.0,0.0,np.mean(tdPmt1) - 3*np.mean(tdPmt1)*(pmt1_avg.err / pmt1_avg.val)],[20.,1550.0,20.,5000.,np.mean(tdPmt1) + 3*tdPmt1[-1]*(pmt1_avg.err / pmt1_avg.val)]))
	else:
		co1 = np.zeros(5)
		cov1 = np.zeros((5,5))
	chi2_1 = np.sum(np.power(doubleExp_bkg(bins,*co1)-tdPmt1,2) \
							/(doubleExp_bkg(bins,*co1)-co1[4]))
	chi2_1 /= 5
	if np.mean(tdPmt2) > 0:
		co2, cov2 = curve_fit(doubleExp_bkg, bins, tdPmt2,\
						  #p0=(co1[0],co1[1],co1[2],co1[3],muCts[r].pmt2*hDCts[r].pmt2),\
						  p0=(co1[0],co1[1],co1[2],co1[3],np.mean(tdPmt2)),maxfev=5000,\
						  #bounds=([0.0,10.0,0.0,300.0,0.0],[np.inf,1000.0,np.inf,np.inf,np.inf]))				  
						  bounds=([0.0,0.0,0.0,0.0,np.mean(tdPmt2) - 3*np.mean(tdPmt2)*(pmt2_avg.err / pmt2_avg.val)],[20.,1550.0,20.,5000.,np.mean(tdPmt2) + 3*np.mean(tdPmt2)*(pmt2_avg.err / pmt2_avg.val)]))				  
						  #bounds=([0.0,10.0,0.0,300.0,0.0],[np.inf,350.0,np.inf,2000.0,np.inf]))
	else:
		co2 = np.zeros(5)
		cov2 = np.zeros((5,5))
	chi2_2 = np.sum(np.power(doubleExp_bkg(bins,*co2)-tdPmt2,2) \
							/(doubleExp_bkg(bins,*co2) - co2[4]))
	chi2_2 /= 5
	if np.mean(tdPmtC) > 0:
		coC, covC = curve_fit(doubleExp_bkg, bins, tdPmtC,\
						  #p0=(0.1,100,0.1,1000,muCts[r].coinc*hDCts[r].coinc),\
						  p0=(0.1,100,0.1,1000,np.mean(tdPmtC)),maxfev=5000,\
						  #bounds=([0.0,10.0,0.0,300.0,0.0],[np.inf,1000.0,np.inf,np.inf,np.inf]))				  
						  bounds=([0.0,10.0,0.0,10.0,np.mean(tdPmtC) - 5*np.std(tdPmtC)],[20.,1550.0,20.,5000.0,np.mean(tdPmtC) + 5*np.std(tdPmtC)]))
	else:
		coC = np.zeros(5)
		covC = np.zeros((5,5))
	chi2_C = np.sum(np.power(doubleExp_bkg(bins,*coC)-tdPmtC,2) \
							/(doubleExp_bkg(bins,*coC) - coC[4]))
	chi2_C /= 5
	print(co1,co2,coC)
	print(cov1)
	print(chi2_1,chi2_2,chi2_C)
	
	scale1 = np.sum(tdPmt1)/len(tdPmt1)
	scale2 = np.sum(tdPmt2)/len(tdPmt1)
	scaleC = np.sum(tdPmtC)/len(tdPmtC)
	#scale1 = 1.
	#scale2 = 1.
	plt.figure()
	plt.plot(bins,tdPmt1/scale1,'b.',label=("PMT 1"))
	#plt.plot(bins,tdPmt1,'b.',label=("PMT 1: t1 = %f, t2 = %f" % (co1[1],co1[3])))
	plt.plot(bins,tdPmt2/scale2,'r.',label=("PMT 2"))
	plt.plot(bins,tdPmtC/scaleC,'g.',label=("Coinc."))
	plt.plot(bins,doubleExp_bkg(bins,*co1)/scale1,'c',\
			 label=(r'$b_1 = %04f * e^{-t/%04f} + %04f * e^{-t/%04f}$' % (co1[0],co1[1],co1[2],co1[3])))
	plt.plot(bins,doubleExp_bkg(bins,*co2)/scale2,'y',\
			 label=(r'$b_2 = %04f * e^{-t/%04f} + %04f * e^{-t/%04f}$' % (co2[0],co2[1],co2[2],co2[3])))
	plt.plot(bins,doubleExp_bkg(bins,*coC)/scaleC,'grey',\
			 label=(r'$b_C = %04f * e^{-t/%04f} + %04f * e^{-t/%04f}$' % (coC[0],coC[1],coC[2],coC[3])))
	#scaleC = np.sum(tdPmtC)
	#plt.plot(bins,tdPmtC/scaleC,'g.',label=("Coinc."))
	#plt.plot(bins,doubleExp_bkg(bins,*coC)/scaleC,'k.')
	plt.legend()
	plt.title("Normalized Time Dependence During Hold, Runs %d to %d" % (rBreakFull[r],rBreakFull[r+1]))
	plt.xlabel("Time (s)")
	plt.ylabel('Rate (arb)')
	
	plt.figure()
	plt.plot(bins,tdPmt1/scale1-tdPmt2/scale2,'k.',label=("PMT 1 - PMT 2"))
	#plt.plot(bins,tdPmt1,'b.',label=("PMT 1: t1 = %f, t2 = %f" % (co1[1],co1[3])))
	plt.plot(bins,doubleExp_bkg(bins,*co1)/scale1 - doubleExp_bkg(bins,*co2)/scale2,'m',\
			 label=('Difference between fitted values'))
	#scaleC = np.sum(tdPmtC)
	#plt.plot(bins,tdPmtC/scaleC,'g.',label=("Coinc."))
	#plt.plot(bins,doubleExp_bkg(bins,*coC)/scaleC,'k.')
	plt.legend()
	plt.title("Normalized Difference Between PMT 1 and PMT2, Runs %d to %d" % (rBreakFull[r],rBreakFull[r+1]))
	plt.xlabel("Time (s)")
	plt.ylabel('Rate (arb)')
	#plt.show()
	
	# Rescale for putting stuff into data:
	#co1  /= len(tdPmt1)
	#cov1 /= len(tdPmt1)
	#co2  /= len(tdPmt2)
	#cov2 /= len(tdPmt2)
	#coC  /= len(tdPmtC)
	#covC /= len(tdPmtC)
	tDepBkg.write("%d,%d," % (rBreakFull[r],rBreakFull[r+1])) # Runs
	tDepBkg.write("%f,%f,%f,%f,%f,%f,%f,%f," % (co1[0]/10,np.sqrt(cov1[0,0])/10, \
												co1[1],np.sqrt(cov1[1,1]), \
												co1[2]/10,np.sqrt(cov1[2,2])/10, \
												co1[3],np.sqrt(cov1[3,3])))  # PMT 1
	tDepBkg.write("%f,%f,%f,%f,%f,%f,%f,%f," % (co2[0]/10,np.sqrt(cov2[0,0])/10, \
												co2[1],np.sqrt(cov2[1,1]), \
												co2[2]/10,np.sqrt(cov2[2,2])/10, \
												co2[3],np.sqrt(cov2[3,3]))) # PMT 2
	tDepBkg.write("%f,%f,%f,%f,%f,%f,%f,%f\n" % (coC[0]/10,np.sqrt(covC[0,0])/10, \
												coC[1],np.sqrt(covC[1,1]), \
												coC[2]/10,np.sqrt(covC[2,2])/10, \
												coC[3],np.sqrt(covC[3,3]))) # Coinc
tDepBkg.close()
	
# # REDO stuff for plots. Could probably optimize but w/e
# # Initialize lists for each height -- 1, 25, 38, 49 cm each (shortened to 1/2/3/4)
# # Beam off
# day1 = []
# day2 = []
# day3 = []
# day4 = []
# # Beam on
# bOn1 = []
# bOn2 = []
# bOn3 = []
# bOn4 = []

# badRun = 0
# # Extract data from each line of our input file
# for row in cts:
	
	# # Extract data from cts
	# runNo = row['run']
	# # make sure we are in the right region -- hardcoding PMT changes above
	# if not (rBreakTmp[0] <= runNo < rBreakTmp[1]):
		# continue

	# meanTime = (row['te'] + row['ts'] ) / 2
	# t1 = row['temp1']
	# t2 = row['temp2']
	
	# # convert raw counts to rates
	# pmt1 = row['pmt1'] / (row['te'] - row['ts'])
	# pmt2 = row['pmt2'] / (row['te'] - row['ts'])
	# coinc = row['coinc'] / (row['te'] - row['ts'])
			
	# # check that we're using a good background (want a non-zero rate)
	# if not((pmt1 > 10) & (pmt2 > 10) & (coinc > 0)):
		# if runNo != badRun:
			# print "Bad background for run ", runNo
			# badRun = runNo
		# continue
		
	# # sort out beam off background runs -- should be 250 - 2*10 seconds on each step
	# if (225.0 < (row['te'] - row['ts']) < 235.0):
		# if (row['hgt'] == 10):
			# day1.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
		# elif (row['hgt'] == 250):
			# day2.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
		# elif (row['hgt'] == 380):
			# day3.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
		# elif (row['hgt'] == 490):
			# day4.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
		# else:
			# print "Unknown height for run", runNo, "!!"
			
	# # Beam-on background runs should all be less than 200s, but there's a timing glitch now to also fix.
	# # The timings are 100s (immediate) and 40, 20, 150, 50 (or with glitch 150, 170, 210) (s/l pairs)
	# else:
		# if (row['hgt'] == 10):
			# bOn1.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
		# elif (row['hgt'] == 250):
			# bOn2.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
		# elif (row['hgt'] == 380):
			# bOn3.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
		# elif (row['hgt'] == 490):
			# bOn4.append(background(runNo, meanTime, pmt1, pmt2, coinc, t1, t2))
		# else:
			# print "Unknown height for run", runNo, "!!"

# # Normalize each run to a height of 10
# n1 = []
# n2 = []
# n3 = []
# n4 = []

# # keep track of mean values for each height (using reduced backgrounds
# mu1 = bkgRed(0.0,0.0,0.0) # This is raw counts
# mu2 = bkgRed(0.0,0.0,0.0) # These will be scaling factors
# mu3 = bkgRed(0.0,0.0,0.0)
# mu4 = bkgRed(0.0,0.0,0.0)

# # Match run numbers of beam-off backgrounds to get normalized data
# for r1 in day1:
		
	# # Add to mean and increment counter
	# mu1 += bkgRed(r1.pmt1,r1.pmt2,r1.coinc)
	# n1.append(bkgRed(r1.pmt1,r1.pmt2,r1.coinc))
	
	# # Match run numbers and add to means		
	# for r2 in day2:
		# if (r1.run == r2.run):
			# n2.append(bkgNorm(r1, r2))
			# mu2 += bkgNorm(r1,r2)
	# for r3 in day3:
		# if (r1.run == r3.run):
			# n3.append(bkgNorm(r1, r3))
			# mu3 += bkgNorm(r1,r3)
	# for r4 in day4:
		# if (r1.run == r4.run):
			# n4.append(bkgNorm(r1, r4))
			# mu4 += bkgNorm(r1,r4)

# # Divide out means and calculate std. error of the mean:
# mu1 = mu1 / len(n1)
# mu2 = mu2 / len(n2)
# mu3 = mu3 / len(n3)
# mu4 = mu4 / len(n4)

# sd1 = bkgRed(0.0,0.0,0.0)
# sd2 = bkgRed(0.0,0.0,0.0)
# sd3 = bkgRed(0.0,0.0,0.0)
# sd4 = bkgRed(0.0,0.0,0.0)
# # Note that due to our object format we have to do this (can't use np.std)
# for r1 in n1:
	# diff = mu1 - r1
	# sd1 += diff*diff
# sd1 = (sd1**0.5) / len(n1)

# for r2 in n2:
	# diff = mu2 - r2
	# sd2 += diff*diff
# sd2 = (sd2**0.5) / len(n2)

# for r3 in n3:
	# diff = mu3 - r3
	# sd3 += diff*diff
# sd3 = (sd3**0.5) / len(n3)

# for r4 in n4:
	# diff = mu4 - r4
	# sd4 += diff*diff
# sd4 = (sd4**0.5) / len(n4)

# print "\nAverage 1 cm Background Rates (runs between", rBreakTmp[0],"and", rBreakTmp[1],"):"
# print "   PMT 1:", mu1.pmt1, "+/-", sd1.pmt1
# print "   PMT 2:", mu1.pmt2, "+/-", sd1.pmt2
# print "   Coinc:", mu1.coinc, "+/-", sd1.coinc

# print "\nPMT 1 Scaling:"
# print "   25cm:", mu2.pmt1, "+/-", sd2.pmt1
# print "   38cm:", mu3.pmt1, "+/-", sd3.pmt1
# print "   49cm:", mu4.pmt1, "+/-", sd4.pmt1
# print "PMT 2 Scaling:"
# print "   25cm:", mu2.pmt2, "+/-", sd2.pmt2
# print "   38cm:", mu3.pmt2, "+/-", sd3.pmt2
# print "   49cm:", mu4.pmt2, "+/-", sd4.pmt2
# print "Coinc Scaling:"
# print "   25cm:", mu2.coinc, "+/-", sd2.coinc
# print "   38cm:", mu3.coinc, "+/-", sd3.coinc
# print "   49cm:", mu4.coinc, "+/-", sd4.coinc
# #-----------------------------------------------------------------------
# # Time steps have been hard-coded in as tSteps. 
# # For each height we're scaling by the previously found height dependence
# # At some point we'll need to carry through errors.

# # time steps (hard coded in)
# tSteps = [50.0+fillTime, 150.0+fillTime, 250.0+fillTime, 90.0+fillTime, 120.0+fillTime, 205.0+fillTime, 305.0+fillTime, 1565.0+fillTime, 1620.0+fillTime, 1650.0+fillTime, 1735.0+fillTime, 1835.0+fillTime]
# if dag2017 and dag2018:
	# fillTime = 300.0
	# tSteps.extend(tSteps = [50.0+fillTime, 150.0+fillTime, 250.0+fillTime, 90.0+fillTime, 120.0+fillTime, 205.0+fillTime, 305.0+fillTime, 1565.0+fillTime, 1620.0+fillTime, 1650.0+fillTime, 1735.0+fillTime, 1835.0+fillTime])

# # generate zeros for mean/std. err
# tCts = []
# muT = []
# #wtS = []
# sdT = []
# for t in tSteps:
	# tCts.append(0.0)
	# muT.append(bkgRed(0.0,0.0,0.0))
	# #wtS.append(bkgRed(0.0,0.0,0.0))
	# sdT.append(bkgRed(0.0,0.0,0.0))

# # Loop through beam on bkgs at each time -- 1cm is unscaled
# for r in bOn1:
	# for i, t in enumerate(tSteps):
		# if ((t - 2.0) <= r.mt < (t + 2.0)):
			# muT[i] += bkgRed(r.pmt1, r.pmt2, r.coinc)
			# sdT[i] += bkgRed(r.pmt1, r.pmt2, r.coinc)
			# tCts[i] += 1.0
			# break
# # Height 25 cm (scaled by beam-off)
# for r in bOn2:
	# for i, t in enumerate(tSteps):
		# if ((t - 2.0) < r.mt < (t + 2.0)):
			# muT[i] += bkgRed(r.pmt1, r.pmt2, r.coinc) * mu2
			# sdT[i] += bkgRed((r.pmt1 + (sd2.pmt1*sd2.pmt1)/mu1.pmt1),(r.pmt2+ (sd2.pmt2*sd2.pmt2)/mu1.pmt2),(r.coinc+ (sd2.coinc*sd2.coinc)/mu2.coinc))
			# tCts[i] += 1
			# break
# # Height 38 cm (scaled by beam-off)
# for r in bOn3:
	# for i, t in enumerate(tSteps):
		# if ((t - 2.0) < r.mt < (t + 2.0)):
			# muT[i] += bkgRed(r.pmt1, r.pmt2, r.coinc) * mu3
			# sdT[i] += bkgRed((r.pmt1 + (sd3.pmt1*sd3.pmt1)/mu3.pmt1),(r.pmt2 + (sd3.pmt2*sd3.pmt2)/mu3.pmt2),(r.coinc + (sd3.coinc*sd3.coinc)/mu3.coinc))
			# tCts[i] += 1
			# break
# # Height 49 cm (scaled by beam-off)

# for r in bOn4:
	# for i, t in enumerate(tSteps):
		# if ((t - 2.0) < r.mt < (t + 2.0)):
			# muT[i] += bkgRed(r.pmt1, r.pmt2, r.coinc) * mu4
			# sdT[i] += bkgRed((r.pmt1 + (sd4.pmt1*sd4.pmt1)/mu4.pmt1),(r.pmt2 + (sd4.pmt2*sd4.pmt2)/mu4.pmt2),(r.coinc + (sd4.coinc*sd4.coinc)/mu4.coinc))
			# tCts[i] += 1
			# break

# # Make sure we have counts in each
# for i, t in enumerate(tSteps):
	# if tCts[i] == 0:
		# tSteps.pop(i)
		# tCts.pop(i)
		# muT.pop(i)
		# #wtS.pop(i)
		# sdT.pop(i)
		
# # Calculate the mean
# for i, m in enumerate(muT):
	# muT[i] = m / tCts[i]
	# #print muT[i].pmt1, muT[i].pmt2, muT[i].coinc

# # Calculate the standard deviation -- loop through each again!
# for r in bOn1:
	# for i, t in enumerate(tSteps):
		# if ((t - 2.0) <= r.mt < (t + 2.0)):
			# diff = muT[i] - r
			# sdT[i] += diff*diff
# for r in bOn2:
	# for i, t in enumerate(tSteps):
		# if ((t - 2.0) <= r.mt < (t + 2.0)):
			# diff = muT[i] - r
			# sdT[i] += diff*diff
# for r in bOn3:
	# for i, t in enumerate(tSteps):
		# if ((t - 2.0) <= r.mt < (t + 2.0)):
			# diff = muT[i] - r
			# sdT[i] += diff*diff
# for r in bOn4:
	# for i, t in enumerate(tSteps):
		# if ((t - 2.0) <= r.mt < (t + 2.0)):
			# diff = muT[i] - r
			# sdT[i] += diff*diff
# # And normalize
# for i, s in enumerate(sdT):
	# sdT[i] = s**0.5 / tCts[i]
	# #print sdT[i].pmt1, sdT[i].pmt2,sdT[i].coinc

# # bkgTime assumes an exponential decay from time-dependence:
# # a*exp(-x/b) + c (at t->inf bkgTime = c)
# # wild guess: b comes from Al28 decay (half life 134.7 s) or Ar41 decay (half life 6560.4 s)
# g1 = [5.0, 134.7]
# g2 = [5.0, 134.7]
# gc = [1.0, 134.7]

# pmt1Cts = []
# pmt1Err = []
# pmt2Cts = []
# pmt2Err = []
# coinCts = []
# coinErr = []
# for i, m in enumerate(muT):
	# pmt1Cts.append(m.pmt1)
	# pmt1Err.append(sdT[i].pmt1)
	# pmt2Cts.append(m.pmt2)
	# pmt2Err.append(sdT[i].pmt2)
	# coinCts.append(m.coinc)
	# coinErr.append(sdT[i].coinc)

# print len(pmt1Cts), len(pmt1Err)
# # for plotting, shift tSteps by 300 (when beam actually shuts off)
# for i, t in enumerate(tSteps):
	# tSteps[i] = t- fillTime

# #p1opt, p1cov = np.polyfit(tSteps, pmt1Cts,w=coinErr,deg=2)
# #p2opt, p2cov = np.polyfit(tSteps, pmt2Cts,w=coinErr,deg=2)
# #pCopt, p2cov = np.polyfit(tSteps, coinCts,w=coinErr,deg=2)

# # Here's a dumb bug fix. Fitting doesn't like when rate(t=inf) > rate(t)
# # if mu1.pmt1 > max(pmt1Cts):
	# # mu1.pmt1 = min(pmt1Cts)
# # if mu1.pmt2 > max(pmt2Cts):
	# # mu1.pmt2 = min(pmt2Cts)
# # p1opt, p1cov = curve_fit(bkgTime, [tSteps,[mu1.pmt1]*len(tSteps)], pmt1Cts, g1, sigma=pmt1Err, absolute_sigma=True,bounds=[[0.0,0.0],[np.inf,np.inf]])
# # p2opt, p2cov = curve_fit(bkgTime, [tSteps,[mu1.pmt2]*len(tSteps)], pmt2Cts, g2, sigma=pmt2Err, absolute_sigma=True,bounds=[[0.0,0.0],[np.inf,np.inf]])
# # pCopt, pCcov = curve_fit(bkgTime, [tSteps,[mu1.coinc]*len(tSteps)], coinCts, gc, sigma=coinErr, absolute_sigma=True,bounds=[[0.0,0.0],[np.inf,np.inf]])
# p1opt, p1cov = curve_fit(bkgTimeLinear, tSteps, pmt1Cts, g1, sigma=pmt1Err, absolute_sigma=True)
# p2opt, p2cov = curve_fit(bkgTimeLinear, tSteps, pmt2Cts, g2, sigma=pmt2Err, absolute_sigma=True)
# pCopt, pCcov = curve_fit(bkgTimeLinear, tSteps, coinCts, gc, sigma=coinErr, absolute_sigma=True)

# print p1opt, p2opt

# #tConst = (p1opt[1]/np.sqrt(np.diag(p1cov))[1] + p2opt[1]/np.sqrt(np.diag(p2cov))[1]) / (1.0/np.sqrt(np.diag(p1cov))[1] + 1.0/np.sqrt(np.diag(p2cov))[1])
# #tErr = 1.0 / (1.0/np.sqrt(np.diag(p1cov))[1] + 1.0/np.sqrt(np.diag(p2cov))[1])**0.5

# print ""
# print "Singles Time Constant:"#, tConst, "+/-", tErr
# print "   PMT 1 Tau:", p1opt[1], "+/-", np.sqrt(np.diag(p1cov))[1]
# print "   PMT 1 N0:", p1opt[0], "+/-", np.sqrt(np.diag(p1cov))[0]
# print "   PMT 2 Tau:", p2opt[1], "+/-", np.sqrt(np.diag(p2cov))[1]
# print "   PMT 2 N0:", p2opt[0], "+/-", np.sqrt(np.diag(p2cov))[0]
# print "Coincidence Time Constant:", pCopt[1], "+/-", np.sqrt(np.diag(pCcov))[1]
# print "   Coinc. N0:", pCopt[0], "+/-", np.sqrt(np.diag(pCcov))[0]




# tList = np.linspace(0.0,2000.0,100)	
# plt.figure(5)
# # Plot single PMTs
# for i, m in enumerate(muT):
	# plt.errorbar(tSteps[i], m.pmt1,yerr=sdT[i].pmt1, fmt="g.")
	# plt.errorbar(tSteps[i], m.pmt2,yerr=sdT[i].pmt2, fmt="y.")

	# p1min = bkgTimeLinear(tList,p1opt[0]+np.sqrt(np.diag(p1cov))[0],p1opt[1]-np.sqrt(np.diag(p1cov))[1])
	# p1max = bkgTimeLinear(tList,p1opt[0]-np.sqrt(np.diag(p1cov))[0],p1opt[1]+np.sqrt(np.diag(p1cov))[1])
	# plt.plot(tList, bkgTimeLinear(tList,p1opt[0],p1opt[1]),"g")
	# plt.plot(tList, p1min, "g,")
	# plt.plot(tList, p1max, "g,")
	# plt.fill_between(tList,p1min,p1max,facecolor="g",alpha=0.05)
	
	# p2min = bkgTimeLinear(tList,p2opt[0]+np.sqrt(np.diag(p2cov))[0],p2opt[1]-np.sqrt(np.diag(p2cov))[1]) 
	# p2max = bkgTimeLinear(tList,p2opt[0]-np.sqrt(np.diag(p2cov))[0],p2opt[1]+np.sqrt(np.diag(p2cov))[1]) 
	# plt.plot(tList, bkgTimeLinear(tList,p2opt[0],p2opt[1]),"y", label=("PMT2 fit, t = (%.2e +/- %.2e) s" % (p2opt[1],np.sqrt(np.diag(p2cov))[1])))
	# plt.plot(tList, p2min, "y,")
	# plt.plot(tList, p2max, "y,")
	# plt.fill_between(tList,p2min,p2max,facecolor="y",alpha=0.05)
	# plt.title("Time dependence in beam-on background (runs between %d and %d)" % (rBreak[0],rBreak[1]))
	# plt.xlabel("Time since beam off (s)")
	# plt.ylabel("Bkg Rate (Hz)")
# # Formatting of legend
# plt.plot([],[],"g",label=("PMT1 fit, t = (%.2e +/- %.2e) s" % (p1opt[1],np.sqrt(np.diag(p1cov))[1])))
# plt.plot([],[],"y",label=("PMT2 fit, t = (%.2e +/- %.2e) s" % (p2opt[1],np.sqrt(np.diag(p2cov))[1])))
# plt.legend()
	
# # Plot coincidence PMTs
# plt.figure(6)
# for i, m in enumerate(muT):
	# plt.errorbar(tSteps[i], m.coinc,yerr=sdT[i].coinc, fmt="r.")
	# cmin = bkgTimeLinear(tList,pCopt[0]+np.sqrt(np.diag(pCcov))[0],pCopt[1]-np.sqrt(np.diag(pCcov))[1]) 
	# cmax = bkgTimeLinear(tList,pCopt[0]-np.sqrt(np.diag(pCcov))[0],pCopt[1]+np.sqrt(np.diag(pCcov))[1]) 
	# plt.plot(tList, bkgTimeLinear(tList,*pCopt),"r")
	# plt.plot(tList, cmin, "r,")
	# plt.plot(tList, cmax, "r,")
	# plt.fill_between(tList,cmin,cmax,facecolor="r",alpha=0.05)
	# plt.title("Time dependence in beam-on background (runs between %d and %d)" % (rBreak[0],rBreak[1]))
	# plt.xlabel("Time since beam off (s)")
	# plt.ylabel("Bkg Rate (Hz)")
# plt.plot([],[],"r",label=("Coinc. fit, t = (%.2e +/- %.2e) s" % (pCopt[1],np.sqrt(np.diag(pCcov))[1])))
# plt.legend()

# #plt.show()
# #-----------------------------------------------------------------------
# # Plotting stuff
# #-----------------------------------------------------------------------
# pl1 = True # plot beam-off backgrounds
# pl2 = True # plot height dependence ratio by run (PMT1/2 only)
# pl3 = True # plot height dependence factor (averaged)
# pl4 = False # plot (and calc) temperature dependence
# pl5 = False # plot time-dependence curves
# pl6 = False # plot mean times of each run
# pl7 = False  # plot time-dependence linear fit

# if (pl1):
	# print "Generating plot: Position Dependent Beam-off Backgrounds by Run"
	# plt.figure(1)
	# ax1 = plt.subplot(311)
	# for r in day1:
		# plt.plot(r.run, r.pmt1, "b.", markersize=2)
	# for r in day2:
		# plt.plot(r.run, r.pmt1, "g.", markersize=2)
	# for r in day3:
		# plt.plot(r.run, r.pmt1, "y.", markersize=2)
	# for r in day4:
		# plt.plot(r.run, r.pmt1, "r.", markersize=2)

	# # Formatting legend -- can really move this to whatever plot I want
	# plt.plot([],[],"b.",label="10")
	# plt.plot([],[],"g.",label="250")
	# plt.plot([],[],"y.",label="380")
	# plt.plot([],[],"r.",label="490")
	# plt.legend()
		
	# plt.ylabel('PMT1 Rate (Hz)')
	# plt.title('Position Dependent Beam-off Backgrounds by Run') # Title needs to be on top axis
		
	# ax2 = plt.subplot(312)
	# for r in day1:
		# plt.plot(r.run, r.pmt2, "b.", markersize=2)
	# for r in day2:
		# plt.plot(r.run, r.pmt2, "g.", markersize=2)
	# for r in day3:
		# plt.plot(r.run, r.pmt2, "y.", markersize=2)
	# for r in day4:
		# plt.plot(r.run, r.pmt2, "r.", markersize=2)
	
	# plt.ylabel('PMT2 Rate (Hz)')
	
	# ax3 = plt.subplot(313)
	# for r in day1:
		# plt.plot(r.run, r.coinc, "b.", markersize=2)
	# for r in day2:
		# plt.plot(r.run, r.coinc, "g.", markersize=2)
	# for r in day3:
		# plt.plot(r.run, r.coinc, "y.", markersize=2)
	# for r in day4:
		# plt.plot(r.run, r.coinc, "r.", markersize=2)
	
	# plt.ylabel('Coinc. Rate (Hz)')
	# plt.xlabel('Run number') # Xlabel needs to be on bottom axis
		
# if (pl2):
	# print "Generating plot: Height dependence ratio"
	# plt.figure(2)
	# ax1 = plt.subplot(211)
	# for r in n2:
		# plt.plot(r.run, r.pmt1, "g.", label="250", markersize=2)
	# for r in n3:
		# plt.plot(r.run, r.pmt1, "y.", label="380", markersize=2)
	# for r in n4:
		# plt.plot(r.run, r.pmt1, "r.", label="490", markersize=2)	

	# plt.title('Height dependence ratio (PMT1)')
	# plt.xlabel('Run number')
	# plt.ylabel('Rate / Base Rate (arb.)')
	# plt.legend()
	
	# ax2 = plt.subplot(212,sharex=ax1,sharey=ax1)
	# for r in n2:
		# plt.plot(r.run, r.pmt2, "g.", markersize=2)
	# for r in n3:
		# plt.plot(r.run, r.pmt2, "y.", markersize=2)
	# for r in n4:
		# plt.plot(r.run, r.pmt2, "r.", markersize=2)
	# plt.title('Height dependence ratio (PMT2)')
	# plt.xlabel('Run number')
	# plt.ylabel('Rate / Base Rate (arb.)')
	
	# plt.plot([],[],"g.",label="250") # Formatting legend
	# plt.plot([],[],"y.",label="380")
	# plt.plot([],[],"r.",label="490")
	# plt.legend()


# if (pl3):
	# print "Plotting: Height dependence of backgrounds"
	# plt.figure(3)
	# hgts = (10,250,380,490)
	# plt.errorbar(hgts,(1.0,mu2.pmt1,mu3.pmt1,mu4.pmt1),(0.0,sd2.pmt1,sd3.pmt1,sd4.pmt1), label="PMT1", linestyle='', marker='.')
	# plt.errorbar(hgts,(1.0,mu2.pmt2,mu3.pmt2,mu4.pmt2),(0.0,sd2.pmt2,sd3.pmt2,sd4.pmt2), label="PMT2", linestyle='', marker='.')
	# plt.errorbar(hgts,(1.0,mu2.coinc,mu3.coinc,mu4.coinc),(0.0,sd2.coinc,sd3.coinc,sd4.coinc), label="coinc", linestyle='', marker='.')
	
	# plt.title("Height dependence of backgrounds (between runs %d and %d)" % (rBreak[0], rBreak[1]))
	# plt.xlabel("Height (mm)")
	# plt.ylabel('Rate / Base Rate (arb.)')
	# plt.legend(loc=2)

# if (pl4):
	
	# pmt1fit = []
	# pmt2fit = []
	# print "Plotting PMT Temperature Dependent Backgrounds"
	# plt.figure(4)
	
	# # PMT 1 plotting
	# ax1 = plt.subplot(211)
	# for r in day1:
		# if (r.temp1 > 0.0):
			# plt.plot(r.temp1, r.pmt1, "b.", markersize=2)
			# pmt1fit.append([r.temp1, 10.0, r.pmt1])
	# for r in day2:
		# if (r.temp1 > 0.0):
			# plt.plot(r.temp1, r.pmt1, "g.", markersize=2)
			# pmt1fit.append([r.temp1, 250.0, r.pmt1])
	# for r in day3:
		# if (r.temp1 > 0.0):
			# plt.plot(r.temp1, r.pmt1, "y.", markersize=2)
			# pmt1fit.append([r.temp1, 380.0, r.pmt1])
	# for r in day4:
		# if (r.temp1 > 0.0):
			# plt.plot(r.temp1, r.pmt1, "r.", markersize=2)
			# pmt1fit.append([r.temp1, 490.0, r.pmt1])
	
	# plt.title("Position Dependent PMT1 Rates")
	# plt.xlabel('Temp (K)')
	# plt.ylabel('Rate (Hz)')
	
	# plt.plot([],[],"b.",label="10")
	# plt.plot([],[],"g.",label="250")
	# plt.plot([],[],"y.",label="380")
	# plt.plot([],[],"r.",label="490")
	# plt.legend()
	
	# # PMT1 fitting
	# pmt1Arr = np.array(pmt1fit)
	
	# guessP1 = 9.0, -0.4, 0.0
	# p1Opt, p1Cov = curve_fit(bkgRate,[pmt1Arr[:,0],pmt1Arr[:,1]], pmt1Arr[:,2], guessP1)
	# xp1vals = np.linspace(273.0,276.5,100)
	
	# plt.plot(xp1vals, bkgRate([xp1vals, 10.0*np.ones(100)],*p1Opt),"b")
	# plt.plot(xp1vals, bkgRate([xp1vals, 250.0*np.ones(100)],*p1Opt),"g")
	# plt.plot(xp1vals, bkgRate([xp1vals, 380.0*np.ones(100)],*p1Opt),"y")
	# plt.plot(xp1vals, bkgRate([xp1vals, 490.0*np.ones(100)],*p1Opt),"r")
	
	# print "Temperature Dependence (PMT1):", p1Opt,"+\-", np.sqrt(np.diag(p1Cov))
		
	# # PMT 2
	# ax2 = plt.subplot(212)
	# for r in day1:
		# if (r.temp2 > 0.0):
			# plt.plot(r.temp2, r.pmt2, "b.", label="10", markersize=2)
			# pmt2fit.append([r.temp2, 10.0, r.pmt2])
	# for r in day2:
		# if (r.temp2 > 0.0):
			# plt.plot(r.temp2, r.pmt2, "g.", label="250", markersize=2)
			# pmt2fit.append([r.temp2, 250.0, r.pmt2])
	# for r in day3:
		# if (r.temp2 > 0.0):
			# plt.plot(r.temp2, r.pmt2, "y.", label="380", markersize=2)
			# pmt2fit.append([r.temp2, 380.0, r.pmt2])
	# for r in day4:
		# if (r.temp2 > 0.0):
			# plt.plot(r.temp2, r.pmt2, "r.", label="490", markersize=2)
			# pmt2fit.append([r.temp2, 490.0, r.pmt2])
	# plt.title("Position Dependent PMT2 Rates")
	# plt.xlabel('Temp (K)')
	# plt.ylabel('Rate (Hz)')
	
	# plt.plot([],[],"b.",label="10")
	# plt.plot([],[],"g.",label="250")
	# plt.plot([],[],"y.",label="380")
	# plt.plot([],[],"r.",label="490")
	# plt.legend()
	
	# # PMT2 fitting
	# pmt2Arr = np.array(pmt2fit)
	# guessP2 = 9.0, -0.4, 0.0
	# p2Opt, p2Cov = curve_fit(bkgRate,[pmt2Arr[:,0],pmt2Arr[:,1]], pmt2Arr[:,2], guessP2)
	# xp2vals = np.linspace(278.0,282.0,100)
	
	# plt.plot(xp2vals, bkgRate([xp2vals, 10.0*np.ones(100)],*p2Opt),"b")
	# plt.plot(xp2vals, bkgRate([xp2vals, 250.0*np.ones(100)],*p2Opt),"g")
	# plt.plot(xp2vals, bkgRate([xp2vals, 380.0*np.ones(100)],*p2Opt),"y")
	# plt.plot(xp2vals, bkgRate([xp2vals, 490.0*np.ones(100)],*p2Opt),"r")
	
	# print "Temperature Dependence:", p2Opt,"+\-", np.sqrt(np.diag(p2Cov))
	
# if (pl5):
	# # Calculate chi2:
	# c2n1 = 0
	# c2n2 = 0
	# c2nC = 0
	# for i, t in enumerate(tSteps):
		# c2n1 += ((bkgTime([t,mu1.pmt1],p1opt[0],tConst)-pmt1Cts[i])/pmt1Err[i])**2
		# c2n2 += ((bkgTime([t,mu1.pmt2],p2opt[0],tConst)-pmt2Cts[i])/pmt2Err[i])**2
		# c2nC += ((bkgTime([t,mu1.coinc],pCopt[0],pCopt[1])-coinCts[i])/coinErr[i])**2
	# c2n1 = c2n1 / (len(tSteps) - 2)
	# c2n2 = c2n2 / (len(tSteps) - 2)
	# c2nC = c2nC / (len(tSteps) - 2)
	# tList = np.linspace(0.0,2000.0,100)
	
	# # Plot single PMTs
	# plt.figure(5)
	# plt.title("Time dependence in beam-on background (runs between %d and %d)" % (rBreak[0],rBreak[1]))
	# for i, m in enumerate(muT):
		# plt.errorbar(tSteps[i], m.pmt1,yerr=sdT[i].pmt1, fmt="g.")
		# plt.errorbar(tSteps[i], m.pmt2,yerr=sdT[i].pmt2, fmt="y.")
		# plt.plot(tList, bkgTime([tList, [mu1.pmt1]*len(tList)],p1opt[0],tConst),"g")
		# plt.plot(tList, bkgTime([tList, [mu1.pmt2]*len(tList)],p2opt[0],tConst),"y")
		
	# plt.xlabel("Time since beam off (s)")
	# plt.ylabel("Bkg Rate (Hz)")		
	# plt.plot([],[],"g",label=("PMT1 fit ($\chi^2$/ndf = %.03f)" % c2n1))
	# plt.plot([],[],"y",label=("PMT2 fit ($\chi^2$/ndf = %.03f)" % c2n2))
	# plt.legend()
	
	# # Plot coincidence PMTs
	# plt.figure(6)
	# plt.title("Time dependence in beam-on background (runs between %d and %d)" % (rBreak[0],rBreak[1]))
	# for i, m in enumerate(muT):
		# plt.errorbar(tSteps[i], m.coinc,yerr=sdT[i].coinc, fmt="r.", label="Coinc. Data")
		# plt.plot(tList, bkgTime([tList, [mu1.coinc]*len(tList)],*pCopt),"r")
		
	# plt.xlabel("Time since beam off (s)")
	# plt.ylabel("Bkg Rate (Hz)")
	# plt.plot([],[],"r",label=("Coinc. fit ($\chi^2$/ndf = %.03f)" % c2nC))
	# plt.legend()

# if (pl6):
	# print "Plotting: mean time of step"
	# plt.figure(7)
	# ax1 = plt.subplot(311)
	# plt.title("Mean Time of Step in Beam-On Bkg")
	# for r in bOn1:
		# plt.plot(r.mt, r.pmt1, "b.", markersize=2)
	# for r in bOn2:
		# plt.plot(r.mt, r.pmt1, "g.", markersize=2)
	# for r in bOn3:
		# plt.plot(r.mt, r.pmt1, "y.", markersize=2)
	# for r in bOn4:
		# plt.plot(r.mt, r.pmt1, "r.", markersize=2)
	# plt.xlabel("Time (s)")
	# plt.ylabel("Rates (PMT1)")
	# # Formatting legend -- can really move this to whatever plot I want
	# plt.plot([],[],"b.",label="10")
	# plt.plot([],[],"g.",label="250")
	# plt.plot([],[],"y.",label="380")
	# plt.plot([],[],"r.",label="490")
	# plt.legend()
	
	# ax2 = plt.subplot(312)
	# for r in bOn1:
		# plt.plot(r.mt, r.pmt2, "b.", label="10", markersize=2)
	# for r in bOn2:
		# plt.plot(r.mt, r.pmt2, "g.", label="250", markersize=2)
	# for r in bOn3:
		# plt.plot(r.mt, r.pmt2, "y.", label="380", markersize=2)
	# for r in bOn4:
		# plt.plot(r.mt, r.pmt2, "r.", label="490", markersize=2)
	# plt.xlabel("Time (s)")
	# plt.ylabel("Rates (PMT2)")
		
	# ax3 = plt.subplot(313)
	# for r in bOn1:
		# plt.plot(r.mt, r.coinc, "b.", label="10", markersize=2)		
	# for r in bOn2:
		# plt.plot(r.mt, r.coinc, "g.", label="250", markersize=2)
	# for r in bOn3:
		# plt.plot(r.mt, r.coinc, "y.", label="380", markersize=2)
	# for r in bOn4:
		# plt.plot(r.mt, r.coinc, "r.", label="490", markersize=2)
	# plt.xlabel("Time (s)")
	# plt.ylabel("Rates (Coinc)")
		
# plt.show()
