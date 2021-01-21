#!/usr/local/bin/python3
import sys
import pdb
import csv
import datetime
from math import *
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
from scipy import stats, special, loadtxt, hstack
from scipy.odr import *
from scipy.optimize import curve_fit, nnls
from datetime import datetime
import matplotlib.pyplot as plt
import corner

from PythonAnalyzer.functions import *
from PythonAnalyzer.classes import *
from PythonAnalyzer.plotting import *


def active_cleaner_values(normRun,acLo,acHi,nCorr,ext='pk1'):
	# Ehhh
	ac1 = []
	ac2 = []
	bac1 = []
	bac2 = []
	time = ('t'+ext)
	rReduced = []
	for run in normRun:
		
		acLoRaw = acLo[acLo['run']==run]
		acHiRaw = acHi[acHi['run']==run]
	
		bkgTime1 = measurement(acLoRaw['thold'],0.0)
		bkgLow1  = measurement(acLoRaw['hold'],np.sqrt(acLoRaw['hold']))
		bkgHigh1 = measurement(acHiRaw['hold'],np.sqrt(acHiRaw['hold']))
		
		bkgTime2 = measurement(acLoRaw['tbkg'],0.0)
		bkgLow2  = measurement(acLoRaw['bkg'],np.sqrt(acLoRaw['bkg']))
		bkgHigh2 = measurement(acHiRaw['bkg'],np.sqrt(acHiRaw['bkg']))
		
		bkgRLow  = bkgLow1  / bkgTime1
		bkgRHigh = bkgHigh1 / bkgTime1
		
		acRLow   = bkgLow2  / bkgTime2
		acRHigh  = bkgHigh2 / bkgTime2
		
		ac1.append(bkgRLow)
		ac2.append(bkgRHigh)
		bac1.append(acRLow)
		bac2.append(acRHigh)
		rReduced.append(run)
						
	return ac1,ac2,bac1,bac2,rReduced

def active_cleaner_raw(normRun,ac,ext='fill'):
	# Raw Counts
	ac1 = []
	#acT = []
	for run in normRun:
		acRaw = ac[ac['run']==run]
		if len(acRaw[ext]) == 0:
			ac1.append(0)
			#acT.append(np.inf)
		else:
			ac1.append(acRaw[ext])
			time = ('t'+ext)
			#acT.append(acRaw[time])
						
	return ac1

def active_cleaner_rate(normRun,ac,ext='bkg'):
	# Raw Counts
	ac1 = []
	#acT = []
	for run in normRun:
		acRaw = ac[ac['run']==run]
		if len(acRaw[ext]) == 0:
			ac1.append(measurement(0,0))
			#acT.append(np.inf)
		else:
			#ac1.append(acRaw[ext])
			cts = measurement(acRaw[ext],np.sqrt(acRaw[ext]))
			time = ('t'+ext)
			t = measurement(acRaw[time],0.0)
			ac1.append(cts/t)
			#acT.append(acRaw[time])
						
	ac1 = np.array(ac1)
	return ac1
	

def active_cleaner_bkgSub(normRun,ac,ext='pk1'):
	# Background/normalized AC counts
	acOut  = []
	acBkg = []
	#acNorm = []
	text = ('t'+ext)
	tHold = [20,50,100,200,1550]
	acBkg1 = []
	acBkg2 = []
	acRawOut = []
	t0 = []
	t1 = []
	t2 = []
	outRun = []
	for t in tHold:
		acBkg1.append([])
		acBkg2.append([])
		acRawOut.append([])
		t0.append([])
		t1.append([])
		t2.append([])
	for i,run in enumerate(normRun):
		# Check 12922
		acRaw = ac[ac['run']==run]
		#if not 7200 < run < 9600:
		#	continue
		outRun.append(run)
		if len(acRaw) == 0:
			print("NO AC DATA FOR", run)
			acOut.append(measurement(0.0,0.0))
			acBkg.append(measurement(0.0,0.0))
			continue
		# Shitty background runs:
		
		
		# Calculate Background Rate
		#if acRaw['thold'] > 1000:
		holdT = measurement(acRaw['thold'],0.0)
		hold  = measurement(acRaw['hold'],np.sqrt(acRaw['hold']))
		#else:
		holdT = measurement(0.0,0.0)
		hold  = measurement(0.0,0.0)
		bkgT = measurement(acRaw['tbkg'],0.0)
		bkg  = measurement(acRaw['bkg'],np.sqrt(acRaw['bkg']))
		# Combine holding and background
		rate = (hold+bkg) / (holdT+bkgT)
		#print(bkg/bkgT,hold/holdT,rate)
		
		# Now pull data from the active cleaner RoI
		unlT = measurement(acRaw[text],0.0)
		unl  = measurement(acRaw[ext],np.sqrt(acRaw[ext]))
		
		unl -= rate*unlT # This is background subtraction
		acOut.append(unl)
		acBkg.append(rate*unlT)
		
		for j,t in enumerate(tHold):
			if t - 1 < acRaw['thold'] < t + 1:			
				acBkg1[j].append(acRaw['hold']/acRaw['thold']*acRaw[text])
				acBkg2[j].append(acRaw['bkg']/acRaw['tbkg']*acRaw[text])
				acRawOut[j].append(acRaw[ext])
				t0[j].append(acRaw['tfill']+acRaw['tclean']+acRaw['thold']/2)
				t1[j].append(acRaw['tfill']+acRaw['tclean']+acRaw['thold']+acRaw['pk1']/2)
				t2[j].append(acRaw['tfill']+acRaw['tclean']+acRaw['thold']+210+acRaw['tbkg']/2)
		#bkgT = measurement(0.0,0.0)
		#bkg  = measurement(0.0,0.0)
		
		
		#if ext=='pk1':
		#	print(unl-rate*unlT)
		
		
		#normUnl = unl/normVal[i]  # Expect the cleaned values to correlate 
		#acNorm.append(normUnl) # with normalized unload
	
	acOut = np.array(acOut)
	acBkg = np.array(acBkg)
	plt.figure()
	print(len(outRun),len(acOut))
	#for i,C in enumerate(acOut):
	#	plt.errorbar(outRun[i],C.val,yerr=C.err)
	
	
	#tmp1 = []
	#tmp2 = []
	#for ac in acOut:
	#	tmp1.append(ac.val)
	#for ac in acBkg:
	#	tmp2.append(ac.val)
	#plt.figure()
	#range2 = np.linspace(np.min(tmp1)-10,np.max(tmp1)+10,50)
	#hist0,bins0 = np.histogram(tmp1,bins=range2,density=False) # Scale histograms
	#plt.hist(range2[:-1],bins=range2,weights=hist0,density=True,ec='r',facecolor='none', \
	#		label=('Out'))
	# for j,t in enumerate(tHold):
		# hold = np.array(acBkg1[j])
		# back = np.array(acBkg2[j])
		# fore = np.array(acRawOut[j])
		# tau = np.array(t2[j]) - np.array(t0[j]) / (np.log(hold) - np.log(back))
		# taum = measurement(np.mean(tau),np.std(tau)/np.sqrt(len(tau)))
		# print(taum)
		# plt.figure()
		# plt.title(t)
		# range1 = np.linspace(0,np.max(fore)+10,50)
		
		# hist1,bins1 = np.histogram(fore,bins=range1,density=False) # Scale histograms
		# plt.hist(range1[:-1],bins=range1,weights=hist1,density=True,ec='r',facecolor='none', \
			# label=('Out'))
		# hist2,bins2 = np.histogram(hold,bins=range1,density=False) # Scale histograms
		# plt.hist(range1[:-1],bins=range1,weights=hist2,density=True,ec='c',facecolor='none', \
				# label=('Hold'))
		# hist3,bins2 = np.histogram(back,bins=range1,density=False) # Scale histograms
		# plt.hist(range1[:-1],bins=range1,weights=hist3,density=True,ec='g',facecolor='none', \
				# label=('Bkg'))
		# plt.legend(loc='upper right')
	#plt.show()
	return acOut,acBkg#,acNorm


# For plotting the active cleaner:
def plot_active_cleaner_bkg(runNum,ac1,ac2,bkg1,bkg2,holdT,plot_gen = 1):

	print(len(runNum),len(ac1),len(ac2))
	plt.figure(plot_gen)
	plot_gen += 1
	for i in range(len(ac1)):
		plt.errorbar(runNum[i],ac1[i].val,yerr=ac1[i].err,fmt='r.')
		plt.errorbar(runNum[i],ac2[i].val,yerr=ac2[i].err,fmt='c.')
	plt.errorbar([],[],fmt='r.',label='2 Photon Threshold')
	plt.errorbar([],[],fmt='c.',label='4 Photon Threshold')
	plt.title("Holding AC Rates")
	plt.xlabel("Run Number")
	plt.ylabel("Rate (Hz)")
	plt.legend()
	
	plt.figure(plot_gen)
	plot_gen += 1
	for i in range(len(bkg1)):
		plt.errorbar(runNum[i],bkg1[i].val,yerr=bkg1[i].err,fmt='r.')
		plt.errorbar(runNum[i],bkg2[i].val,yerr=bkg2[i].err,fmt='c.')
	plt.errorbar([],[],fmt='r.',label='2 Photon Threshold')
	plt.errorbar([],[],fmt='c.',label='4 Photon Threshold')
	plt.title("End of Run AC Rates")
	plt.xlabel("Run Number")
	plt.ylabel("Rate (Hz)")
	plt.legend()
	
	plt.figure(plot_gen)
	plot_gen += 1
	for i in range(len(bkg1)):
		diff1 = ac1[i] - bkg1[i]
		diff2 = ac2[i] - bkg2[i]
		plt.errorbar(runNum[i],diff1.val,yerr=diff1.err,fmt='r.')
		plt.errorbar(runNum[i],diff2.val,yerr=diff2.err,fmt='c.')
	plt.errorbar([],[],fmt='r.',label='2 Photon Threshold')
	plt.errorbar([],[],fmt='c.',label='4 Photon Threshold')
	plt.title("Difference between End of Run and Holding AC Rates")
	plt.xlabel("Run Number")
	plt.ylabel("Rate (Hz)")
	plt.legend()
	
	plt.figure(plot_gen)
	plot_gen += 1
	a1 = []
	for ac in ac1:
		a1.append(ac.val)
	a2 = []
	for ac in ac2:
		a2.append(ac.val)
	b1 = []
	for ac in bkg1:
		b1.append(ac.val)
	b2 = []
	for ac in bkg2:
		b2.append(ac.val)
	
	mA1 = np.average(a1)
	mA2 = np.average(a2)
	mB1 = np.average(b1)
	mB2 = np.average(b2)
	sA1  = np.std(a1)
	sA2  = np.std(a2)
	sB1  = np.std(b1)
	sB2  = np.std(b2)
	
	print(mA1,sA1/np.sqrt(float(len(a1))),mA2,sA2/np.sqrt(float(len(a2))))
	print(mB1,sB1/np.sqrt(float(len(b2))),mB2,sB2/np.sqrt(float(len(b2))))
	
	print("Diff =", mA1 - mB1,"LoT")
	print("Diff =", mA2 - mB2,"HiT")
	# plt.figure(plot_gen)
	# plot_gen += 1
	# binspace = np.linspace(0,20,20)
	# plt.hist(a1, bins=binspace, ec='r',facecolor="none",label=("2 photon (%0.04f +/- %0.04f)" % (mA1, sA1)))	
	# plt.hist(a2, bins=binspace, ec='c',facecolor="none",label=("4 photon (%0.04f +/- %0.04f)" % (mA2, sA2)))	
	# plt.title("Histogram of Holding Time Rates")
	# plt.xlabel("Percent")
	# plt.ylabel("Number (arb.)")
	# plt.legend()
	
	# plt.figure(plot_gen)
	# plot_gen += 1
	# plt.hist(b1, bins=binspace, ec='r',facecolor="none",label=("2 photon (%0.04f +/- %0.04f)" % (mB1, sB1)))	
	# plt.hist(b2, bins=binspace, ec='c',facecolor="none",label=("4 photon (%0.04f +/- %0.04f)" % (mB2, sB2)))	
	# plt.title("Histogram of End of Run Rates")
	# plt.xlabel("Percent")
	# plt.ylabel("Number (arb.)")
	# plt.legend()
	
	return plot_gen

def plot_active_cleaner_norm(runNum,ac1,ac2,bkg1,bkg2,holdT,plot_gen = 1):

	print(len(runNum),len(ac1),len(ac2))
	plt.figure(plot_gen)
	plot_gen += 1
	for i in range(len(ac1)):
		plt.errorbar(runNum[i],ac1[i].val,yerr=ac1[i].err,fmt='r.')
		plt.errorbar(runNum[i],ac2[i].val,yerr=ac2[i].err,fmt='c.')
	plt.errorbar([],[],fmt='r.',label='2 Photon Threshold')
	plt.errorbar([],[],fmt='c.',label='4 Photon Threshold')
	plt.title("Filling Time Counts")
	plt.xlabel("Run Number")
	plt.ylabel("Counts")
	plt.legend(loc='upper right')
	
	acpk1_2017 = measurement(0.0,0.0)
	n_2017 = 0.0
	acpk2_2018 = measurement(0.0,0.0)
	n_2018 = 0.0
	pk1_s7 = []
	pk1_l7 = []
	pk1_s8 = []
	pk1_l8 = []
	for i in range(len(ac1)):
		if 12878 <= runNum[i] < 12923:
			continue
		
		if runNum[i] < 9600:
			acpk1_2017 += ac2[i]
			if 1549 < float(holdT[i]) < 1551:
				pk1_l7.append(float(ac2[i]))
			elif 19 < float(holdT[i]) < 21:
				pk1_s7.append(float(ac2[i]))
			n_2017 += 1
		else:
			acpk2_2018 += ac2[i]
			if 1549 < float(holdT[i]) < 1551:
				pk1_l8.append(float(ac2[i]))
			elif 19 < float(holdT[i]) < 21:
				pk1_s8.append(float(ac2[i]))
			n_2018 += 1
	#acpk1_2017 / n_2017
	#acpk		
	ms7 = np.mean(pk1_s7)
	ml7 = np.mean(pk1_l7)
	ms8 = np.mean(pk1_s8)
	ml8 = np.mean(pk1_l8)
	
	ss7 = np.std(pk1_s7)/np.sqrt(len(pk1_s7))
	sl7 = np.std(pk1_l7)/np.sqrt(len(pk1_l7))
	ss8 = np.std(pk1_s8)/np.sqrt(len(pk1_s7))
	sl8 = np.std(pk1_l8)/np.sqrt(len(pk1_l8))
	
	print("2017:",acpk1_2017,n_2017)
	print("2018:",acpk2_2018,n_2018)
	
	binspace = np.linspace(0,20,20)
	plt.figure(plot_gen)
	plot_gen += 1
	plt.hist(pk1_s7, bins=40, ec='r',facecolor="none",label=("20s (%0.04f +/- %0.04f)" % (ms7, ss7)))	
	plt.hist(pk1_l7, bins=40, ec='c',facecolor="none",label=("1550s (%0.04f +/- %0.04f)" % (ml7, sl7)))	
	plt.title("Peak 1 Unload, 2017")
	plt.xlabel("Counts")
	plt.ylabel("Prob. (arb.)")
	plt.legend(loc='upper right')
	
	plt.figure(plot_gen)
	plot_gen += 1
	plt.hist(pk1_s8, bins=40, ec='r',facecolor="none",label=("20s (%0.04f +/- %0.04f)" % (ms8, ss8)))	
	plt.hist(pk1_l8, bins=40, ec='c',facecolor="none",label=("1550s (%0.04f +/- %0.04f)" % (ml8, sl8)))	
	plt.title("Peak 1 Unload, 2018")
	plt.xlabel("Counts")
	plt.ylabel("Prob. (arb.)")
	plt.legend(loc='upper right')
	
	
	plt.figure(plot_gen)
	plot_gen += 1
	for i in range(len(bkg1)):
		plt.errorbar(runNum[i],bkg1[i].val,yerr=bkg1[i].err,fmt='r.')
		plt.errorbar(runNum[i],bkg2[i].val,yerr=bkg2[i].err,fmt='c.')
	plt.errorbar([],[],fmt='r.',label='2 Photon Threshold')
	plt.errorbar([],[],fmt='c.',label='4 Photon Threshold')
	plt.title("Normalized Filling Time Counts")
	plt.xlabel("Run Number")
	plt.ylabel("Counts (arb.)")
	plt.legend(loc='upper right')
		
	return plot_gen

def ac_list_format(val):
	
	if val == 0:
		f = 'r.'
		l = '2 Photon Threshold'
	elif val == 1:
		f = 'y.'
		l = '3 Photon Threshold'
	elif val == 2:
		f = 'c.' 
		l = '4 Photon Threshold'
	elif val == 3:
		f = 'g.'
		l = '5 Photon Threshold'
	else:
		f = 'k.'
		l = 'other'
	
	return f,l
	
def plot_and_get_shit(runNum,acList,title,plot_gen = 1):
		
	plt.figure(plot_gen)
	plot_gen += 1
	mu7  = []
	std7 = []
	mu8  = []
	std8 = []
	for k,ac1 in enumerate(acList):
		# Assuming a list of lists
		f,l = ac_list_format(k)
		v7 = []
		v8 = []
		for i in range(len(ac1)):
			plt.errorbar(runNum[i],ac1[i].val,yerr=ac1[i].err,fmt=f)
			if runNum[i] < 9600:
				v7.append(ac1[i].val)
			else:
				v8.append(ac1[i].val)
		plt.errorbar([],[],fmt=f,label=l)
		mu7.append(np.mean(v7))
		std7.append(np.std(v7)/np.sqrt(len(v7)))
		mu8.append(np.mean(v8))
		std8.append(np.std(v8)/np.sqrt(len(v8)))
	plt.title(title+" Rates")
	plt.xlabel("Run Number")
	plt.ylabel("Rate (Hz)")
	plt.legend(loc='upper right')
	
	for i in range(len(mu7)):
		print(title+" Averages:")
		f,l = ac_list_format(i)
		print("   2017", l,":",mu7[i],"+/-",std7[i])
		print("   2018", l,":",mu8[i],"+/-",std8[i])
	
	return plot_gen

def plot_and_get_background(runNum,acList,title,plot_gen = 1):
		
	
	mu7  = []
	std7 = []
	mu8  = []
	std8 = []
	for k,ac1 in enumerate(acList):
		# Assuming a list of lists
		#if not k == 2:
		#	continue
		plt.figure(plot_gen)
		plot_gen += 1
		f,l = ac_list_format(k)
		v7 = []
		v8 = []
		for i in range(len(ac1)):
			plt.errorbar(runNum[i],ac1[i].val,yerr=ac1[i].err,fmt=f)
			if runNum[i] < 9600:
				v7.append(ac1[i].val)
			else:
				v8.append(ac1[i].val)
		plt.errorbar([],[],fmt=f,label=l)
		mu7.append(np.mean(v7))
		std7.append(np.std(v7)/np.sqrt(float(len(v7))))
		mu8.append(np.mean(v8))
		std8.append(np.std(v8)/np.sqrt(float(len(v8))))
		
		plt.title(title+" Counts")
		plt.xlabel("Run Number")
		plt.ylabel("Counts (arb.)")
		plt.legend(loc='upper right')
	
	for i in range(len(mu7)):
		print(title+" Averages:")
		f,l = ac_list_format(i)
		print("   2017", l,":",mu7[i],"+/-",std7[i])
		print("   2018", l,":",mu8[i],"+/-",std8[i])
	
	return plot_gen

def plot_active_cleaner_fuck_it(runNum,holdT,acFill,acFillN,acCln,acClnN,acHold,acPk1,acPk1N,acBkg):
	# This recreates the note on 07/06/2020 all at once with arbitrary AC
	# I don't fucking care about how long it'll take
	# Also it's coded terribly
	plot_gen = 1
	#plot_gen = plot_and_get_shit(runNum,acHold,'Holding Time AC',plot_gen)
	#plot_gen = plot_and_get_shit(runNum,acBkg,'End of Run AC',plot_gen)
	
	#acDiff = []
	#for k, ac1 in enumerate(acBkg):
	#	acDiff.append(acBkg[k] - acHold[k])
	#plot_gen = plot_and_get_shit(runNum,acDiff,'Difference between End of Run and Holding Time',plot_gen)
	
	#plot_gen = plot_and_get_background(runNum,acFill,'Filling Time',plot_gen)
	#plot_gen = plot_and_get_background(runNum,acFillN,'Normalized Filling Time',plot_gen)
		
	#plot_gen = plot_and_get_background(runNum,acCln,'Cleaning Time',plot_gen)
	#plot_gen = plot_and_get_background(runNum,acClnN,'Normalized Cleaning Time',plot_gen)
	
	plot_gen = plot_and_get_background(runNum,acPk1,'Peak 1',plot_gen)
	binspace = np.linspace(0,20,20)
	plt.figure(plot_gen)
	plot_gen += 1
	for k,ac in enumerate(acPk1):
		ctsSum7 = [] # Sum Counts
		ctsSum8 = []
		for i,r in enumerate(ac):
			if runNum[i] < 9600:
				ctsSum7.append(ac[i].val)
			else:
				ctsSum8.append(ac[i].val)
		print(np.sum(ctsSum7))
		print(np.sum(ctsSum8))
		f,l = ac_list_format(k)
		#plt.hist(ctsSum8, bins=40, ec=f[0],facecolor="none",label=("2018 (%0.04f +/- %0.04f)" % (np.mean(ctsSum8), np.std(ctsSum8)/np.sqrt(len(ctsSum8)))))	
		#plt.hist(ctsSum7, bins=40, ec=f[0],facecolor="none",label=("2017 (%0.04f +/- %0.04f)" % (np.mean(ctsSum7), np.std(ctsSum7)/np.sqrt(len(ctsSum7)))))	
	#plt.title("Peak 1 Unload, 2017")
	#plt.xlabel("Counts")
	#plt.ylabel("Prob. (arb.)")
	#plt.legend(loc='upper right')
	plot_gen = plot_and_get_background(runNum,acPk1N,'Normalized Peak 1',plot_gen)
		
	
	
		
		
	return plot_gen
	# #-------------------------------------------------------------------
	# # Diff
	# plt.figure(plot_gen)
	# plot_gen += 1
	# mu7  = []
	# std7 = []
	# mu8  = []
	# std8 = []
	# for k,ac1 in enumerate(acBkg):
		# # Assuming a list of lists
		# f,l = ac_list_format(k)
		# v7 = []
		# v8 = []
		# for i in range(len(ac1)):
			# diff = ac1[i] - acHold[k][i]
			# plt.errorbar(runNum[i],diff.val,yerr=diff.err,fmt=f)
			# if runNum[i] < 9600:
				# v7.append(ac1[i].val)
			# else:
				# v8.append(ac1[i].val)
		# plt.errorbar([],[],fmt=f,label=l)
		# mu7.append(np.mean(v7))
		# std7.append(np.std(v7)/np.sqrt(len(v7)))
		# mu8.append(np.mean(v8))
		# std8.append(np.std(v8)/np.sqrt(len(v8)))
	# plt.title("Difference Between End of Run and Hold AC Rates")
	# plt.xlabel("Run Number")
	# plt.ylabel("Rate (Hz)")
	# plt.legend(loc='upper right')
	
	# for i in range(len(acHold)):
		# print("Difference Between End of Run and Hold AC Averages:")
		# f,l = ac_list_format(i)
		# print("   2017", l,":",mu7[i],"+/-",std7[i])
		# print("   2018", l,":",mu8[i],"+/-",std8[i])
	# #-------------------------------------------------------------------
	# # Filling Time
	# plt.figure(plot_gen)
	# plot_gen += 1
	# mu7  = []
	# std7 = []
	# mu8  = []
	# std8 = []
	# for k,ac1 in enumerate(acBkg):
		# # Assuming a list of lists
		# f,l = ac_list_format(k)
		# v7 = []
		# v8 = []
		# for i in range(len(ac1)):
			# plt.errorbar(runNum[i],ac1[i].val,yerr=ac1[i].err,fmt=f)
			# if runNum[i] < 9600:
				# v7.append(ac1[i].val)
			# else:
				# v8.append(ac1[i].val)
		# plt.errorbar([],[],fmt=f,label=l)
		# mu7.append(np.mean(v7))
		# std7.append(np.std(v7)/np.sqrt(len(v7)))
		# mu8.append(np.mean(v8))
		# std8.append(np.std(v8)/np.sqrt(len(v8)))
	# plt.title("End of Run AC Rates")
	# plt.xlabel("Run Number")
	# plt.ylabel("Rate (Hz)")
	# plt.legend(loc='upper right')
	
	# for i in range(len(acHold)):
		# print("End of Run Averages:")
		# f,l = ac_list_format(i)
		# print("   2017", l,":",mu7[i],"+/-",std7[i])
		# print("   2018", l,":",mu8[i],"+/-",std8[i])
	
	# # Holding
	# plt.figure(plot_gen)
	# plot_gen += 1
	# for i in range(len(ac1)):
		# plt.errorbar(runNum[i],ac1[i].val,yerr=ac1[i].err,fmt='r.')
		# plt.errorbar(runNum[i],ac2[i].val,yerr=ac2[i].err,fmt='c.')
	# plt.errorbar([],[],fmt='r.',label='2 Photon Threshold')
	# plt.errorbar([],[],fmt='c.',label='4 Photon Threshold')
	# plt.title("Filling Time Counts")
	# plt.xlabel("Run Number")
	# plt.ylabel("Counts")
	# plt.legend(loc='upper right')
	
	
	# return plot_gen
	


