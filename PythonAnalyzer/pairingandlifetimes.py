#!/usr/local/bin/python3

from PythonAnalyzer.classes import measurement
from PythonAnalyzer.functions import *

from math import *
#from statsmodels.stats.weightstats import DescrStatsW
from scipy.optimize import curve_fit, nnls, leastsq, lsq_linear
from scipy.odr import *

import numpy as np

# Global blinding factor hardcoded in here
#totally_sick_blinding_factor = 4.2069
totally_sick_blinding_factor = 0.0

def calc_lifetime_paired(ltVec,runPair=[],vb=True):
	# Calculate the paired, weighted lifetime. 
	# This should track well with an exponential, but is subject to
	# statistical bias (it'll undershoot lifetime for low statistics)
	if vb:
		print("Calculating Lifetime (Paired)...")
			
	ltval = [] # Generate list of lifetimes
	lterr = []
	for lt in ltVec:
		ltval.append(float(lt.val)+totally_sick_blinding_factor) # Blind here
		lterr.append(1.0/float(lt.err**2)) # Value of weighting is 1/err^2
	
	ltAvg = 0.0
	ltWts = 0.0
	for i,lt in enumerate(ltval): # Weighted sum of lifetimes
		ltAvg += lt*lterr[i]
		ltWts += lterr[i]
	
	if ltWts > 0: 
		ltFin = ltAvg/ltWts # Lifetime is weighted sum over sum of weights
		ltErr = np.sqrt(1/ltWts) # Uncertainty is sqrt of sum of weights
	else:
		print("Error! Unable to predict uncertainty on lifetime!")
		ltFin = ltAvg
		ltErr = np.inf
	
	ltval = np.array(ltval)
	unc   = 1./np.sqrt(np.array(lterr))
	
	chi2 = np.sum(np.power((ltval - ltFin)/unc,2))
	chi2NDF = chi2 / (len(ltval)-1)
	ltErr2 = ltErr*np.sqrt(chi2NDF)
	
	if vb:
		print("Paired (weighted) lifetime is:",ltFin,"+/-",ltErr)
		print("Chi2 is:",chi2,"(NDF):",chi2NDF)
		print("Scale Err = ",ltErr2)
	
	return measurement(ltFin,ltErr2)

def calc_lifetime_paired_sections(ltVec,runPair):
	# Calculate the paired, weighted lifetime. 
	# Here I've divided this into majorSections, which are the separation places
	ltval = [] # Generate list of lifetimes
	lterr = []
	for lt in ltVec:
		ltval.append(float(lt.val)+totally_sick_blinding_factor) # Blind here
		lterr.append(1.0/float(lt.err**2)) # Value of weighting is 1/err^2
	
	majorSections = [4200,4711,7326,9600,11669,13209,14517]
	years = [4200,9600,14517]
	ltSections = []
	for s in range(len(majorSections)-1): # Dividing the lifetime into chunks
		ltAvg = 0.0
		ltWts = 0.0
		for i,lt in enumerate(ltval): # Weighted sum of lifetimes
			if majorSections[s] <= runPair[i][0] < majorSections[s+1]:
				ltAvg += lt*lterr[i]
				ltWts += lterr[i]
		
		if ltWts > 0: 
			ltFin = ltAvg/ltWts # Lifetime is weighted sum over sum of weights
			ltErr = np.sqrt(1/ltWts) # Uncertainty is sqrt of sum of weights
		else:
			ltFin = ltAvg
			ltErr = np.inf
		
		print("   For runs",majorSections[s],"to",majorSections[s+1],":")
		print("       Lifetime is:",ltFin,"+/-",ltErr)
		ltSections.append(measurement(ltFin,ltErr))
	
	ltYears = []
	print("   ---------------------------------------------------")
	for s in range(len(years)-1): # Dividing the lifetime into chunks
		ltAvg = 0.0
		ltWts = 0.0
		for i,lt in enumerate(ltval): # Weighted sum of lifetimes
			if years[s] <= runPair[i][0] < years[s+1]:
				ltAvg += lt*lterr[i]
				ltWts += lterr[i]
		
		if ltWts > 0: 
			ltFin = ltAvg/ltWts # Lifetime is weighted sum over sum of weights
			ltErr = np.sqrt(1/ltWts) # Uncertainty is sqrt of sum of weights
		else:
			ltFin = ltAvg
			ltErr = np.inf
		
		print("   For runs",years[s],"to",years[s+1],":")
		print("       Lifetime is:",ltFin,"+/-",ltErr)
		ltYears.append(measurement(ltFin,ltErr))
	return ltSections,ltYears

def calc_lifetime_unw_paired(ltVec, runPair):
	# Calculate the paired, weighted lifetime. 
	# This should be less statistically biased than weighted. 
	# However it might still overshoot the lifetime a bit.
	#
	# Additionally it's not a great use of statistics and will give 
	# inflated error bars
	
	print("Calculating Lifetime (Paired but Unweighted)...")
		
	ltval = [] # Generate list of lifetimes
	for lt in ltVec: # Just look at the value of lifetimes -- ignore errors
		ltval.append(float(lt)+totally_sick_blinding_factor) # float fcn auto-casts .val for measurement class
		
	ltAvg = np.mean(ltval)
	ltStd = np.std(ltval)
	
	print("Paired (unweighted) lifetime is: "+str(ltAvg)+" +/- "+str(ltStd / np.sqrt(len(ltVec))))
	return measurement(ltAvg, ltStd/np.sqrt(len(ltVec)))#, runPair,ltVec

		
def calc_lifetime_exp(rRed,cfg):
	# Single normalization exponential lifetime	
		
	if cfg.vb:
		print("Calculating Lifetime (Exponential Fit)...")
	rawCts = []
	rawErr = []
	timeV = []
	for x in rRed:
		
		if x.normalize_cts().err > 0: # Avoid divide by zero errors
			rawCts.append(x.normalize_cts().val)
			rawErr.append(x.normalize_cts().err)
			if cfg.useMeanArr:
				timeV.append(x.mat) # Assume perfect knowledge of time
			else:
				timeV.append(x.hold)
	
	pFitE, pVarE = curve_fit(explt, timeV, rawCts, p0=(1,880.0), sigma=rawErr,absolute_sigma=False)
	
	print("Exponential lifetime (floating) is: "+str(pFitE[1])+" +/- "+str(np.sqrt(np.diag(pVarE))[1]))
	print(str(pVarE))
	varT = np.sqrt(pFitE[1]*pFitE[1]*pVarE[1][1]*pVarE[1][1] + 2*pFitE[0]*pFitE[1]*pVarE[0][1]) # Attempt to correct for cov.
	print("Uncertainty of t given fixed a is "+str(varT/len(rawCts)))
	
	#return measurement(pFitE[1], np.sqrt(np.diag(pVarE))[1]), measurement(pFitE[0], np.sqrt(np.diag(pVarE))[0])
	
	#pFitE, pVarE = curve_fit(explt_fix, timeV, rawCts, p0=(880.0), sigma=rawErr,absolute_sigma=True)
	#print("Exponential lifetime (fixed) is: "+str(pFitE[0]+totally_sick_blinding_factor)+" +/- "+str(np.sqrt(np.diag(pVarE))[0]))
	return measurement(pFitE[0], np.sqrt(np.diag(pVarE))[0])

def expLT_all_1(x,a):
	# Dead function
	
	return np.exp(-x/b)
	
def calc_lifetime_globalChi2(rawCtsVec,nCorr,holdVec,normT):
	# Here we're doing a chi^2 minimization assuming normalized counts
	# nCorr is the expected value at each point
	
	t_sub = np.linspace(795.0,1065.0,300)
	rawVal = []
	rawErr = []
	tI = []
	expVal = []
	for i,c in enumerate(rawCtsVec):
		rawVal.append(float(c.val))
		rawErr.append(float(c.err))
		tI.append(float(holdVec[i])+20)
		expVal.append(float(nCorr[i]))
			
	rawVal = np.array(rawVal)
	rawErr = np.array(rawErr)
	tI     = np.array(tI)
	expVal = np.array(expVal)
	
	chisq = []
	for i,tau in enumerate(t_sub):
		#print (rawVal[i]-expVal[i])/expVal[i]*np.exp(tI[i]/tau)
		chisq.append(sum(((rawVal/expVal*np.exp(-tI/tau) - np.exp(-tI/tau))**2)))# / rawErr**2)))
	#chisq, cov = curve_fit(explt_fix,tI,(rawVal/expVal),p0=880,sigma=rawErr,absolute_sigma=False)
	print(str(min(chisq)))
	plt.figure(16)
	plt.plot(t_sub,chisq)
	

	
	plt.figure(18)
	for i in range(0,len(rawVal)):
		if 19 < tI[i]-20 < 21:
			fmt = 'r.'
		elif 49 < tI[i]-20 < 51:
			fmt = 'y.'
		elif 99 < tI[i]-20 < 101:
			fmt = 'g.'
		elif 199 < tI[i]-20 < 201:
			fmt = 'b.'
		elif 1549 < tI[i]-20 < 1551:
			fmt = 'c.'
		else:
			fmt = 'k.'
		plt.plot(i, np.exp(-tI[i]/tau), fmt)

	return chisq
			
def calc_lifetime_ODR(nCtsVec,holdVec, useMeanArr = True):
	
	# TEST of Orthogonal Distance Regression (allows errors in X and Y)
	#print len(timeV), len(timeE), len(rawCts), len(rawErr)
	rawCts = []
	rawErr = []
	for x in nCtsVec: #separate out counts
		rawCts.append(float(x.val))
		rawErr.append(float(x.err))
	
	expModel = Model(lnlt)
	if useMeanArr:
		odrData  = RealData(timeV,rawCts,sx=timeE,sy=rawErr)
		testODR  = ODR(odrData,expModel,beta0=[1,880.0])
		output   = testODR.run()
		print("ODR lifetime is: "+str(output.beta[1])+" +/- "+str(output.sd_beta[1]))
	
	return measurement(output.beta[1],output.sd_beta[1])

def calc_lifetime_ODR_meanFit(ctsVec,holdVec, mArrVec = []):
	# TEST of Orthogonal Distance Regression (allows errors in X and Y)
	# meanFit averages all the points before doing a fit.
	
	# If we're not doing mean arrival time, assume mean arrival is just hold
	if len(mArrVec) == 0:
		mArrVec = holdVec
		
	fixTimes = [] # Generate the total number of time bins
	for t in holdVec:
		if round(t.val) not in fixTimes:
			fixTimes.append(round(t.val))
	
	# Count our buffers
	tBuff = np.zeros(len(fixTimes))
	tEBuff = np.zeros(len(fixTimes))
	ctsBuff = np.zeros(len(fixTimes))
	ctsEBuff = np.zeros(len(fixTimes))
	#tBuff = np.array([measurement(0.0,0.0) for y in range(len(fixTimes))])
	#ctsBuff = np.array([measurement(0.0,0.0) for y in range(len(fixTimes))])
	
	num  = np.zeros(len(fixTimes))
	for i, t in enumerate(holdVec):
		ind = int(np.argwhere(np.array(fixTimes) == round(t.val)))
		ctsBuff[ind] += ctsVec[i].val/(ctsVec[i].err*ctsVec[i].err)
		ctsEBuff[ind] += 1/(ctsVec[i].err*ctsVec[i].err)
		if t.err > 0:
			tBuff[ind] += t.val/(t.err*t.err)
			tEBuff[ind] += 1/(t.err*t.err)
		else:
			tBuff[ind] += t.val
			tEBuff[ind] += 1
		#tBuff[ind] += t
		num[ind]  += 1
				
	meanCts  = []
	meanCtsE = []
	meanT  = []
	meanTE = []
	for i, x in enumerate(fixTimes): #separate out counts
		meanCts.append(float(ctsBuff[i]/ctsEBuff[i]))
		meanCtsE.append(1/np.sqrt(ctsEBuff[i]))
		meanT.append(float(tBuff[i]/tEBuff[i]))
		meanTE.append(1/np.sqrt(tEBuff[i]))
		
	expModel = Model(lnlt)
	if len(meanTE) > 0:
		odrData  = RealData(meanT,meanCts,sx=meanTE,sy=meanCtsE)
	else:
		odrData  = RealData(meanT,meanCts,sy=meanCtsE)
	testODR  = ODR(odrData,expModel,beta0=[1,880.0])
	output   = testODR.run()
	print("ODR (mean fit) lifetime is: "+str(output.beta[1])+" +/- "+str(output.sd_beta[1]))
	linetxt = np.linspace(0,5000,5000)
	ltTest = output.beta[0]*np.exp(-linetxt/output.beta[1])
	#plt.figure(9001)
	#if len(meanTE) > 0:
	#	plt.errorbar(meanT,meanCts,xerr=meanTE,yerr=meanCtsE,fmt='b.')
	#else:
	#	plt.errorbar(meanT,meanCts,yerr=meanCtsE,fmt='b.')
	#plt.plot(linetxt,ltTest)
	#plt.yscale('log')
	#plt.show()
	return measurement(output.beta[1],output.sd_beta[1]),meanT, meanCts

def pair_runs(runBreaks, rRed, cfg):
	# Pairing algorithm. I wrote this with a bunch of individual lists
	# so it could be optimized better.
	if cfg.vb:
		print("Pairing runs...")
	
	runsS = [] # Separate run list into short and long
	nCtsS = []
	hldTS = []
	matTS = []
	runsL = []
	nCtsL = []
	matTL = []
	hldTL = []
	#if len(normFac) != len(rN):
	normFac = []
	for r in rRed: # This doesn't mean anything now
		normFac.append(1.0)
		
	nFS = []
	nFL = []		
	for r in rRed:
		
		if (r.hold > 2000.0) and not (cfg.useLong):
			continue
			
		#unld = (r.ctsSum-r.bkgSum)/r.norm_unl(cfg)
		#unld  = r.pct_cts(cfg)/r.norm_unl(cfg)
		#print((r.ctsSum-r.bkgSum)-r.cts,((r.ctsSum-r.ctsSum)-(r.bkgSum-r.bkgSum)))
		unld = r.cts/r.norm
		#unld /= (r.eff[0] + r.eff[1]) 
		if (not np.isfinite(unld.err)) or unld.err == 0.0: # Weird error state catch
			if cfg.vb:
				print("%d has infinite unload error" % r.run)
			continue 
		
		if (r.hold < 500.0):
			runsS.append(r.run)
			nCtsS.append(unld)
			hldTS.append(measurement(r.hold,0.0))
			if cfg.useMeanArr:
				matTS.append(r.mat)
			else:
				matTS.append(measurement(r.hold,0.0))
			#nFS.append(normFac[i])
			nFS.append(r.norm)
			#nFS.append(r.norm_unl(cfg))
		else:
			runsL.append(r.run)
			nCtsL.append(unld)
			hldTL.append(measurement(r.hold,0.0))
			if cfg.useMeanArr:
				matTL.append(r.mat)
			else:
				matTL.append(measurement(r.hold,0.0))
			#nFL.append(r.norm_unl(cfg))
			nFL.append(r.norm)
			
			
	# Pairing
	scMax = 16 # +/- 2 octets
	lInd = 0
	bCount = 0
	corr = 0.9 # No crazy norm drop-offs
	lts = []
	ltsC = []
	runPair = []
	hTPair  = []
	for gap in range(1,scMax): # Pairing algorithm looks to "minimize" spacing between short/long pairs
		bCount = 0
		for sInd, sr in enumerate(runsS): # Loop through short
			paired = False # For breaking this loop
			while runBreaks[bCount+1] < sr: # Find runBreaks region
				bCount+=1
				if bCount == len(runBreaks) - 2:
					break
			for lInd, lr in enumerate(runsL): # and through long
				#print sr,lr, gap, runBreaks[bCount],runBreaks[bCount+1]
				if abs(sr-lr) < gap: # We're in possible range, let's now check other things
					# First, check if this is really the "best" pair
					if lInd != len(runsL)-1:
						if runBreaks[bCount] <= runsL[lInd+1] < runBreaks[bCount+1]: # Make sure we don't cross a break
							if abs(sr-runsL[lInd+1]) < abs(sr-lr): # We have another pair that's closer in range!
								continue
							elif abs(sr - runsL[lInd+1]) == abs(sr-lr): # tiebreaker should be normalization.
								if abs(1.0 - float(nFS[sInd]/nFL[lInd])) > abs(1.0 - float(nFS[sInd]/nFL[lInd+1])):
									continue # I'll let this continue -- you could beat this if you auto-paired this here.
							
					if ((runBreaks[bCount] <= sr < runBreaks[bCount+1] and runBreaks[bCount] <= lr < runBreaks[bCount+1]) # check break regions
						and (corr < float(nFS[sInd]/nFL[lInd]) < 1.0/corr)): # Check that the normalization is roughly the same too
						# We're good! Now just calculate the lifetime
						lifetime  = ltMeas(nCtsS[sInd], nCtsL[lInd], matTS[sInd], matTL[lInd])
						lifetimeC = ltMeas_corr(nCtsS[sInd], nCtsL[lInd], matTS[sInd], matTL[lInd])
						if lifetime.err < lifetime.val: # ignore giant errorbars (for plotting)
							lts.append(lifetime) # output vector
							ltsC.append(lifetimeC)
							runPair.append([sr,lr])
							hTPair.append([hldTS[sInd],hldTL[lInd]])
						
						# And remove these runs, since we've successfully paired them
						runsL.pop(lInd)
						nCtsL.pop(lInd)
						hldTL.pop(lInd)
						matTL.pop(lInd)
						nFL.pop(lInd)
						paired = True #For breaking out of the other loop
						break
				elif lr - sr > scMax: # If lr is scMax more than sr, we can assume lr is sorted and break
					break
			if paired: # If we found a long run for this short run, take this run out for the future!
				runsS.pop(sInd)
				nCtsS.pop(sInd)
				hldTS.pop(sInd)
				matTS.pop(sInd)
				nFS.pop(sInd)
	
	if len(lts) == 0:
		print("Something went wrong, no pairs created!")
	else:
		print("Using "+str(len(lts))+" short-long pairs!")
		print("   First lifetime in list: "+str(lts[0].val)+" +/- "+str(lts[0].err))
		
	return lts, runPair, hTPair,ltsC

def pair_runs_from_list(rRed, pairsList, cfg):
	# Pairing algorithm. I wrote this with a bunch of individual lists
	# so it could be optimized better.
	runsS = [] # Separate run list into short and long
	nCtsS = []
	hldTS = []
	matTS = []
	runsL = []
	nCtsL = []
	matTL = []
	hldTL = []
	#if len(normFac) != len(rN):
	normFac = []
	for r in rRed: # This doesn't mean anything now
		normFac.append(1.0)
		
	pairs = np.zeros(len(pairsList),dtype=[('r1','f8'),('r2','f8')])
	for i in range(len(pairsList)):
		try:
			pairs[i]['r1'] = pairsList[i][0]
			pairs[i]['r2'] = pairsList[i][1]
		except ValueError:
			continue
	nFS = []
	nFL = []
	rUse = []	
	for r in rRed:
		if r.run in pairs['r1'] or r.run in pairs['r2']:
			rUse.append(r)
		else:
			continue
		unld  = r.cts/r.norm
		if (not np.isfinite(unld.err)) or unld.err == 0.0: # Weird error state catch
			if cfg.vb:
				print("%d has infinite unload error" % r.run)
			continue 
		
		if (r.hold < 500.0):
			runsS.append(r.run)
			nCtsS.append(unld)
			hldTS.append(measurement(r.hold,0.0))
			if cfg.useMeanArr:
				matTS.append(r.mat)
			else:
				matTS.append(measurement(r.hold,0.0))
			nFS.append(r.norm)
			
		else:
			runsL.append(r.run)
			nCtsL.append(unld)
			hldTL.append(measurement(r.hold,0.0))
			if cfg.useMeanArr:
				matTL.append(r.mat)
			else:
				matTL.append(measurement(r.hold,0.0))
			nFL.append(r.norm)
			
	runsS=np.array(runsS)
	runsL=np.array(runsL)
	lts     = []
	ltsC    = []
	hTPair  = []
	runPair = []
	
	for pair in pairs:
		runS = pair['r1']
		runL = pair['r2']
		
		sInd = np.where(runsS==runS)[0]
		lInd = np.where(runsL==runL)[0]
		
		if not (np.size(sInd) > 0 and np.size(lInd) > 0):
			continue
		else:
			sInd = int(sInd)
			lInd = int(lInd)
		#if (float(matTS[sInd]) < float(matTL[lInd])): # Run 1 is short, run 2 is long
		# At one point I was concerned about short/long, but I don't think(?) that's an issue
		lifetime   = ltMeas(nCtsS[sInd], nCtsL[lInd], matTS[sInd], matTL[lInd])
		lifetime_C = ltMeas_corr(nCtsS[sInd], nCtsL[lInd], matTS[sInd], matTL[lInd])
		lts.append(lifetime)
		ltsC.append(lifetime_C)
		hTPair.append([hldTS[sInd],hldTL[lInd]])
		runPair.append([runS,runL])
		#else: # Run 2 is short, run 1 is long
		#	lifetime  = ltMeas(nCtsS[lInd], nCtsL[sInd], matTS[lInd], matTL[sInd])
		#	lifetime_C = ltMeas_corr(nCtsS[lInd], nCtsL[sInd], matTS[sInd], matTL[lInd])
		#	lts.append(lifetime)
		#	ltsC.append(lifetime_C)
		#	hTPair.append([hldTS[sInd],hldTL[lInd]])
		#	runPair.append([runS,runL])
		#print (str(lifetime))
	return lts, runPair, hTPair,ltsC

def extract_paired_runs(runPair,redRun,cfg):
	# If we already have a paired list of runs, this produces lifetimes
	# and peak values
	if cfg.vb:
		print("Combining pre-paired runs")
	
	runNum = [] # Get a list of runs for indexing
	for r in redRun: 
		runNum.append(r.run)
		
	# This is for scattering lifetime vs. rate and bkg rate.
	lts = []
	rates = []
	bkgs  = []
	hlds  = []
	norms = []
	for pair in runPair:
		ind1 = np.where(runNum==pair[0])[0]
		ind2 = np.where(runNum==pair[1])[0]
		if not (np.size(ind1) > 0 and np.size(ind2) > 0): # Didn't get this pair!
			continue
		else:
			#int(ind1,ind2)
			# Convert to indices
			if redRun[int(ind1)].hold < redRun[int(ind2)].hold:# Run 1 is short, run 2 is long
				sInd = int(ind1)
				lInd = int(ind2)
			else:
				sInd = int(ind2)
				lInd = int(ind1)
		normS = redRun[sInd].pct_cts(cfg)/redRun[sInd].norm_unl(cfg)
		normL = redRun[lInd].pct_cts(cfg)/redRun[lInd].norm_unl(cfg)
		if cfg.useMeanArr:
			matS = redRun[sInd].mat
			matL = redRun[lInd].mat
		else:
			matS = measurement(redRun[sInd].hold,0.0)
			matL = measurement(redRun[lInd].hold,0.0)
		lifetime = ltMeas_corr(normS,normL,matS,matL)
		lts.append(lifetime)
		rateS = redRun[sInd].pct_raw_cts(cfg) / redRun[sInd].len
		rateL = redRun[lInd].pct_raw_cts(cfg) / redRun[lInd].len
		rates.append([rateS,rateL])
		bkgRS = redRun[sInd].bkgSum / redRun[sInd].len
		bkgRL = redRun[lInd].bkgSum / redRun[lInd].len
		bkgs.append([bkgRS,bkgRL])
		norms.append([redRun[sInd].norm_unl(cfg) / redRun[lInd].norm_unl(cfg)])
		hlds.append([measurement(redRun[sInd].hold,0.0),measurement(redRun[lInd].hold,0.0)])

	return lts, rates, bkgs, hlds, norms


def pair_runs_summed(runBreaks, rN, unld, time,normFac = [],corr=0.9):
	# Redoing the pairing algorithm but with summed ext. counts
	# If we sum up the counts before taking lifetimes, we should still get
	# the right lifetime. There's a little bit of weirdness required here.
	#
	# Namely, we have to be careful with the long runs since they're paired
	# with multiple short lengths (and vice versa)
		
	#print runBreaks
		
	runsS = [] # Separate run list into short and long
	nCtsS = []
	hldTS = []
	runsL = []
	nCtsL = []
	hldTL = []
	
	hldTVecS = [] # This is the list of short run times
	hldTVecL = [] # This is the list of long run times
	if len(normFac) != len(rN):
		normFac = []
		for r in rN:
			normFac.append(1.0)
		
	nFS = []
	nFL = []	
	for i, run in enumerate(rN):
		if unld[i].err == np.inf or unld[i].err == 0.0: # Weird error state catch
			continue 		
		t = time[i].val
		
		if (t < 500.0):
			runsS.append(run)
			nCtsS.append(unld[i])
			hldTS.append(time[i])
			nFS.append(normFac[i])
			if round(t) not in hldTVecS:
				hldTVecS.append(round(t))
		else:
			runsL.append(run)
			nCtsL.append(unld[i])
			hldTL.append(time[i])
			nFL.append(normFac[i])
			if round(t) not in hldTVecL:
				hldTVecL.append(round(t))
	hldTVecS.sort()
	hldTVecL.sort()
	
	nRunsMat = np.zeros((len(hldTVecS),len(hldTVecL))) # Create matrices for runs (number of runs)
	sCtsMat  = np.array([[measurement(0.0,0.0) for y in range(len(hldTVecL))] for x in range(len(hldTVecS))]) # Create matrices for runs (short cts)
	lCtsMat  = np.array([[measurement(0.0,0.0) for y in range(len(hldTVecL))] for x in range(len(hldTVecS))]) # Create matrices for runs (long cts)
	#lCtsMat  = np.empty((len(hldTVecS),len(hldTVecL)),dtype=measurement) # Create matrices for runs (long cts)
	#lCtsMat  = np.empty((len(hldTVecS),len(hldTVecL)),dtype=measurement) # Create matrices for runs (long cts)
		
	#print sCtsMat, hldTVecS, hldTVecL
	# Pairing
	scMax = 16 # Max separation
	lInd  = 0
	bCount = 0
	runPair = []
	#print hldTVecS
	#print hldTVecL
	for gap in range(1,scMax): # Pairing algorithm looks to "minimize" spacing between short/long pairs
		bCount = 0
		for sInd, sr in enumerate(runsS): # Loop through short
			paired = False # For breaking this loop
			while runBreaks[bCount+1] < sr: # Find runBreaks region
				bCount+=1
			for lInd, lr in enumerate(runsL): # and through long
				#print sr,lr, gap, runBreaks[bCount],runBreaks[bCount+1]
				if abs(sr-lr) < gap: # We're in possible range, let's now check other things
					# First, check if this is really the "best" pair
					if lInd != len(runsL)-1:
						if runBreaks[bCount] <= runsL[lInd+1] < runBreaks[bCount+1]: # Make sure we don't cross a break
							if abs(sr-runsL[lInd+1]) < abs(sr-lr): # We have another pair that's closer in range!
								continue
							elif abs(sr - runsL[lInd+1]) == abs(sr-lr): # tiebreaker should be normalization.
								if abs(1.0 - float(nFS[sInd]/nFL[lInd])) > abs(1.0 - float(nFS[sInd]/nFL[lInd+1])):
									continue # I'll let this continue -- you could beat this if you auto-paired this here.
							
					if ((runBreaks[bCount] <= sr < runBreaks[bCount+1] and runBreaks[bCount] <= lr < runBreaks[bCount+1]) # check break regions
						and (corr < float(nFS[sInd]/nFL[lInd]) < 1.0/corr)): # Check that the normalization is roughly the same too
						
						tvIndS = int(np.argwhere(np.array(hldTVecS) == round(hldTS[sInd].val))) # figure out the short index
						tvIndL = int(np.argwhere(np.array(hldTVecL) == round(hldTL[lInd].val))) # figure out the long index
						# We're good! Now just calculate the lifetime	
						nRunsMat[tvIndS][tvIndL] += 1
						sCtsMat[tvIndS][tvIndL]  += measurement(float(nCtsS[sInd].val),float(nCtsS[sInd].err))
						lCtsMat[tvIndS][tvIndL]  += measurement(float(nCtsL[lInd].val),float(nCtsL[lInd].err))
						#	runPair.append([sr,lr])
						#	lts.append(lifetime) # output vector
						runPair.append([sr,lr])
						
						# And remove these runs, since we've successfully paired them
						runsL.pop(lInd)
						nCtsL.pop(lInd)
						hldTL.pop(lInd)
						nFL.pop(lInd)
						paired = True #For breaking out of the other loop
						break
				elif lr - sr > scMax: # If lr is scMax more than sr, we can assume lr is sorted and break
					break
			if paired: # If we found a long run for this short run, take this run out for the future!
				runsS.pop(sInd)
				nCtsS.pop(sInd)
				hldTS.pop(sInd)
				nFS.pop(sInd)
	
	
	# for sInd, sr in enumerate(runsS):
		
		# if nCtsS[sInd].err == np.inf: # If we have an infinite errorbar, continue.
			# #print sr
			# continue
		
		# tvIndS = int(np.argwhere(np.array(hldTVecS) == round(hldTS[sInd].val))) # figure out the short index
		
		# while runBreaks[bCount+1] < sr: # If the short index is in the right "break" region it's OK		
			# bCount += 1
		# if runBreaks[bCount] <= sr < runBreaks[bCount+1]:
			# for lInd, lr in enumerate(runsL): # Loop through long to see if we're in the right "break" region
				# if runBreaks[bCount] <= lr < runBreaks[bCount+1]:
					# if abs(sr - lr) < scMax:
						# #print np.argwhere(np.array(hldTVecL)) == hldTL[lInd]
						# tvIndL = int(np.argwhere(np.array(hldTVecL) == round(hldTL[lInd].val))) # figure out the long index
						# #try:
						# if not nCtsL[lInd].err == np.inf: # If we have an infinite errorbar, continue.							
							# # add counts to the proper position our summing matrices
							# nRunsMat[tvIndS][tvIndL] += 1
							# sCtsMat[tvIndS][tvIndL]  += measurement(float(nCtsS[sInd].val),float(nCtsS[sInd].err))
							# lCtsMat[tvIndS][tvIndL]  += measurement(float(nCtsL[lInd].val),float(nCtsL[lInd].err))
							# runPair.append([sr,lr])
						# #except:
						# #	continue
						# runsL.pop(lInd)
						# nCtsL.pop(lInd)
						# hldTL.pop(lInd)
						# break
	
	lts = []
	wts = []
	for tvIndS, tS in enumerate(hldTVecS):
		for tvIndL, tL in enumerate(hldTVecL):		
				#lifetime = ltMeas(sCtsMat[tvIndS][tvIndL]/nRunsMat[tvIndS][tvIndL],lCtsMat[tvIndS][tvIndL]/nRunsMat[tvIndS][tvIndL],measurement(tS,0.0),measurement(tL,0.0))
			lifetime = ltMeas(sCtsMat[tvIndS][tvIndL],lCtsMat[tvIndS][tvIndL],measurement(tS,0.0),measurement(tL,0.0))
			
			#	lifetime = measurement(0.0,np.inf)
			#print sCtsMat[tvIndS][tvIndL]
			#print lCtsMat[tvIndS][tvIndL]
			
			#print lifetime, tS, tL, nRunsMat[tvIndS][tvIndL]
			lifetime.err = lifetime.err / np.sqrt(nRunsMat[tvIndS][tvIndL]) # This is a sqrt then another sqrt.
			lts.append(lifetime)
			wts.append(nRunsMat[tvIndS][tvIndL])
	
	if len(lts) == 0:
		print("Something went wrong, no pairs created!")
	else:
		print("Summing "+str(len(runPair))+" short-long pairs!")
	
	lt_meas = measurement(0.0,0.0)
	for i,x in enumerate(lts):
		if x.val > 0.0:
			lt_meas += x * measurement(wts[i],0.0)
		else:
			wts[i] = 0.0
	print("Summed Lifetime is: "+str(lt_meas / measurement(np.sum(wts),0.0)))
	#print lts
	
	return lts, runPair

#def pair_runs_from_file(runNum, nCtsVec, holdT,inFName='/home/frank/run_pairs.csv'):
def pair_runs_from_file( rRed, cfg,inFName='/home/frank/run_pairs.csv'):

	#-------------------------------------------------------------------
	# This script loads pairs of runs from a file
	#-------------------------------------------------------------------
	print("Loading list of pairs from "+inFName)
		
	datatype = [('r1','i4'),('r2','i4')]
	pairs = np.loadtxt(inFName, delimiter=",",dtype=datatype)
	
	
	runsS = [] # Separate run list into short and long
	nCtsS = []
	hldTS = []
	matTS = []
	runsL = []
	nCtsL = []
	matTL = []
	hldTL = []
	#if len(normFac) != len(rN):
	normFac = []
	for r in rRed: # This doesn't mean anything now
		normFac.append(1.0)
		
	nFS = []
	nFL = []
	rUse = []	
	for r in rRed:
		if r.run in pairs['r1'] or r.run in pairs['r2']:
			rUse.append(r)
		if (r.hold > 2000.0) and not (cfg.useLong):
			continue
		#unld = (r.ctsSum-r.bkgSum)/r.norm_unl(cfg)	
		unld  = r.pct_cts(cfg)/r.norm_unl(cfg)
		#unld /= (r.eff[0] + r.eff[1]) 
		if (not np.isfinite(unld.err)) or unld.err == 0.0: # Weird error state catch
			if cfg.vb:
				print("%d has infinite unload error" % r.run)
			continue 
		
		if (r.hold < 500.0):
			runsS.append(r.run)
			nCtsS.append(unld)
			hldTS.append(measurement(r.hold,0.0))
			if cfg.useMeanArr:
				matTS.append(r.mat)
			else:
				matTS.append(measurement(r.hold,0.0))
			#nFS.append(normFac[i])
			nFS.append(r.norm_unl(cfg))
			
		else:
			runsL.append(r.run)
			nCtsL.append(unld)
			hldTL.append(measurement(r.hold,0.0))
			if cfg.useMeanArr:
				matTL.append(r.mat)
			else:
				matTL.append(measurement(r.hold,0.0))
			nFL.append(r.norm_unl(cfg))
	runsS=np.array(runsS)
	runsL=np.array(runsL)
	lts     = []
	runPair = []
	for pair in pairs:
		runS = pair['r1']
		runL = pair['r2']
		
		sInd = np.where(runsS==runS)[0]
		lInd = np.where(runsL==runL)[0]
		
		if not (np.size(sInd) > 0 and np.size(lInd) > 0):
			continue
		else:
			sInd = int(sInd)
			lInd = int(lInd)
		if (float(matTS[sInd]) < float(matTL[lInd])): # Run 1 is short, run 2 is long
			lifetime = ltMeas_corr(nCtsS[sInd], nCtsL[lInd], matTS[sInd], matTL[lInd])
			lts.append(lifetime)
			runPair.append([runS,runL])
		else: # Run 2 is short, run 1 is long
			lifetime = ltMeas_corr(nCtsS[lInd], nCtsL[sInd], matTS[lInd], matTL[sInd])
			lts.append(lifetime)
			runPair.append([runS,runL])
		#print (str(lifetime))
	
	return lts, runPair,rUse

# def pair_runs(runBreaks, rN, unld,  time,normFac = [],corr=0.9,normT = []):
	
	# print("Pairing runs...")
	# runsS = [] # Separate run list into short and long
	# nCtsS = []
	# hldTS = []
	# runsL = []
	# nCtsL = []
	# hldTL = []
	# if len(normFac) != len(rN):
		# normFac = []
		# for r in rN:
			# normFac.append(1.0)
		
	# nFS = []
	# nFL = []		
	# for i, run in enumerate(rN):
		# if unld[i].err == np.inf or unld[i].err == 0.0: # Weird error state catch
			# print(str(run)+" has infinite unload error")
			# continue 
		# #print run, unld[i], time[i]
		# #if useMeanArr:
		# t = time[i].val
		# #else:
		# #	t = time[i]
		# #if (t > 2000.0):
		# #	continue
		
		# if (t < 500.0):
			# runsS.append(run)
			# nCtsS.append(unld[i])
			# hldTS.append(time[i])
			# nFS.append(normFac[i])
			
		# else:
			# runsL.append(run)
			# nCtsL.append(unld[i])
			# hldTL.append(time[i])
			# nFL.append(normFac[i])
					
	# # Pairing
	# scMax = 16 # +/- 2 octets
	# lInd = 0
	# bCount = 0
	# lts = []
	# ltsC = []
	# runPair = []
	# hTPair  = []
	# for gap in range(1,scMax): # Pairing algorithm looks to "minimize" spacing between short/long pairs
		# bCount = 0
		# for sInd, sr in enumerate(runsS): # Loop through short
			# paired = False # For breaking this loop
			# while runBreaks[bCount+1] < sr: # Find runBreaks region
				# bCount+=1
			# for lInd, lr in enumerate(runsL): # and through long
				# #print sr,lr, gap, runBreaks[bCount],runBreaks[bCount+1]
				# if abs(sr-lr) < gap: # We're in possible range, let's now check other things
					# # First, check if this is really the "best" pair
					# if lInd != len(runsL)-1:
						# if runBreaks[bCount] <= runsL[lInd+1] < runBreaks[bCount+1]: # Make sure we don't cross a break
							# if abs(sr-runsL[lInd+1]) < abs(sr-lr): # We have another pair that's closer in range!
								# continue
							# elif abs(sr - runsL[lInd+1]) == abs(sr-lr): # tiebreaker should be normalization.
								# if abs(1.0 - float(nFS[sInd]/nFL[lInd])) > abs(1.0 - float(nFS[sInd]/nFL[lInd+1])):
									# continue # I'll let this continue -- you could beat this if you auto-paired this here.
							
					# if ((runBreaks[bCount] <= sr < runBreaks[bCount+1] and runBreaks[bCount] <= lr < runBreaks[bCount+1]) # check break regions
						# and (corr < float(nFS[sInd]/nFL[lInd]) < 1.0/corr)): # Check that the normalization is roughly the same too
						# # We're good! Now just calculate the lifetime
						# lifetime  = ltMeas(nCtsS[sInd], nCtsL[lInd], hldTS[sInd], hldTL[lInd])
						# lifetimeC = ltMeas_corr(nCtsS[sInd], nCtsL[lInd], hldTS[sInd], hldTL[lInd])
						# if lifetime.err < lifetime.val: # ignore giant errorbars (for plotting)
							# lts.append(lifetime) # output vector
							# ltsC.append(lifetimeC)
							# runPair.append([sr,lr])
							# hTPair.append([hldTS[sInd],hldTL[lInd]])
						
						# # And remove these runs, since we've successfully paired them
						# runsL.pop(lInd)
						# nCtsL.pop(lInd)
						# hldTL.pop(lInd)
						# nFL.pop(lInd)
						# paired = True #For breaking out of the other loop
						# break
				# elif lr - sr > scMax: # If lr is scMax more than sr, we can assume lr is sorted and break
					# break
			# if paired: # If we found a long run for this short run, take this run out for the future!
				# runsS.pop(sInd)
				# nCtsS.pop(sInd)
				# hldTS.pop(sInd)
				# nFS.pop(sInd)
	
	# if len(lts) == 0:
		# print("Something went wrong, no pairs created!")
	# else:
		# print("Using "+str(len(lts))+" short-long pairs!")
		# print("   First lifetime in list: "+str(lts[0].val)+" +/- "+str(lts[0].err))
		
	# return lts, runPair, hTPair,ltsC
