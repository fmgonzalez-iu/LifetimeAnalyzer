import numpy as np
from datetime import date
from PythonAnalyzer.classes import measurement

#-----------------------------------------------------------------------
# Writing things out
#-----------------------------------------------------------------------
def write_adam_lifetimes(ltB, yB, fmtB,cfg):
	print("Writing these lifetimes for ATH!")
	today  = date.today()
	todayS = today.strftime("%m-%d-%Y")
	fileName = "FMG_Lifetimes-"+todayS+".txt"
	outfile = open(fileName,"a")
	# Format from Adam
	outfile.write("Analyzer Type Year Peaks Method Threshold tau_raw etau_raw tau_rde etau_rde\n")
	for i, lt in enumerate(ltB):
		# Load formatting strings:
		analyzer='Frank '
		typ = 'paired '
		peaks = ''
		for j in cfg.dips:
			peaks += str(j+1)
		peaks += ' '
		if fmtB[i].sing:
			if fmtB[i].pmt1 and fmtB[i].pmt2:
				meth = 'pmt12 '
			elif fmtB[i].pmt1:
				meth = 'pmt1 '
			else:
				meth = 'pmt2 '
		else:
			meth = 'coinc '		
		if fmtB[i].thresh:
			thr = 'high '
		else:
			thr = 'low '
		if not cfg.useDTCorr:	
			write_2017 = analyzer+typ+'2017 '+peaks+meth+thr+str(yB[i][0].val)+' '+str(yB[i][0].err)+' '+'-1 -1\n'
			write_2018 = analyzer+typ+'2018 '+peaks+meth+thr+str(yB[i][1].val)+' '+str(yB[i][1].err)+' '+'-1 -1\n'
			write_comb = analyzer+typ+'comb '+peaks+meth+thr+str(ltB[i].val)  +' '+str(ltB[i].err)+  ' '+'-1 -1\n'
		else:
			write_2017 = analyzer+typ+'2017 '+peaks+meth+thr+'-1 -1 '+str(yB[i][0].val)+' '+str(yB[i][0].err)+' \n'
			write_2018 = analyzer+typ+'2018 '+peaks+meth+thr+'-1 -1 '+str(yB[i][1].val)+' '+str(yB[i][1].err)+' \n'
			write_comb = analyzer+typ+'comb '+peaks+meth+thr+'-1 -1 '+str(ltB[i].val)  +' '+str(ltB[i].err)+  ' \n'
		outfile.write(write_2017)
		outfile.write(write_2018)
		outfile.write(write_comb)
	outfile.close()

def write_lifetime_telapsed(ltB, yB, cfg):
	print("Writing Lifetimes Based on Config")
	fileName = open("lifetimes_cut.csv","a")
	for i in range(len(ltB)):
		write_2017  = '2017,'+str(yB[i][0].val)+','+str(yB[i][0].err)+','+str(cfg.maxUnl)
		write_2018  = '2018,'+str(yB[i][1].val)+','+str(yB[i][1].err)+','+str(cfg.maxUnl)
		write_comb  = '2019,'+str(ltB[i].val)+','+str(ltB[i].err)+','+str(cfg.maxUnl) # CHEATING
		outfile.write(write_2017)
		outfile.write(write_2018)
		outfile.write(write_2019)
	outfile.close()

def write_background_by_peak(runList):
	print("Writing out: BkgByPk.csv")
	bkgOut = open("BkgByPk.csv","w")
	for r in runList:
		bkgOut.write("%05d,%f,%f,%f\n" % (r.run,r.bkgCts*r.pct[0],r.bkgCts*r.pct[1],r.bkgCts*r.pct[2]))
	bkgOut.close()
		
def write_lifetime_pairs(runPair,ltVec):	
	print("Writing out: LTPairs.txt")
	ltPOut = open("LTPairs.txt", "w")
	for i, lt in enumerate(ltVec):
		ltPOut.write("%05d,%05d\n" % (runPair[i][0], runPair[i][1]))#, lt.val,lt.err))
	ltPOut.close()
		
def write_all_runs(reducedRun):
	print("Writing out: RunList_FMG_all.txt")
	runsOut = open("RunList_FMG_all.txt", "w")
	for run in reducedRun:
		runsOut.write("%05d" % run.run)
		if run != reducedRun[-1]:
			runsOut.write("\n")
	runsOut.close()

def write_run_breaks(rB):
	print("Writing out: RunBreaks.txt")
	runBreaks = open("RunBreaks.txt","w")
	for run in rB:
		runBreaks.write("%05d" % run)
		if run!=rB[-1]:
			runBreaks.write(",")
	runBreaks.close()

def write_long_runs(runNum,holdVec,nCtsVec):
	print("Writing out: LongNorm.txt")
	longOut = open("LongRuns.txt", "w")
	for i, run in enumerate(runNum):
		if 1549.0 < holdVec[i] < 1551.0:
			#longOut.write("%05d,%f,%f\n" % (run, nCtsVec[i].val, nCtsVec[i].err))
			longOut.write("%05d" % (run))
			if run != runNum[-1]:
				longOut.write(",")
	longOut.close()

def write_summed_counts(runs,t,cts,mons):
	print("Writing out: observable counts")
	obsOut = open("summedCts.csv", "w")
	obsOut.write("run,holdT,yield,mon1,mon1Err,mon2,mon2Err\n")
	dataOut = []
	for i, r in enumerate(runs):
		obsOut.write("%05d,%f,%f,%f,%f,%f,%f\n" % (r,t[i],cts[i].val,mons[i][0].val,mons[i][0].err,mons[i][1].val,mons[i][1].err))
		dataOut.append([float(r),float(t[i]),float(cts[i].val),float(mons[i][0].val),float(mons[i][0].err),float(mons[i][1].val),float(mons[i][1].err)])
	dataArr = np.array(dataOut)
	obsOut.close()
	return dataArr

def write_summed_counts_with_bkg(runs,t,cts,mons,bkg):
	print("Writing out: observable counts with bkgs")
	obsOut = open("summedCtsBkg.csv", "w")
	obsOut.write("run,holdT,fore,bkg,bkgPos,mon1,mon1Err,mon2,mon2Err\n")
	dataOut = []
	for i, r in enumerate(runs):
		obsOut.write("%05d,%f,%f,%f,%f,%f,%f,%f,%f\n" % (r,t[i],cts[i].val,bkg[i].val,bkg[i].err,mon[i][0].val,mons[i][0].err,mons[i][1].val,mons[i][1].err))
		dataOut.append([float(r),float(t[i]),float(cts[i].val),float(mons[i][0].val),float(mons[i][0].err),float(mons[i][1].val),float(mons[i][1].err)])
	dataArr = np.array(dataOut)
	obsOut.close()
	return dataArr

def write_extracted_rRed(rRed):
	print("Writing out: observables")
	obsOut = open("fmg_norm.csv","w")
	for r in rRed:
		obsOut.write("%05d,%f,%f,%f,%f\n" % (r.run,r.cts.val,r.cts.err,r.norm.val,r.norm.err))
	obsOut.close()
	return 1
	
def write_extracted_rRed_2(rRed):
	print("Writing out: observables")
	obsOut = open("fmg_norm.csv","w")
	for r in rRed:
		obsOut.write("%05d,%d,%f,%f,%f,%f,%f,%f,%f\n" % \
					(r.run,r.ctsSum.val,r.ctsSum.err,r.bkgSum.val,r.bkgSum.err,\
					 r.norm.val,r.norm.err,r.hold,0.))
	obsOut.close()
	return 1

def write_extracted_rRed_3(rRed):
	print("Writing out: observables")
	obsOut = open("fmg_norm.csv","w")
	for r in rRed:
		obsOut.write("%05d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % \
					(r.run,r.ctsSum.val,r.ctsSum.err,r.bkgSum.val,r.bkgSum.err,\
					 r.norm.val,r.norm.err,r.mon[0].val,r.mon[0].err,r.mon[1].val,r.mon[1].err,r.hold,r.mat))
	obsOut.close()
	return 1

def write_extracted_pairs(rRed,pairs):
	print("Writing out: Observables w/ pairs")
	obsOut = open("fmg_comp.csv","w")
	# Need to calculate which pair each run belongs to.
	pairList = np.zeros((len(pairs),2))
	for i,p in enumerate(pairs):
		#print(i,p)
		pairList[i,0] = p[0]
		pairList[i,1] = p[1]
	
	#pairs = np.asmatrix(pairs)
	for r in rRed:
		pairS = np.where(pairList[:,0]==r.run)
		pairL = np.where(pairList[:,1]==r.run)
		#print(pairS,pairL)
		if np.size(pairS)==1:
			pair = int(pairS[0])
		elif np.size(pairL)==1:
			pair = int(pairL[0])
		else:
			pair = -1
		#print(pair)
		obsOut.write("%05d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d\n" % \
					(r.run,r.hold,r.ctsSum.val,r.ctsSum.err,r.bkgSum.val,r.bkgSum.err,\
					r.norm.val,r.norm.err,r.mon[0].val,r.mon[0].err,r.mon[1].val,r.mon[1].err,pair))
		
	obsOut.close()
	return 1

def write_backgrounds_out(rRed):
	print("Writing out: backgrounds")
	obsOut = open("background_test.csv","w")
	for r in rRed:
		obsOut.write("%05d,%f,%f,%f\n" % \
					(r.run,r.hold,r.bkgSum.val,r.bkgSum.err))
	obsOut.close()
	return 1
	# plt.figure(666)
	# lineBla = []
	# for r in rTest1:
		# lineBla.append(r)
	# plt.plot(rTest1,rTest2,'b.')
	# plt.plot(rTest1,lineBla)
	# plt.xlabel("Run Number")
	# plt.ylabel("Norm. run number")
	# for xl in runB:
		# plt.axvline(x=xl)
	# #plt.show()
def write_mean_arr(rRed):
	print("Writing out: Mean Arrival Times")
	obsOut = open("mat_test.csv","w")
	for r in rRed:
		obsOut.write("%05d,%f,%f,%f\n" % \
					 (r.run,r.hold,r.mat.val,r.mat.err))
	obsOut.close()
	return 1
