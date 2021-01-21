import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# This script adds and fits deconstructed histograms.
#global clocktics
#clocktics = 0.8

#def multiexp(x,k1,t1,k2,t2,k3,t3):
	# Multiexponential. Units in ns
#	global t_t
	#t_t = 40000 # We've presently gone to 40 us for truncation
	
#	fs1 = k1 * np.exp(-x/t1) / (t1 * (1 - np.exp(-t_t / t1)))
#	fs2 = k2 * np.exp(-x/t2) / (t2 * (1 - np.exp(-t_t / t2)))
#	fs3 = k3 * np.exp(-x/t3) / (t3 * (1 - np.exp(-t_t / t3)))
#	return fs1 + fs2 + fs3

configFolder = sys.argv[1]
nSamples = 24

tName = '/holdingBkg'

breaksIn  = [4231,7332,9600,13219]
breaksOut = [4200,7327,9600,13309]

csv     = ".csv"

dtypeTime = [('bin','i4'),('pmt1','f8'),('pmt2','f8'),('coinc','f8')]

# First figure out the size of our histogram
for i in range(len(breaksIn)):
	testName = configFolder+str(breaksIn[i])+tName+csv+str(0)
	timingHistTmp = np.loadtxt(testName,delimiter=",",dtype=dtypeTime) # Timing Histogram
	timingHist = timingHistTmp['bin'] # Load numpy arrays
	
	output = np.zeros(len(timingHist),dtype=dtypeTime)
		
	output['bin'] = timingHist
	for s in range(0,nSamples):
		name  = configFolder + str(breaksIn[i]) + tName + csv  + str(s)
		try:
			timing  = np.loadtxt(name,  delimiter=",",dtype=dtypeTime) # Load the data
			output['pmt1']  += timing['pmt1']
			output['pmt2']  += timing['pmt2']
			output['coinc'] += timing['coinc']
		except:
			print("Skipping",name)	
		
	outfileName = configFolder+tName+'tot'+csv+str(breaksOut[i])
	outfile = open(outfileName,"w")
	
	for o in output:
		outfile.write("%d,%f,%f,%f\n" %(o['bin'],o['pmt1'],o['pmt2'],o['coinc']))
		
	outfile.close()
#pmts = range(0,4) # PMT possibilities
#timingCts = [] # initialize counts buffer 
#timingBkg = []
#for p in pmts:
	
	#timingCts.append(np.zeros(len(timingHistTmp_0)))
	#timingBkg.append(np.zeros(len(timingHistTmp_0)))

## Now we want to loop through the data
#for p in pmts:
	#for s in range(0,nSamples):
		#name  = configFolder + tName + str(p+1) + csv  + str(s)
		#nameB = configFolder + tName + str(p+1) + csvB + str(s)
		#timing  = np.loadtxt(name,  delimiter=",",dtype=dtypeTime) # Load the data
		#timingB = np.loadtxt(nameB, delimiter=",",dtype=dtypeTime) # Load the background
				
		#timingCts[p] += timing['cts']
		#timingBkg[p] += timingB['cts']

## Writeout
##for p in pmts:
	##name  = configFolder + tName + str(p+1) + csv  + str("_tot")
	##nameB = configFolder + tName + str(p+1) + csvB + str("_tot")
	##fileTot  = open(name,"w")
	##fileBTot = open(nameB,"w")
	##for i in range(len(timingHist)):
		##fileTot.write("%d,%d\n" % (timingCts[p][i],timingHist[i]))
		##fileBTot.write("%d,%d\n" % (timingBkg[p][i],timingHist[i]))
	##fileTot.close()
	##fileBTot.close()
#print("Should have saved the PMT Hit Structures!")

## We should have the timing bins now.
## Let's do some basic fitting and plotting
#guess = (0.3,100,0.3,1000,0.3,10000)
#bound = ([0,0,0,0,0,0],[1.0,np.inf,1.0,np.inf,1.0,np.inf])
##bound = ([0,0,0,0,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
#fitparams = []
#fitcurves = []
#bkgparams = []
#bkgcurves = []
#chi2Plts  = []
#realtime = timingHist * clocktics
#sum1 = 0
#sum2 = 0
#for p in pmts: # First try, do parameter fits by PMT
	#timingCts_tmp = timingCts[p]
	#scale = np.sum(timingCts_tmp)
	#sum1 += scale
	#timingCts_tmp /= scale # Rescale histogram
	#val,cov = curve_fit(multiexp,realtime[1000:-1],timingCts_tmp[1000:-1],p0=guess,bounds=bound)
	##print(p+1,val,np.sqrt(np.diag(cov)))
	#fitparams.append(val)
	#fitcurves.append(cov)
	
	#timingBkg_tmp = timingBkg[p]
	#scaleB = np.sum(timingBkg_tmp)
	#sum2 += scaleB
	#timingBkg_tmp /= scaleB # Rescale histogram
	#val2,cov2 = curve_fit(multiexp,realtime[1000:-1],timingBkg_tmp[1000:-1],p0=guess,bounds=bound)
	##print(p+1,val,np.sqrt(np.diag(cov)))
	#bkgparams.append(val2)
	#bkgcurves.append(cov2)
		
	#chi2 = np.sum((multiexp(realtime,*val) - timingCts_tmp)**2/multiexp(realtime,*val))
	#ndf  = len(realtime) - len(val)
	#print(scale,chi2*scale, chi2*scale / float(ndf)) # Have to scale the chi2 to be counts again

#s2b = sum2/sum1
#tTot = np.zeros(len(timingHist))
#bkgTot = np.zeros(len(timingHist))
#for p in pmts:
	#tTot += timingCts[p]
	#bkgTot += timingBkg[p]

#tS = np.sum(tTot)
#tSB = np.sum(bkgTot)
#print(tSB/tS)
#tTot_tmp = tTot / tS
#v1,c1 = curve_fit(multiexp,realtime[250:-1],tTot_tmp[250:-1],p0=guess,bounds=bound)
#print(v1,np.sqrt(np.diag(c1)))
#for p in pmts:
	#print(bkgparams[p],np.sqrt(np.diag(bkgcurves[p])))


#for p in pmts:
	#chi2Plts.append(np.sum((multiexp(realtime,*v1) - timingCts[p]/np.sum(timingCts[p]))**2 \
							#/multiexp(realtime,*v1)) \
					#/ (len(realtime) - len(v1)) \
					#* (np.sum(timingCts[p])))

#try:
	#iniRun + 1
	#endRuns = [7332, 9600,13219,14509]
	#endRun = 14509
	## Hardcode because I'm dumb
	#if iniRun == 4231:
		#endRun = 7332
	#elif iniRun == 7332:
		#endRun = 9600
	#elif iniRun == 9600:
		#endRun = 13219
	#elif iniRun == 13219:
		#endRun = 14509
	#else: 
		#iniRun = 4200
	
	#title  = 'PMT Coincidence Structure, Runs ' + str(iniRun) +' to ' + str(endRun)
	#titleB = 'PMT Background Structure, Runs '  + str(iniRun) +' to ' + str(endRun)
#except:
	#title = 'PMT Coincidence Structure'
	#titleB = 'PMT Background Structure'

## With our fit, now we plot
##plt.figure()
#fig,axs = plt.subplots(4,sharex=True,sharey=True,gridspec_kw={'hspace':0})
#fig.suptitle(title)
#axs[0].plot(realtime,timingCts[0],'b+',label='PMT 1, PMT 1 First')
#axs[0].plot(realtime,multiexp(realtime,*v1),'g.')#,label=(r'$\chi^2 / NDF = %f$' % chi2Plts[0]))
#axs[1].plot(realtime,timingCts[1],'r+',label='PMT 2, PMT 1 First')
#axs[1].plot(realtime,multiexp(realtime,*v1),'g.')#,label=(r'$\chi^2 / NDF = %f$' % chi2Plts[1]))
#axs[2].plot(realtime,timingCts[2],'b+',label='PMT 1, PMT 2 First')
#axs[2].plot(realtime,multiexp(realtime,*v1),'g.')#,label=(r'$\chi^2 / NDF = %f$' % chi2Plts[2]))
#axs[3].plot(realtime,timingCts[3],'r+',label='PMT 2, PMT 2 First')
#axs[3].plot(realtime,multiexp(realtime,*v1),'g.')#,label=(r'$\chi^2 / NDF = %f$' % chi2Plts[3]))
#for ax in axs:
	#ax.label_outer()
	#ax.set_yscale('log')
	#ax.set_ylabel('Rate (arb.)')
	#ax.set_xlabel('Time (ns)')
	#ax.legend(loc='upper right')

#fig,axs = plt.subplots(4,sharex=True,sharey=True,gridspec_kw={'hspace':0})
#fig.suptitle(titleB)
#axs[0].plot(realtime,timingBkg[0]*s2b,'b+',label='PMT 1, PMT 1 First')
#axs[0].plot(realtime,multiexp(realtime,*v1),'g.')#,label=(r'$\chi^2 / NDF = %f$' % chi2Plts[0]))
#axs[1].plot(realtime,timingBkg[1]*s2b,'r+',label='PMT 2, PMT 1 First')
#axs[1].plot(realtime,multiexp(realtime,*v1),'g.')#,label=(r'$\chi^2 / NDF = %f$' % chi2Plts[1]))
#axs[2].plot(realtime,timingBkg[2]*s2b,'b+',label='PMT 1, PMT 2 First')
#axs[2].plot(realtime,multiexp(realtime,*v1),'g.')#,label=(r'$\chi^2 / NDF = %f$' % chi2Plts[2]))
#axs[3].plot(realtime,timingBkg[3]*s2b,'r+',label='PMT 2, PMT 2 First')
#axs[3].plot(realtime,multiexp(realtime,*v1),'g.')#,label=(r'$\chi^2 / NDF = %f$' % chi2Plts[3]))
#for ax in axs:
	#ax.label_outer()
	#ax.set_yscale('log')
	#ax.set_ylabel('Rate (arb.)')
	#ax.set_xlabel('Time (ns)')
	#ax.legend(loc='upper right')


#fig,axs = plt.subplots(4,sharex=True,sharey=True,gridspec_kw={'hspace':0})
#fig.suptitle(title)
#axs[0].plot(realtime[0:1000],timingCts[0][0:1000],'b+',label='PMT 1, PMT 1 First')
#axs[0].plot(realtime[0:1000],multiexp(realtime,*v1)[0:1000],'g.')#,label=(r'$\chi^2 / NDF = %f$' % chi2Plts[0]))
#axs[1].plot(realtime[0:1000],timingCts[1][0:1000],'r+',label='PMT 2, PMT 1 First')
#axs[1].plot(realtime[0:1000],multiexp(realtime,*v1)[0:1000],'g.')#,label=(r'$\chi^2 / NDF = %f$' % chi2Plts[1]))
#axs[2].plot(realtime[0:1000],timingCts[2][0:1000],'b+',label='PMT 1, PMT 2 First')
#axs[2].plot(realtime[0:1000],multiexp(realtime,*v1)[0:1000],'g.')#,label=(r'$\chi^2 / NDF = %f$' % chi2Plts[2]))
#axs[3].plot(realtime[0:1000],timingCts[3][0:1000],'r+',label='PMT 2, PMT 2 First')
#axs[3].plot(realtime[0:1000],multiexp(realtime,*v1)[0:1000],'g.')#,label=(r'$\chi^2 / NDF = %f$' % chi2Plts[3]))
#for ax in axs:
	#ax.label_outer()
	#ax.set_ylim(bottom=10^-5)
	#ax.set_yscale('log')
	#ax.set_ylabel('Rate (arb.)')
	#ax.set_xlabel('Time (ns)')
	#ax.legend(loc='upper right')

#plt.show()

## With our fit, now we plot
##plt.figure()
#fig,axs = plt.subplots(4,sharex=True,sharey=True,gridspec_kw={'hspace':0})
#fig.suptitle('Absolute Residuals')
#axs[0].plot(realtime,np.absolute(timingCts[0]-multiexp(realtime,*v1)),'k.',label='PMT 1, PMT 1 First Residual')
#axs[1].plot(realtime,np.absolute(timingCts[1]-multiexp(realtime,*v1)),'k.',label='PMT 2, PMT 1 First Residual')
#axs[2].plot(realtime,np.absolute(timingCts[2]-multiexp(realtime,*v1)),'k.',label='PMT 1, PMT 2 First Residual')
#axs[3].plot(realtime,np.absolute(timingCts[3]-multiexp(realtime,*v1)),'k.',label='PMT 2, PMT 2 First Residual')
#for ax in axs:
	#ax.label_outer()
	#ax.set_yscale('log')
	#ax.set_ylabel('Rate (arb.)')
	#ax.set_xlabel('Time (ns)')
	#ax.legend(loc='upper right')

#fig,axs = plt.subplots(4,sharex=True,sharey=True,gridspec_kw={'hspace':0})
#fig.suptitle('Absolute Background Residuals')
#axs[0].plot(realtime,np.absolute(timingBkg[0]*s2b-multiexp(realtime,*v1)),'k.',label='PMT 1, PMT 1 First Residual')
#axs[1].plot(realtime,np.absolute(timingBkg[1]*s2b-multiexp(realtime,*v1)),'k.',label='PMT 2, PMT 1 First Residual')
#axs[2].plot(realtime,np.absolute(timingBkg[2]*s2b-multiexp(realtime,*v1)),'k.',label='PMT 1, PMT 2 First Residual')
#axs[3].plot(realtime,np.absolute(timingBkg[3]*s2b-multiexp(realtime,*v1)),'k.',label='PMT 2, PMT 2 First Residual')
#for ax in axs:
	#ax.label_outer()
	#ax.set_yscale('log')
	#ax.set_xlabel('Time (ns)')
	#ax.set_ylabel('Rate (arb.)')
	#ax.legend(loc='upper right')

#plt.show()	
	
## With our fit, now we plot
##plt.figure()
#fig,axs = plt.subplots(4,sharex=True,sharey=True,gridspec_kw={'hspace':0})
#fig.suptitle('Residuals')
#axs[0].plot(realtime[0:1000],np.absolute(timingCts[0]-multiexp(realtime,*v1))[0:1000],'k.',label='PMT 1, PMT 1 First Residual')
#axs[1].plot(realtime[0:1000],np.absolute(timingCts[1]-multiexp(realtime,*v1))[0:1000],'k.',label='PMT 2, PMT 1 First Residual')
#axs[2].plot(realtime[0:1000],np.absolute(timingCts[2]-multiexp(realtime,*v1))[0:1000],'k.',label='PMT 1, PMT 2 First Residual')
#axs[3].plot(realtime[0:1000],np.absolute(timingCts[3]-multiexp(realtime,*v1))[0:1000],'k.',label='PMT 2, PMT 2 First Residual')
#for ax in axs:
	#ax.label_outer()
	##ax.set_yscale('log')
	#ax.set_xlabel('Time (ns)')
	#ax.set_ylabel('Rate (arb.)')
	#ax.legend(loc='upper right')

#plt.show()	

#pmt1 = timingCts[0]+timingCts[2]
#pmt2 = timingCts[1]+timingCts[3]
#pmt1B = timingBkg[0]+timingBkg[2]
#pmt2B = timingBkg[1]+timingCts[3]
#plt.figure()
#plt.title("PMT 1 vs PMT2 Residuals")
#plt.plot(realtime[0:1000],(pmt1 - 2*multiexp(realtime,*v1))[0:1000],'b.',label='PMT 1 Residual')
#plt.plot(realtime[0:1000],(pmt2 - 2*multiexp(realtime,*v1))[0:1000],'r.',label='PMT 2 Residual')
#plt.xlabel("Time (ns)")
#plt.ylabel("Rate (arb.)")
#plt.legend()

#plt.figure()
#plt.title("PMT 1 vs PMT2 Background Residuals")
#plt.plot(realtime[0:1000],(pmt1B - 2*multiexp(realtime,*v1))[0:1000],'b.',label='PMT 1 Residual')
#plt.plot(realtime[0:1000],(pmt2B - 2*multiexp(realtime,*v1))[0:1000],'r.',label='PMT 2 Residual')
#plt.xlabel("Time (ns)")
#plt.ylabel("Rate (arb.)")
#plt.legend()
#plt.show()

##fig.add_gridspec(
	
	
		

		
