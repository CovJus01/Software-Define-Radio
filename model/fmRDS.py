#
# Comp Eng 3DY4 (Computer Systems Integration Project)
#
# Copyright by Nicola Nicolici
# Department of Electrical and Computer Engineering
# McMaster University
# Ontario, Canada
#

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
import math, cmath
from fmSupportLib import my_fmDemod, fmDemodArctan, fmPlotPSD
import fmPll
import fmRRC


rf_Fs = 2.4e6
rf_Fc = 100e3
rf_taps = 501
rf_decim = 10
if_Fs =  2.4e5


rds_decim = 5 # Decimation Rate
rds_taps = 501 # Number of taps for the filter
#Samples per symbol
SPS = 19
def conv_resample(x, h, zi, U, D):
	# x = input sample array
	# h = input impulse response array (coefficients)
	# zi = initial filter state
	# D = Downsample factor
	# U = Upsample factor
	# returns output signal passed through filter and the filter delay

	y = np.zeros(int((U*(len(x)))/D))

	for n in range(len(y)):
		yu = n*D
		phase = yu % U
		for k in range(int(len(h)/U)):
			hu = k*U + phase
			x_ind = int((yu - hu)/U)
			if(x_ind>=0):
				y[n] += x[x_ind]*h[hu]
			else:
				y[n] += zi[x_ind]*h[hu]

	# Return the output of this convolution and the last values of x as the state
	return y, x[len(x)-len(h)+1:]

#Bandpass filter implementation
def my_bandpass(fb, fe, fs, N_taps):
	h = [0] * N_taps
	norm_center = ((fe + fb)/2.0)/(fs/2.0)
	norm_pass = (fe - fb)/(fs/2.0)

	for i in range(N_taps):
		if (i == ((N_taps-1)/2)):
			h[i] = norm_pass # avoid division by zero in sinc function
		else:
			h[i] = norm_pass * (np.sin(cmath.pi*(norm_pass/2.0)*(i-((N_taps-1)/2))) / (np.pi*(norm_pass/2)*(i-((N_taps-1)/2))))

		h[i] = h[i] * np.cos(i*np.pi*norm_center) # apply a frequnecy shift by the center frequency
		h[i] = h[i]*pow(np.sin((i*np.pi)/(N_taps)),2) # apply the Hann window

	return h


#Squaring extracted signal
def my_squaring(signal):
	squared = np.zeros(len(signal))
	for i in range(0, len(signal)):
		squared[i] = signal[i]**2

	return squared

def my_lowpass(N_taps, cutoff):
	h = [0] * N_taps
	for i in range(N_taps):
		if (i == (N_taps - 1)/2):
			h[i] = cutoff
		else:
			h[i] = cutoff * (( np.sin(cmath.pi*cutoff*(i-(N_taps-1)/2)))/(cmath.pi*cutoff*(i-(N_taps-1)/2)))
		h[i] = h[i] * (np.sin((i*cmath.pi) / (N_taps)))**2

	return h

def my_CDR(wave):

	#Array to store values
	data_array = [0]*(int(len(wave)/SPS))
	#Find offset
	max =  np.argmin(abs(wave[:SPS*2-2])) + int(SPS/2)
	offset = max % 19
	print(offset)
	#Gather every symbol
	for i in range(int(len(wave)/SPS)):
		if(wave[(i*SPS) +offset]> 0):
			data_array[i] = 1
	return data_array

if __name__ == "__main__":

	# read the raw IQ data from the recorded file
	# IQ data is assumed to be in 8-bits unsigned (and interleaved)
	in_fname = "../data/samples8.raw"
	raw_data = np.fromfile(in_fname, dtype='uint8')
	print("Read raw RF data from \"" + in_fname + "\" in unsigned 8-bit format")
	# IQ data is normalized between -1 and +1 in 32-bit float format
	iq_data = (np.float32(raw_data) - 128.0)/128.0
	print("Reformatted raw RF data to 32-bit float format (" + str(iq_data.size * iq_data.itemsize) + " bytes)")

	# coefficients for the front-end low-pass filter
	rf_coeff = signal.firwin(rf_taps, rf_Fc/(rf_Fs/2), window=('hann'))

	# coefficients for the filter to extract RDBS signal
	extraction_coeff = my_bandpass(54e3, 60e3, if_Fs, rds_taps)
	# coefficients for the filter to recover the carrier
	recovery_coeff = my_bandpass(113.5e3,114.5e3, if_Fs, rds_taps)
	#coefficients for the allpass filter
	allpass_coeff = [0]*(rds_taps)

	#BEST FILTER FOR THIS AMOUNT OF TAPS, NEED TO KEEP RATIO SIMILAR FOR BEST CONSTELLATION 21
	#21 is the flattest I found
	allpass_coeff[int(rds_taps/2) + 7] = 1
	#coefficients for the lowpass demodulating filter
	lowpass_coeff = my_lowpass(rds_taps, 3.0e3/(if_Fs/2))
	lowpass_resample_coeff = my_lowpass(rds_taps*361, 3.0e3/(if_Fs/2))
	#Root-raised cosine filter coefficients
	RRC_coeff = fmRRC.impulseResponseRootRaisedCosine(45.125e3, 19)

	subfig_height = np.array([0, 0, 2,4]) # relative heights of the subfigures
	plt.rc('figure', figsize=(7.5, 7.5))	# the size of the entire figure
	fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, gridspec_kw={'height_ratios': subfig_height})
	fig.subplots_adjust(hspace = .6)

	# select a block_size that is a multiple of KB
	# and a multiple of decimation factors
	block_size = 1024 * rf_decim * rds_decim * 2
	block_count = 0

	# states needed for continuity in block processing
	state_i_lpf_100k = np.zeros(rf_taps-1)
	state_q_lpf_100k = np.zeros(rf_taps-1)
	state_phase = 0
	I_state = 0
	Q_state = 0
	# states for RDBS and PLL
	state_rds = np.zeros(rds_taps-1)
	state_allpass = np.zeros(rds_taps-1)
	state_carr = np.zeros(rds_taps-1)
	state_pll = [0.0, 0.0, 1.0, 0.0, 1.0, 0]
	state_pll_Q = [0.0, 0.0, 1.0, 0.0, 1.0, 0]
	state_lowpass = np.zeros(rds_taps-1)
	state_lowpass_Q = np.zeros(rds_taps-1)
	state_lowpass_resample = np.zeros(rds_taps*361-1)
	state_lowpass_resample_Q = np.zeros(rds_taps*361-1)
	state_RRC = np.zeros(19-1)
	state_RRC_Q = np.zeros(19-1)
	rds_extract = np.zeros(int((block_size/2)/rf_decim))

	# audio buffer that stores all the audio blocks
	rds_data = np.array([])
	rds_mixed = []
	rds_resample = [] # used to concatenate filtered blocks (audio data)
	# if the number of samples in the last block is less than the block size
	# it is fine to ignore the last few samples from the raw IQ file
	#while (block_count+1)*block_size < len(iq_data):
	while block_count< 12:

		# if you wish to have shorter runtimes while troubleshooting
		# you can control the above loop exit condition as you see fit

        #---------------------RF FRONTEND---------------------
		print('Processing block ' + str(block_count))

		# filter to extract the FM channel (I samples are even, Q samples are odd)
		i_filt, state_i_lpf_100k = signal.lfilter(rf_coeff, 1.0, \
				iq_data[(block_count)*block_size:(block_count+1)*block_size:2],
				zi=state_i_lpf_100k)
		q_filt, state_q_lpf_100k = signal.lfilter(rf_coeff, 1.0, \
				iq_data[(block_count)*block_size+1:(block_count+1)*block_size:2],
				zi=state_q_lpf_100k)
		# downsample the I/Q data from the FM channel
		i_ds = i_filt[::rf_decim]
		q_ds = q_filt[::rf_decim]

		# FM demodulator, Last step for RF Front-End
		fm_demod, I_state, Q_state = my_fmDemod(np.array(i_ds), np.array(q_ds), I_state, Q_state)


        #---------------------END RF FRONTEND---------------------

        #---------------------RDS Processing---------------------



		#RDS channel Extraction
		rds_extract, state_rds = signal.lfilter(extraction_coeff, 1.0, fm_demod, zi=state_rds)

		#RDS Carrier Recovery-----------------------------------
		#Squaring
		rds_squared = my_squaring(rds_extract)

		#Second BPF
		rds_carrier, state_carr = signal.lfilter(recovery_coeff, 1.0, rds_squared, zi=state_carr)
		rds_recover, state_pll = fmPll.fmPll(rds_carrier, 114e3, if_Fs, state_pll, ncoScale= 0.5)
		rds_recover_Q, state_pll_Q = fmPll.fmPll(rds_carrier, 114e3, if_Fs, state_pll_Q, ncoScale= 0.5, phaseAdjust= math.pi/2)

		#-------------------------------------------------------
		#All pass filter?
		rds_allpass , state_allpass = signal.lfilter(allpass_coeff, 1.0, rds_extract, zi = state_allpass)

		#RDS MIX------------------------------------------------
		rds_mixed = 2*rds_allpass*rds_recover[:len(rds_recover)-1]
		rds_mixed_Q = 2*rds_allpass*rds_recover_Q[:len(rds_recover)-1]
		#-------------------------------------------------------

		#RDS demodulation---------------------------------------

		rds_lowpassed, state_lowpass = signal.lfilter(lowpass_coeff, 1.0, rds_mixed, zi=state_lowpass)
		rds_lowpassed_Q, state_lowpass_Q = signal.lfilter(lowpass_coeff, 1.0, rds_mixed_Q, zi=state_lowpass_Q)
		#Resampling (Can use same as mono for mode 0)
		#Mode 0 = 19x2375 GCD = 125, U=361, D = 1920
		#Mode 2 =42 x 2375 Ksamples/sec

		rds_resample , state_lowpass_resample = conv_resample(rds_lowpassed, lowpass_resample_coeff, state_lowpass_resample, 361, 1920)
		rds_resample_Q , state_lowpass_resample_Q = conv_resample(rds_lowpassed_Q, lowpass_resample_coeff, state_lowpass_resample_Q, 361, 1920)
		#RRC filtering
		rds_RRC, state_RRC = signal.lfilter(RRC_coeff, 1.0, rds_resample, zi= state_RRC)
		rds_RRC_Q, state_RRC_Q = signal.lfilter(RRC_coeff, 1.0, rds_resample_Q, zi= state_RRC_Q)

		#Clock and data recovery

		rds_cdr = my_CDR(rds_RRC)

		#-------------------------------------------------------


		#RDS Data Processing-----------------------------------


		#-------------------------------------------------------

		#RDS FINAL----------------------------------------------

		rds_data = np.concatenate((rds_data, rds_recover))

		if block_count >= 11 and block_count < 12:

			# plot PSD of selected block after FM demodulation
			ax0.clear()
			fmPlotPSD(ax0, fm_demod, (rf_Fs/rf_decim)/1e3, subfig_height[0], \
					'Demodulated FM (block ' + str(block_count) + ')')
			# output binary file name (where samples are written from Python)
			fm_demod_fname = "../data/fm_demod_" + str(block_count) + ".bin"
			# create binary file where each sample is a 32-bit float
			fm_demod.astype('float32').tofile(fm_demod_fname)

			# plot PSD of selected block after extracting mono audio
			fmPlotPSD(ax1, rds_extract, if_Fs/1e3, subfig_height[1], \
					'Extracted rds (block' + str(block_count) + ')')


			ax2.plot(range(len(rds_RRC)), rds_RRC)

			#ax5.plot(range(len(rds_RRC)), rds_RRC)
			ax2.grid(which='major', alpha=0.75)
			ax2.grid(which='minor', alpha=0.25)
			ax2.set_xlabel('Sample #')
			ax2.set_ylabel('Signal')
			ax2.set_title("Resampled")

			for i in rds_cdr:
				print(i)
			#TO TUNE PLL CONSTELLATION
			ax3.scatter(rds_RRC, rds_RRC_Q, s=1)
			ax3.set_xlim(-0.012, 0.012)
			ax3.set_ylim(-0.012, 0.012)
			ax3.set_aspect(1)




		block_count += 1

	print('Finished processing all the blocks from the recorded I/Q samples')


	# uncomment assuming you wish to show some plots
	plt.show()
