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

import time

# use fmDemodArctan and fmPlotPSD
from fmSupportLib import my_fmDemod, fmDemodArctan, fmPlotPSD

mode = 0

# Mode selection code
if(mode == 0):
	rf_Fs = 2.4e6
	rf_decim = 10
	if_Fs = rf_Fs/rf_decim 	# = 240 Ksamples/sec
	audio_decim = 5
	audio_Fs = 48e3
	mix_decim = 5
elif(mode == 1):
	rf_Fs = 960e3
	rf_decim = 4
	if_Fs = rf_Fs/rf_decim 	# = 240 Ksamples/sec
	audio_decim = 5
	audio_Fs = 48e3
elif(mode == 2):
	rf_Fs = 2.4e6
	rf_decim = 10
	if_Fs = rf_Fs/rf_decim	# = 240 Ksamples/sec
	audio_decim = -1 # Used to indicate that resampling is required
	audio_Fs = 44.1e3
elif(mode == 3):
	rf_Fs = 1.44e6
	rf_decim = 4
	if_Fs = rf_Fs/rf_decim	# = 360 Ksamples/sec
	audio_decim = -1 # Used to indicate resampling is required
	audio_Fs = 44.1e3


rf_Fc = 100e3
rf_taps = 151

audio_Fc = 16e3 # Cutoff Frequency
audio_taps = 151 # Number of taps for the filter
U_fact = 1
D_fact = U_fact*audio_decim
audio_taps_US = audio_taps * U_fact

def my_lowpass(N_taps, cutoff):
	h = [0] * N_taps
	for i in range(N_taps):
		if (i == (N_taps - 1)/2):
			h[i] = cutoff
		else:
			h[i] = cutoff * (( np.sin(cmath.pi*cutoff*(i-(N_taps-1)/2)))/(cmath.pi*cutoff*(i-(N_taps-1)/2)))
		h[i] = h[i] * (np.sin((i*cmath.pi) / (N_taps)))**2

	return h

def mono_conv_resample(x, h, zi, du, ds):
	# x = input sample array
	# h = input impulse response array (coefficients)
	# zi = initial filter state
	# returns output signal passed through filter and the filter delay

	y = np.zeros(int((du*(len(x)))/ds))
	for n in range(len(y)):
		yu = n*ds
		phase = yu % du
		for k in range(int(len(h)/du)):
			hu = k*du + phase
			x_ind = int((yu - hu)/du)
			#if(n >=15 and n<=20):
				#print("yu[%d] = xu[%d]*hu[%d]" %(yu, x_ind, hu))
			if(x_ind>=0):
				y[n] += du*(x[x_ind]*h[hu])
			else:
				y[n] += du*(zi[x_ind]*h[hu])


	# Return the output of this convolution and the last values of x as the state
	return y, x[len(x)-len(h)+1:]

'''
def mono_conv_resample_slow(x, h, zi, du, ds):
	# x = input sample array
	# h = input impulse response array (coefficients)
	# zi = initial filter state
	# returns output signal passed through filter and the filter delay

	y = np.zeros(du*len(x))
	# Upsample the input
	xu = np.zeros(du*len(x))
	for n in range(len(x)):
		xu[n*du] = x[n]

	# Filter
	for n in range(len(xu)):
		#print("yu = ", yu)
		for k in range(len(h)):
			if(n-k>=0):
				y[n] += du*(xu[n-k]*h[k])
			else:
				y[n] += du*(zi[n-k]*h[k])

	# Downsample the output
	# Return the output of this convolution and the last values of x as the state
	return y[::ds], x[len(x)-len(h)+1:]
'''

def RF_filter_downsample(x, h, zi):
	# x = input sample array
	# h = input impulse response array (coefficients)
	# zi = initial filter state
	# returns output signal passed through filter and the filter delay

	# Only convolve to the length of the block (The length of input x)
	# (Same effect as truncating the output of the convolution)
	decim = audio_decim
	K = len(h)
	y = np.zeros(int((len(x))/decim))

	for n in range(int(len(x)/decim)):
		for k in range(K):
			N = decim*n
			if (N-k) >= 0:
				# if (N-k) < len(x):
				y[n] = y[n] + x[N-k]*h[k]
			else:
				# Use the previous state's values when the x index goes negative
				y[n] = y[n] + zi[N-k]*h[k]

	# Return the output of this convolution and the last values of x as the state
	return y, x[len(x)-len(h)+1:]

'''
def my_filter(x, h, zi):
	# x = input sample array
	# h = input impulse response array (coefficients)
	# zi = initial filter state
	# returns output signal passed through filter and the filter delay

	# Only convolve to the length of the block (The length of input x)
	# (Same effect as truncating the output of the convolution)
	N = len(x)
	y = [0] * N
	K = len(h)

	for n in range(N):
		for k in range(K):
				if (n-k) >= 0:
					y[n] = y[n] + x[n-k]*h[k]
				else:
					# Use the previous state's values when the x index goes negative
					y[n] = y[n] + zi[n-k]*h[k]

	# Return the output of this convolution and the last values of x as the state
	return y, x[len(x)-len(h)+1:]
'''

if __name__ == "__main__":
	# read the raw IQ data from the recorded file
	# IQ data is assumed to be in 8-bits unsigned (and interleaved)
	in_fname = "../data/iq_samples.raw"
	raw_data = np.fromfile(in_fname, dtype='uint8')
	print("Read raw RF data from \"" + in_fname + "\" in unsigned 8-bit format")
	# IQ data is normalized between -1 and +1 in 32-bit float format
	iq_data = (np.float32(raw_data) - 128.0)/128.0
	print("Reformatted raw RF data to 32-bit float format (" + str(iq_data.size * iq_data.itemsize) + " bytes)")

	# coefficients for the front-end low-pass filter
	rf_coeff = signal.firwin(rf_taps, rf_Fc/(rf_Fs/2), window=('hann'))

	# coefficients for the filter to extract mono audio
	# audio_coeff = signal.firwin(audio_taps_US, (audio_Fc)/(if_Fs*U_fact/2), window=('hann'))
	audio_coeff = np.array(my_lowpass(audio_taps_US, audio_Fc/(if_Fs/2)))

	# set up the subfigures for plotting
	subfig_height = np.array([0.8, 2, 1.6]) # relative heights of the subfigures
	plt.rc('figure', figsize=(7.5, 7.5))	# the size of the entire figure
	fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, gridspec_kw={'height_ratios': subfig_height})
	fig.subplots_adjust(hspace = .6)

	# select a block_size that is a multiple of KB
	# and a multiple of decimation factors
	block_size = 1024 * rf_decim * audio_decim * 2
	block_count = 0

	# states needed for continuity in block processing
	state_i_lpf_100k = np.zeros(rf_taps-1)
	state_q_lpf_100k = np.zeros(rf_taps-1)
	state_phase = 0
	I_state = 0
	Q_state = 0
	# add state as needed for the mono channel filter
	state_mono = np.zeros(audio_taps_US-1)

	# audio buffer that stores all the audio blocks
	audio_data = np.array([]) # used to concatenate filtered blocks (audio data)

	# if the number of samples in the last block is less than the block size
	# it is fine to ignore the last few samples from the raw IQ file
	st = time.time()
	while (block_count+1)*block_size < len(iq_data):

		# if you wish to have shorter runtimes while troubleshooting
		# you can control the above loop exit condition as you see fit
		print('Processing block ' + str(block_count))


		# filter to extract the FM channel (I samples are even, Q samples are odd)
		'''
		i_filt, state_i_lpf_100k = RF_filter_downsample(iq_data[(block_count)*block_size:(block_count+1)*block_size:2], \
		rf_coeff, state_i_lpf_100k)
		q_filt, state_q_lpf_100k = RF_filter_downsample(iq_data[(block_count)*block_size+1:(block_count+1)*block_size:2], \
		rf_coeff, state_q_lpf_100k)
		'''
		# This should be using RF_filter_downsample (As above) but it's too slow to be included here
		i_filt, state_i_lpf_100k = signal.lfilter(rf_coeff, 1.0, \
				iq_data[(block_count)*block_size:(block_count+1)*block_size:2],
				zi=state_i_lpf_100k)
		q_filt, state_q_lpf_100k = signal.lfilter(rf_coeff, 1.0, \
				iq_data[(block_count)*block_size+1:(block_count+1)*block_size:2],
				zi=state_q_lpf_100k)

		# downsample the I/Q data from the FM channel (Remove if using Rf_Filter_downsample)
		i_ds = i_filt[::rf_decim]
		q_ds = q_filt[::rf_decim]

		#i_ds = i_filt
		#q_ds = q_filt

		# FM demodulator, Last step for RF Front-End
		fm_demod, I_state, Q_state = my_fmDemod(np.array(i_ds), np.array(q_ds), I_state, Q_state)
		# Begin Mono Audio Processing
		# extract the mono audio data through filtering and downsample
		audio_filt, state_mono = mono_conv_resample(fm_demod, audio_coeff, state_mono, U_fact, D_fact)

		# concatenate the most recently processed audio_block
		# to the previous blocks stored already in audio_data
		audio_data = np.concatenate((audio_data, audio_filt))


		# to save runtime select the range of blocks to log data
		# this includes both saving binary files as well plotting PSD
		# below we assume we want to plot for graphs for blocks 10 and 11
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
			fmPlotPSD(ax1, audio_data, audio_Fs/1e3, subfig_height[0], \
					'Extracted mono audio (block ' + str(block_count) + ')')

			# save figure to file
			fig.savefig("../data/fmMonoBlock" + str(block_count) + ".png")

		block_count += 1
	et = time.time()
	print('Finished processing all the blocks from the recorded I/Q samples')
	print('Elapsed Time = ', et-st)
	# write audio data to file
	out_fname = "../data/fmMonoBlock.wav"
	wavfile.write(out_fname, int(audio_Fs), np.int16((audio_data/2)*32767))
	print("Written audio samples to \"" + out_fname + "\" in signed 16-bit format")

	# uncomment assuming you wish to show some plots
	plt.show()
