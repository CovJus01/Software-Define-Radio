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

# use fmDemodArctan and fmPlotPSD
from fmSupportLib import my_fmDemod, fmDemodArctan, fmPlotPSD
import fmPll


# Mode Selection Prototype
mode = 0

if(mode == 0):
	rf_Fs = 2.4e6
	rf_decim = 10
	audio_decim = 5
	U_fact = 1
	D_fact = audio_decim
	if_Fs = (rf_Fs/rf_decim)*U_fact 	# = 240 Ksamples/sec
	audio_Fs = 48e3
elif(mode == 1):
	rf_Fs = 960e3
	rf_decim = 4
	audio_decim = 5
	U_fact = 1
	D_fact = audio_decim
	if_Fs = rf_Fs/rf_decim 	# = 240 Ksamples/sec
	audio_Fs = 48e3
elif(mode == 2):
	rf_Fs = 2.4e6
	rf_decim = 10
	audio_decim = -1 # Used to indicate that resampling is required
	U_fact = 147
	D_fact = 800
	if_Fs = rf_Fs/rf_decim	# = 240 Ksamples/sec
	audio_Fs = 44.1e3
elif(mode == 3):
	rf_Fs = 1.44e6
	rf_decim = 4
	audio_decim = -1 # Used to indicate resampling is required
	U_fact = 49
	D_fact = 400
	if_Fs = rf_Fs/rf_decim	# = 360 Ksamples/sec
	audio_Fs = 44.1e3




rf_Fc = 100e3
rf_taps = 151

audio_Fc = 16e3 # Cutoff Frequency
audio_taps = 151*U_fact
state_carr = np.zeros(audio_taps-1) # state saving for pilot tone block processing
carr_data = np.array([]) # array to hold filtered pilot data

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

def my_lowpass(N_taps, cutoff):
	h = [0] * N_taps
	for i in range(N_taps):
		if (i == (N_taps - 1)/2):
			h[i] = cutoff
		else:
			h[i] = cutoff * (( np.sin(cmath.pi*cutoff*(i-(N_taps-1)/2)))/(cmath.pi*cutoff*(i-(N_taps-1)/2)))
		h[i] = h[i] * (np.sin((i*cmath.pi) / (N_taps)))**2

	return h

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

def stereo_conv_resample(x, h, zi):
	# x = input sample array
	# h = input impulse response array (coefficients)
	# zi = initial filter state

	# Returns the resampled version of the signal
	du = 1  			# Upsample factor
	ds  = audio_decim 	# Downsample Factor
	y = [0] * int((du*(len(x)))/ds)

	for n in range(len(y)):
		phase = n*ds % du
		for k in range(len(h)):
			yu = n*ds
			xu = yu - (phase + k*du)
			if(xu>=0):
				y[n] += x[int(xu/du)]*h[phase + k*du]
			else:
				y[n] += zi[int(xu/du)]*h[phase + k*du]

	# Return the output of this convolution and the last values of x as the state
	return y, x[len(x)-len(h)+1:]


if __name__ == "__main__":

	# read the raw IQ data from the recorded file
	# IQ data is assumed to be in 8-bits unsigned (and interleaved)
	in_fname = "../data/stereo_l0_r9.raw"
	raw_data = np.fromfile(in_fname, dtype='uint8')
	print("Read raw RF data from \"" + in_fname + "\" in unsigned 8-bit format")
	# IQ data is normalized between -1 and +1 in 32-bit float format
	iq_data = (np.float32(raw_data) - 128.0)/128.0
	print("Reformatted raw RF data to 32-bit float format (" + str(iq_data.size * iq_data.itemsize) + " bytes)")

	# coefficients for the front-end low-pass filter
	rf_coeff = signal.firwin(rf_taps, rf_Fc/(rf_Fs/2), window=('hann'))

	# coefficients for the filter to extract mono audio
	audio_coeff = np.array(my_lowpass(audio_taps, audio_Fc/(if_Fs/2)))
	delay_coeff = np.zeros(int(audio_taps/D_fact))
	delay_coeff[int(len(delay_coeff)/2)] = 1

	fb = 18.5e3 # Beginning of pass band
	fe = 19.5e3 # End of pass band

	# Derive filter coefficients for pilot tone using scipy
	#h_carr = signal.firwin(audio_taps, [fb, fe], pass_zero="bandpass", fs = (rf_Fs/rf_decim))
	h_carr = my_bandpass(fb, fe, (rf_Fs/rf_decim), audio_taps)

	# Bandpass coefficient generation for stereo channel
	f_stereo_l = 22e3;
	f_stereo_r = 54e3;
	h_stereo = my_bandpass(f_stereo_l, f_stereo_r, if_Fs, audio_taps)
	#h_stereo = signal.firwin(audio_taps, [f_stereo_l/(if_Fs/2),f_stereo_r/(if_Fs/2)], window=('hann'), pass_zero=False)

	# set up the subfigures for plotting
	subfig_height = np.array([2, 2, 2, 0.8]) # relative heights of the subfigures
	plt.rc('figure', figsize=(10, 10))	# the size of the entire figure
	fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, gridspec_kw={'height_ratios': subfig_height})
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

	# Mono Channel Arrays
	mono_data = np.array([]) # used to concatenate filtered blocks (audio data)
	state_mono = np.zeros(audio_taps-1)
	state_mono_allpass = np.zeros(audio_taps-1)

	# Pilot tone arrays
	carr_block = np.zeros(int(block_size/2))
	state_carr = np.zeros(audio_taps-1)
	state_Pll = [0.0, 0.0, 1.0, 0.0, 1.0, 0]

	# Stereo Channel Arrays
	stereo_block = np.zeros(int((block_size/2)/rf_decim))
	state_stereo = np.zeros(audio_taps-1)
	audio_mix_state = np.zeros(audio_taps-1)
	stereo_data = np.array([])

	# Mixing Arrays
	audio_right = np.array([])
	right_block = np.zeros(1024)
	left_block = np.zeros(1024)
	audio_left = np.array([])
	audio_mix = np.zeros(int(len(stereo_block)/D_fact))

	# if the number of samples in the last block is less than the block size
	# it is fine to ignore the last few samples from the raw IQ file
	while (block_count+1)*block_size < len(iq_data)/6:

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

		#---------------------Mono Processing---------------------
		# extract the mono audio data through filtering
		mono_block, state_mono = conv_resample(fm_demod, audio_coeff, state_mono, U_fact, D_fact)
		#mono_block, state_mono = signal.lfilter(audio_coeff, 1.0, fm_demod, zi=state_mono)
		#mono_block = mono_block[::D_fact]


		# All pass filter for delay
		#mono_block, state_mono_allpass = signal.lfilter(delay_coeff, 1.0, mono_block, zi=state_mono_allpass)
		mono_block, state_mono_allpass = conv_resample(mono_block, delay_coeff, state_mono_allpass, 1, 1)

		# concatenate the most recently processed audio_block
		# to the previous blocks stored already in audio_data
		mono_data = np.concatenate((mono_data, mono_block))


        #---------------------Stereo Processing---------------------

		#	-> Stereo Carrier Recovery:
		# Filter to retrieve  pilot tone

		# Perform filtering
		#carr_block, state_carr= signal.lfilter(h_carr, 1.0, fm_demod, zi=state_carr)
		carr_block, state_carr = conv_resample(fm_demod, h_carr, state_carr, 1, 1)

		#carr_block, state_carr = stereo_conv_resample(carr_block, h_carr, state_carr)
		#carr_data = np.concatenate((carr_data, carr_block))

		# Synchronize with f_subcarrier
		carr_noPLL = carr_block
		carr_block, state_Pll = fmPll.fmPll(carr_block, 19e3, (rf_Fs/rf_decim), state_Pll, ncoScale = 2.0)
		# Add block to data
		carr_data = np.concatenate((carr_data, carr_block))


		#	-> Stereo Channel Extraction
		#stereo_block, state_stereo = signal.lfilter(h_stereo, 1.0, fm_demod, zi=state_stereo)
		stereo_block, state_stereo = conv_resample(fm_demod, h_stereo, state_stereo, 1, 1)
		stereo_data = np.concatenate((stereo_data, stereo_block))

		# Perform mixing and downsampling
		audio_mix = 2*stereo_block * carr_block[0:(len(carr_block)-1)]
		#audio_mix, audio_mix_state = signal.lfilter(audio_coeff, 1.0, audio_mix, zi=audio_mix_state)
		#audio_mix = audio_mix[::D_fact]
		audio_mix, audio_mix_state = conv_resample(audio_mix, audio_coeff, audio_mix_state, U_fact, D_fact)

		# Stereo combining
		left_block = mono_block + audio_mix
		right_block = mono_block - audio_mix
		'''
		for i in range(len(mono_block)):
			left_block[i] = mono_block[i] + audio_mix[i]
			right_block[i] = mono_block[i] - audio_mix[i]
		'''
		audio_left = np.concatenate((audio_left, left_block))
		audio_right = np.concatenate((audio_right, right_block))
		#print(audio_data)

		# Checking stereo channels are different:
		'''
		diff = False
		for j in range(len(audio_right)):
			if audio_right[j] != audio_left[j]:
				diff = True
				break
		if (diff == False):
			print("Audio channels are identical")
		else:
			print("Audio channels are different")
		'''

		# Plot PSD of output
		if block_count > 10 and block_count < 12:

			# plot PSD of selected block after FM demodulation
			ax0.clear()
			fmPlotPSD(ax0, stereo_block, (rf_Fs/rf_decim)/1e3, subfig_height[2], \
					'Extracted Stereo (block ' + str(block_count) + ')')
			# output binary file name (where samples are written from Python)
			fm_demod_fname = "../data/fm_demod_" + str(block_count) + ".bin"
			# create binary file where each sample is a 32-bit float
			fm_demod.astype('float32').tofile(fm_demod_fname)

			# plot PSD of selected block after extracting stereo carrier recovery
			fmPlotPSD(ax1, carr_noPLL, (rf_Fs/rf_decim)/1e3, subfig_height[2], \
					'Extracted Stereo Carrier Recovery, no PLL (block ' + str(block_count) + ')')

			# plot PSD of selected block after extracting stereo carrier recovery
			fmPlotPSD(ax2, carr_block, (rf_Fs/rf_decim)/1e3, subfig_height[2], \
					'Extracted Stereo Carrier Recovery, with PLL (block ' + str(block_count) + ')')

			# plot PSD of selected block after extracting stereo carrier recovery
			fmPlotPSD(ax3, audio_mix, audio_Fs/1e3, subfig_height[0], \
					'Mixed audio (block ' + str(block_count) + ')')

			# save figure to file
			fig.savefig("../data/fmStereoBlock" + str(block_count) + ".png")

		block_count += 1

	print('Finished processing all the blocks from the recorded I/Q samples')

	# write audio data to file
	out_fname = "../data/fmStereo.wav"
	stereo_out = np.array([np.int16((audio_left/2)*32767), np.int16((audio_right/2)*32767)]).T
	wavfile.write(out_fname, int(audio_Fs), stereo_out)
	print("Written audio samples to \"" + out_fname + "\" in signed 16-bit format")

	# uncomment assuming you wish to show some plots
	plt.show()
