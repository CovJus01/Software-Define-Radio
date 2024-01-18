
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
from scipy import signal


from fmSupportLib import estimatePSD

# Read binary file
in_fname_left = "../data/test_data_stereo_l.bin"
in_fname_right = "../data/test_data_stereo_r.bin"
raw_fname = "../data/stereo_l0_r9.raw"
bin_fname = "../data/stereo_l0_r9.bin"
outwave_fname = "../data/test_wave_stereo.wav"


def fmPlot(samples, Fs, title, height=1, psd=True):
    fig, ax = plt.subplots()
    x_major_interval = (Fs/12)		# adjust grid lines as needed
    x_minor_interval = (Fs/12)/4
    y_major_interval = 20
    x_epsilon = 1e-3
    if(psd):
        x_max = x_epsilon + Fs/2		# adjust x/y range as needed
        x_min = 0
        y_max = 10
        y_min = y_max-100*height
    else:
        x_max = Fs
        x_min = 0
        y_max = max(samples)
        y_min = min(samples)
    # ax.psd(samples, NFFT=512, Fs=Fs)
    #
    # below is the custom PSD estimate, which is based on the Bartlett method
    # it less accurate than the PSD from matplotlib, however it is sufficient
    # to help us visualize the power spectra on the acquired/filtered data
    #
    if(psd):
        freq, my_psd = estimatePSD(samples, NFFT=512, Fs=Fs)
        ax.plot(freq, my_psd)
    else:
        ax.plot(samples)
    #
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xticks(np.arange(x_min, x_max, x_major_interval))
    ax.set_xticks(np.arange(x_min, x_max, x_minor_interval), minor=True)
    ax.set_yticks(np.arange(y_min, y_max, y_major_interval))
    ax.grid(which='major', alpha=0.75)
    ax.grid(which='minor', alpha=0.25)
    if(psd):
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('PSD (db/Hz)')
    else:
        ax.set_xlabel('Sample #')
        ax.set_ylabel('Magnitude')
    ax.set_title(title)

def raw_to_bin(raw_fname, bin_fname):
    print("Read raw RF data from \"" + raw_fname + "\" in unsigned 8-bit format")
    raw_data = np.fromfile(raw_fname, dtype='uint8')     # For extracting raw data
    float_data = (np.float32(raw_data) - 128.0)/128.0
    print("Write binary RF data to \"" + bin_fname + "\" in float 32-bit format")
    float_data.astype('float32').tofile(bin_fname)

def write_to_wav(outwave_fname, audio_data, audio_Fs):
    wavfile.write(outwave_fname, int(audio_Fs), np.int16((audio_data/2)*32767))
    print("Written audio samples to \"" + outwave_fname + "\" in signed 16-bit format")

def write_twochannels_to_wav(outwave_fname, audio_left, audio_right, audio_Fs):
    stereo_out = np.array([np.int16((audio_left/2)*32767), np.int16((audio_right/2)*32767)]).T
    wavfile.write(outwave_fname, int(audio_Fs), stereo_out)
    print("Written audio samples to \"" + outwave_fname + "\" in signed 16-bit format")

if __name__ == "__main__":
    data_left = np.fromfile(in_fname_left, dtype='float32')     # For extracting bin data
    data_right = np.fromfile(in_fname_right, dtype='float32')     # For extracting bin data
    print("Read binary RF data from \"" + in_fname_left + "\" in float 32-bit format")
    print("Read binary RF data from \"" + in_fname_right + "\" in float 32-bit format")
    # data = np.fromfile(in_fname, dtype='uint8')     # For extracting raw data
    # data = (np.float32(data) - 128.0)/128.0     # Converting raw data to float32
    #raw_to_bin(raw_fname, bin_fname)

    # print("test_data = ", data)
    # print("test_data length = ", len(data))
    # plt.plot(data)

    # fmPlot(data[len(data)-2048:], 2048, "Time Plot", psd=False)
    start = 1000;
    fmPlot(data_left[start:start + 1024], 48, "Left Audio Data")
    # fmPlot(data, 2.4e2, "Second Plot")

    write_twochannels_to_wav(outwave_fname, data_left, data_right, 48e3)
    plt.show()
