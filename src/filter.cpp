/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "filter.h"
#include <stdio.h>
#include <cmath>
#include <algorithm>

#include"iofunc.h"


// function to compute the impulse response "h" based on the sinc function
void impulseResponseLPF(float Fs, float Fc, unsigned short int num_taps, std::vector<float> &h)
{
  /* Allocate memory for the impulse response */
  h.clear(); h.resize(num_taps, 0.0);
  /* Define the normalized cutoff freqency */
  float cutoff = Fc/(Fs/2.0);
  /* Used in a lot of the future computations */
  float half_len = (num_taps - 1)/2;

  for (unsigned int i=0; i < num_taps; i++) {
    /* Deal with case of division by zero */
    if (i == static_cast<unsigned int>(half_len)) {
      h[i] = cutoff;
    }
    /* Calculate filter coefficients based on the sinc function */
    else {
      h[i] = cutoff * sin(PI*cutoff*(i-half_len))/(PI*cutoff*(i-half_len));
    }
    /* Apply Hann window */
    h[i] = h[i] * pow(sin((i*PI)/num_taps), 2);
  }
}

/* Function to compute the impulse response 'h' of a bandpass filter with
   beginning of passband at frequency Fb and end of passband at Frequency
   Fe */
void impulseResponseBPF(float Fs, float Fb, float Fe, unsigned short int num_taps, std::vector<float> &h) {
  /* Define the normalized center frequency */
  float norm_center = ((Fe+Fb)/2.0)/(Fs/2.0);
  /* Define the normalized passband */
  float norm_pass = (Fe-Fb)/(Fs/2.0);

  float half_len = (num_taps - 1)/2;
  /* Clear impulse response before filling with new filter coefficients */
  h.clear();
  h.resize(num_taps, 0.0);

  for (unsigned int i=0;i<num_taps;i++) {
    /* Deal with case of division by zero */
    if (i == static_cast<unsigned int>(half_len)) {
      h[i] = norm_pass;
    }
    /* Calculate filter coefficients based on the sinc function */
    else {
      h[i] = norm_pass*(sin(PI*(norm_pass/2.0)*(i-half_len))/(PI*(norm_pass/2.0)*(i-half_len)));
    }
    /* Apply a frequency shift by the center frequency */
    h[i] = h[i] * cos(i*PI*norm_center);
    /* Apply the Hann window */
    h[i] = h[i] * pow(sin((i*PI)/(num_taps)),2);
  }
}

// function to compute the filtered output "y" by doing the convolution
// of the input data "x" with the impulse response "h"
void convolveFIR_DS(std::vector<float> &y, const std::vector<float> &x, const std::vector<float> &h, std::vector<float> &filter_state, const signed int D)
{
	// allocate memory for the output (filtered) data
	y.clear();
  y.resize(static_cast<int>(x.size()/D), 0.0);
  signed int N;
  signed int diff; // So N-k (two unsigned ints) make negative numbers

	for(unsigned int n = 0; n < y.size() ; n++){
    N = D*n;
		for(signed int k = 0; k < static_cast<signed int>(h.size()); k++){
      diff = static_cast<signed int>(N-k);
			if(diff >= 0){
				y[n] = y[n] + h[k] * x[diff];
			}
			else {
				y[n] = y[n] + filter_state[filter_state.size()+(diff)] * h[k];
			}
		}
	}
  //y.assign(y.begin(), y.end()-h.size()+1); //Truncate output to block size
	// Update filter filter_state
	filter_state.assign(x.end()-h.size(), x.end());
}

// function to compute the filtered output "y" by doing the convolution
// of the input data "x" with the impulse response "h"
void convolveFIR_Resample(std::vector<float> &y, const std::vector<float> &x, const std::vector<float> &h, std::vector<float> &filter_state, const signed int U, const signed int D)
{
	// allocate memory for the output (filtered) data
	y.clear();
  y.resize(static_cast<unsigned int>((U*x.size())/D), 0.0);
  int yu, hu, phase;
  int x_ind; // So N-k (two unsigned ints) make negative numbers

	for(int n = 0; n < static_cast<int>(y.size()) ; n++){
    yu = n*D;
    phase = yu % U;
		for(int k = 0; k < static_cast<int>(h.size()/U); k++){
      hu = k*U + phase;
      x_ind = static_cast<int>((yu-hu)/U);
			if(x_ind >= 0){
				y[n] = y[n] + h[hu] * x[x_ind] * U;
			}
			else {
				y[n] = y[n] + filter_state[static_cast<signed int>(filter_state.size())+(x_ind)] * h[hu] * U;
			}
		}
	}
  //y.assign(y.begin(), y.end()-h.size()+1); //Truncate output to block size
	// Update filter filter_state
	filter_state.assign(x.end()-static_cast<int>((h.size()/U)+1), x.end());
}

void fmDemod(std::vector<float> &fm_demod, const std::vector<float> &I, const std::vector<float> &Q, std::vector<float> &prev_IQ) {
  // fm_demod = vector to store demodulated values
  // I = vector to store in-phase values from RF Dongle
  // Q = vector to store quadrature values from RF Dongle
  // prev_IQ = array to store previous I and Q values, in form [I_prev, Q_prev]

  // Clear fm_demod before filling with demodulated values
  fm_demod.clear();
  fm_demod.resize(I.size());

  float DI, DQ, denom;
  for (unsigned int k=0; k<I.size(); k++) {
    // Calculate instantaneous changes in I and Q
    DI = I[k] - prev_IQ[0];
    DQ = Q[k] - prev_IQ[1];
    // Calculate change in phase (fm_demod)
    denom = pow(I[k], 2) + pow(Q[k], 2);
    if (denom == 0) {
      fm_demod[k] = 0;
    }
    else {
      fm_demod[k] = (I[k]*DQ - Q[k]*DI)/denom;
    }
    //Assign previous I anad Q values for next iteration of loop
    prev_IQ[0] = I[k];
    prev_IQ[1] = Q[k];
  }
}

// function to split an audio data where the In phase is in even samples
// and the Quadrature phase is in odd samples
void split_audio_into_IQ(const std::vector<float> &audio_data, std::vector<float> &i_data, std::vector<float> &q_data) {
	i_data.resize(audio_data.size()/2, 0.0);
  q_data.resize(audio_data.size()/2, 0.0);
  unsigned int in_i = 0;
  unsigned int quad_i = 0;
  for (int i=0; i<(int)audio_data.size(); i++) {
		if (i%2 == 0)
			i_data[in_i++] = audio_data[i];
		else
			q_data[quad_i++] = audio_data[i];
	}
}





// Takes the data, filter coefficients, and block size, and outputs the filtered data
void IQ_filter_block_processing(const std::vector<float> &i_data, const std::vector<float> &q_data, const std::vector<float> &h, int block_size, std::vector<float> &filtered_data, unsigned int D) {
	filtered_data.clear(); filtered_data.resize(static_cast<int>(i_data.size()/D), 0.0);
	std::vector<float> filter_state(h.size()-1, 0.0);
	std::vector<float> i_filter_block, q_filter_block;
	std::vector<float> i_block, q_block;
  std::vector<float> fm_demod;
  std::vector<float> prev_IQ;
  prev_IQ.resize(2);
  prev_IQ[0] = 0.0;
  prev_IQ[1] = 0.0;

	unsigned int position = 0;

	while (true) {
		i_block = std::vector<float>(i_data.begin() + position, i_data.begin() + position + block_size);
    q_block = std::vector<float>(q_data.begin() + position, q_data.begin() + position + block_size);

		// TODO Change this to resampling
    // Filter i samples
		convolveFIR_DS(i_filter_block, i_block, h, filter_state, D);
    // Filter q samples
    convolveFIR_DS(q_filter_block, q_block, h, filter_state, D);

    fmDemod(fm_demod, i_filter_block, q_filter_block, prev_IQ);
		// Assign filtered section to output filtered vector
		for (unsigned int i=0; i < static_cast<unsigned int>(block_size/D); i++) {
			filtered_data.at((position/D) + i) = fm_demod.at(i);
		}

		position += block_size;
    if(position > i_data.size() - block_size){
      break;
    }
	}
  //writeBinData("../data/test_data.bin", filter_state);
}


// Takes the data, filter coefficients, and block size, and outputs the filtered data
void filter_block_processing(const std::vector<float> &in_data, const std::vector<float> &h, int block_size, std::vector<float> &filtered_data, unsigned int D, unsigned int U) {

	filtered_data.clear(); filtered_data.resize(static_cast<int>(in_data.size()/D), 0.0);
	std::vector<float> filter_state(h.size()-1, 0.0);
	std::vector<float> filter_block;
	std::vector<float> data_block;

	unsigned int position = 0;

	while (true) {
		data_block = std::vector<float>(in_data.begin() + position, in_data.begin() + position+block_size);
		// TODO Change this to resampling
		convolveFIR_DS(filter_block, data_block, h, filter_state, D);
		/* Assign filtered section to output filtered vector */
		for (unsigned int i=0;i<block_size/D;i++) {
			filtered_data.at(static_cast<int>(position/D) + i) = filter_block.at(i);
		}

		position += block_size;
		if (position > (in_data.size()-block_size)){
			break;
		}
  }
}


void Pll(std::vector<float> &pllOut, const std::vector<float> &pllIn, const float freq, const float Fs, std::vector<float> &state, const float ncoScale, const float phaseAdjust, const float normBandwidth){
  float Cp = 2.666;
  float Ci = 3.555;

  float Kp = (normBandwidth)*Cp;
  float Ki = (normBandwidth*normBandwidth)*Ci;

  pllOut.clear();
  pllOut.resize(pllIn.size()+1, 0.0);

  float integrator = state[0];
	float phaseEst = state[1];
	float feedbackI = state[2];
	float feedbackQ = state[3];
  pllOut[0] = state[4];
	float trigOffset = state[5];

  float errorI, errorQ, errorD, trigArg;
  for(unsigned int i = 0; i < pllIn.size(); i++){
    errorI = pllIn[i] * (+feedbackI);
    errorQ = pllIn[i] * (-feedbackQ);

    errorD = atan2(errorQ, errorI);

    integrator = integrator + Ki*errorD;

    phaseEst = phaseEst + Kp*errorD + integrator;

    trigOffset = trigOffset + 1;
    trigArg = 2*PI*(freq/Fs)*(trigOffset) + phaseEst;
    feedbackI = cos(trigArg);
    feedbackQ = sin(trigArg);
    pllOut[i+1] = cos(trigArg*ncoScale + phaseAdjust);
  }

  state[0] = integrator;
  state[1] = phaseEst;
  state[2] = feedbackI;
  state[3] = feedbackQ;
  state[4] = pllOut[pllIn.size()];
  state[5] = trigOffset;

}


void impulseResponseRootRaisedCosine(std::vector<float> &impulseResponseRRC, const float Fs, const int N_taps){
  /*
  Root raised cosine (RRC) filter

  Fs  		sampling rate at the output of the resampler in the RDS path
        sampling rate must be an integer multipler of 2375
        this integer multiple is the number of samples per symbol

  N_taps  	number of filter taps
  */

  impulseResponseRRC.clear();
  impulseResponseRRC.resize(N_taps, 0.0);
  float T_symbol = 1.0/2375.0;
  float beta = 0.90;
  float t;
  for(int k = 0; k < N_taps; k++){
    t = static_cast<float>(k-N_taps/2)/Fs;
    if(t == 0.0){
      impulseResponseRRC[k] = 1.0 + beta*((4/PI)-1);
    }
    else if((t == -T_symbol/(4*beta)) || (t == T_symbol/(4*beta))){
      impulseResponseRRC[k] = (beta/sqrt(2))*(((1+2/PI)*(sin(PI/(4*beta)))) + ((1-2/PI)*(cos(PI/(4*beta)))));
    }
    else{
      impulseResponseRRC[k] = (sin(PI*t*(1-beta)/T_symbol) + 4*beta*(t/T_symbol)*cos(PI*t*(1+beta)/T_symbol))/(PI*t*(1-(4*beta*t/T_symbol)*(4*beta*t/T_symbol))/T_symbol);
    }
  }
}


void my_squaring(std::vector<float> &squared, const std::vector<float> &signal){
  squared.clear();
  squared.resize(signal.size(), 0.0);
  for(unsigned int i = 0; i < signal.size(); i++){
    squared[i] = signal[i]*signal[i];
  }
}


void my_CDR(std::vector<float> &data_array, const std::vector<float> &wave, const int SPS){
  data_array.clear();
  data_array.resize(static_cast<int>(wave.size()/SPS), 0.0);
  std::vector<float> first_slice = std::vector<float>(wave.begin(), wave.begin() + SPS*2-2);
  int max = std::min_element(first_slice.begin(), first_slice.end()) + static_cast<int>(SPS/2) - first_slice.begin();
  int offset = max % SPS;
  for(int i = 0; i < static_cast<int>(wave.size()/SPS); i++){
    if(wave[(i*SPS) + offset] > 0){
      data_array[i] = 1.0;
    }
  }
}




