/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Copyright by Nicola Nicolici
Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "filter.h"
#include "fourier.h"
#include "genfunc.h"
#include "iofunc.h"
#include "logfunc.h"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

//Values to terminate looping in threads, increase for more interations
#define TOTAL_BLOCKS 10000000
#define QUEUE_ELEMS 5

void my_RF_Frontend(std::queue<std::vector<float>> &my_IF_queue, \
	std::mutex& my_IF_mutex, \
	std::condition_variable& my_queue_cvar, \
	unsigned int mode)
	{
		//INITIALIZE RF REQUIRED VALUES------------------------------------------------------------
		int rf_Fs, rf_decim, D_fact, U_fact;
		if(mode==0){
			rf_Fs = 2.4e6;
			rf_decim = 10;
			U_fact = 1;
			D_fact = 5;
		}
		else if(mode == 1){
			rf_Fs = 960e3;
			rf_decim = 4;
			U_fact = 1;
			D_fact = 5;
		}
		else if(mode == 2){
			rf_Fs = 2.4e6;
			rf_decim = 10;
			U_fact = 147;
			D_fact = 800;
		}
		else{
			rf_Fs = 1.44e6;
			rf_decim = 4;
			U_fact = 49;
			D_fact = 400;
		}

		int rf_taps = 151;
		int rf_Fc = 100e3;
/*
		// Split the sample data into i and q samples
		std::vector<float> i_samples, q_samples;
		split_audio_into_IQ(bin_data, i_samples, q_samples);
*/
		std::vector<float> rf_coeff;
		impulseResponseLPF(rf_Fs, rf_Fc, rf_taps, rf_coeff);

		std::vector<float> i_state(rf_coeff.size()-1, 0.0); // i state saving
		std::vector<float> q_state(rf_coeff.size()-1, 0.0);	// q state saving
		std::vector<float> i_block, q_block;	// i and q pre filtering blocks
		std::vector<float> i_filter_block, q_filter_block;	// i and q post filtering blocks
		std::vector<float> fm_demod_block; // block of fmdemod data
		std::vector<float> prev_IQ(2, 0.0); // Stores the previous I and Q values for fmdemod;
		prev_IQ[0] = 0.0;
		prev_IQ[1] = 0.0;

		int audio_decim = static_cast<int>(D_fact/U_fact);
		int block_size = 1024 * rf_decim * audio_decim;

		std::vector<float> bin_data;
		//-----------------------------------------------------------------------------------------------------------

		//Counter for Thread termination
		int generated_IF_blocks = 0;

		//Limit for thread termination
		while(generated_IF_blocks < TOTAL_BLOCKS) {

			// Read raw files
			readStdinBlockData(block_size, 4, bin_data);

			// Split the sample data into i and q samples
			split_audio_into_IQ(bin_data, i_block, q_block);

			// Filter i samples
			convolveFIR_DS(i_filter_block, i_block, rf_coeff, i_state, rf_decim);
			// Filter q samples
			convolveFIR_DS(q_filter_block, q_block, rf_coeff, q_state, rf_decim);

			// Perform fm demodulation
			fmDemod(fm_demod_block, i_filter_block, q_filter_block, prev_IQ);
			//Initialize Mutex
			std::unique_lock<std::mutex> my_lock(my_IF_mutex);
			//Add block to IF queue
			while(my_IF_queue.size() >= QUEUE_ELEMS) {
				my_queue_cvar.wait(my_lock);
			}

			my_IF_queue.push(fm_demod_block);
			my_queue_cvar.notify_all();
			my_lock.unlock();
			//Increment Position in data
			generated_IF_blocks++;
		}

	}

void my_Mono(std::queue<std::vector<float>> &my_IF_queue, \
std::queue<std::vector<float>> &my_mono_queue, \
std::mutex& my_IF_mutex, \
std::condition_variable& my_queue_cvar, \
std::mutex& my_mono_mutex, \
std::condition_variable& my_mono_queue_cvar, \
std::vector<float> &current_block, \
unsigned int mode)
{
	// ------------------------------ Mode Selection -----------------------------
	int if_Fs, U_fact, D_fact;
	if(mode==0){
		U_fact = 1;
		D_fact = 5;
		if_Fs = 2.4e5*U_fact;
		//audio_Fs = 48e3;

	}
	else if(mode == 1){
		U_fact = 1;
		D_fact = 5;
		if_Fs = 240e3*U_fact; 	// = 240 Ksamples/sec
		//audio_Fs = 48e3;
	}
	else if(mode == 2){
		U_fact = 147;
		D_fact = 800;
		if_Fs = 2.4e5*U_fact; 	// = 240 Ksamples/sec
		//audio_Fs = 44.1e3;

	}
	else{
		U_fact = 49;
		D_fact = 400;
		if_Fs = 360e3*U_fact; 	// = 360 Ksamples/sec
		//audio_Fs = 44.1e3;
	}

	// ------------------------  Mono Processing Variables -----------------------
	int audio_taps = 101 * U_fact;
	//int audio_decim = static_cast<int>(D_fact/U_fact);
	int audio_Fc = 16e3;

	// Generate the filter coefficients for mono audio
	std::vector<float> mono_coeff;
	impulseResponseLPF(if_Fs, audio_Fc, audio_taps, mono_coeff);

	// Generate the delay coefficients for syncing mono with stereo
	std::vector<float> delay_coeff(static_cast<int>(audio_taps/D_fact), 0.0);
	delay_coeff[static_cast<unsigned int>(delay_coeff.size()/2)+1] = 1.0;
	//delay_coeff[14] = 1.0;

	// Variables needed for mono processing
	std::vector<float> mono_block;
	std::vector<float> mono_state(mono_coeff.size()-1, 0.0);	// state saving for mono audio
	std::vector<float> mono_block_sync;
	std::vector<float> mono_sync_state(mono_coeff.size()-1, 0.0);

	// ------------------------  Stereo Processing Variables ---------------------
	int rf_taps = 151;
	int carr_Fc_start = 18.5e3;
	int carr_Fc_end = 19.5e3;
	int stereo_Fc_start = 22e3;
	int stereo_Fc_end = 54e3;

	// Generate filter coefficients for the carrier filter
	std::vector<float> carr_coeff;
	impulseResponseBPF(if_Fs/U_fact, carr_Fc_start, carr_Fc_end, rf_taps, carr_coeff);

	// Generate filter coefficients for the stereo audio filter
	std::vector<float> stereo_coeff;
	impulseResponseBPF(if_Fs/U_fact, stereo_Fc_start, stereo_Fc_end, rf_taps, stereo_coeff);

	// Carrier Block Processing Variables
	std::vector<float> carr_block;
	std::vector<float> carr_state(carr_coeff.size()-1, 0.0);

	// Stereo Block Processing Variables
	std::vector<float> stereo_block;
	std::vector<float> stereo_state(stereo_coeff.size()-1, 0.0);

	// Mixing blocks arrays
	std::vector<float> audio_mix;
	std::vector<float> audio_mix_out;
	std::vector<float> audio_mix_state(audio_taps-1, 0.0);
	
	// ------------------------  Threading Process Begins ---------------------
	int generated_mono_blocks = 0;
	while(generated_mono_blocks < TOTAL_BLOCKS) {

		// ------------------------ Reading data from Queue ---------------------
		std::unique_lock<std::mutex> my_lock(my_IF_mutex);

		//Wait for data
		while(my_IF_queue.empty()) {
			my_queue_cvar.wait(my_lock);
		}

		//Retireve Data
		current_block = my_IF_queue.front();
		my_IF_queue.pop();

		//Notify that a queue item has been removed
		my_queue_cvar.notify_one();

		my_lock.unlock();
		//Notify that there is a block ready
		//current_block_cvar.notify_one();

		// ------------------------  Block read, begin processing ---------------------
		//DO MONO Processing using current_block as input data. DO NOT ALTER it, it is shared data Read only************************

		// -------------------------  Mono Processing ------------------------------
		// Convolution for extracting mono audio channel
		convolveFIR_Resample(mono_block, current_block, mono_coeff, mono_state, U_fact, D_fact);
		// Allpass for syncing to stereo
		convolveFIR_DS(mono_block_sync, mono_block, delay_coeff, mono_sync_state, 1);


		//ADD MONO BLOCK TO my_mono_queue like in RF_frontend and notify that there is a mono block ready for mixing in stereo************
		
		//Initialize Mutex
		std::unique_lock<std::mutex> my_mono_lock(my_mono_mutex);
		//Add block to IF queue
		while(my_mono_queue.size() >= QUEUE_ELEMS) {
			my_mono_queue_cvar.wait(my_mono_lock);
		}

		my_mono_queue.push(mono_block_sync);
		my_mono_queue_cvar.notify_one();
		my_mono_lock.unlock();
		//Increment Position in data
		

		generated_mono_blocks++;
	}

}	

void my_Stereo(std::queue<std::vector<float>> &my_IF_queue, \
std::queue<std::vector<float>> &my_mono_queue, \
std::mutex& my_IF_mutex, \
std::mutex& my_mono_mutex, \
std::condition_variable& my_queue_cvar, \
std::condition_variable& my_mono_queue_cvar, \
std::vector<float> &current_block, \
unsigned int mode)
{
	// ------------------------------ Mode Selection -----------------------------
	int if_Fs, U_fact, D_fact;
	if(mode==0){
		U_fact = 1;
		D_fact = 5;
		if_Fs = 2.4e5*U_fact;
		//audio_Fs = 48e3;

	}
	else if(mode == 1){
		U_fact = 1;
		D_fact = 5;
		if_Fs = 240e3*U_fact; 	// = 240 Ksamples/sec
		//audio_Fs = 48e3;
	}
	else if(mode == 2){
		U_fact = 147;
		D_fact = 800;
		if_Fs = 2.4e5*U_fact; 	// = 240 Ksamples/sec
		//audio_Fs = 44.1e3;

	}
	else{
		U_fact = 49;
		D_fact = 400;
		if_Fs = 360e3*U_fact; 	// = 360 Ksamples/sec
		//audio_Fs = 44.1e3;
	}

	// ------------------------  Mono Processing Variables -----------------------
	int audio_taps = 101 * U_fact;
	//int audio_decim = static_cast<int>(D_fact/U_fact);
	int audio_Fc = 16e3;

	// Generate the filter coefficients for mono audio
	std::vector<float> mono_coeff;
	impulseResponseLPF(if_Fs, audio_Fc, audio_taps, mono_coeff);

	// ------------------------  Stereo Processing Variables ---------------------
	int rf_taps = 151;
	int carr_Fc_start = 18.5e3;
	int carr_Fc_end = 19.5e3;
	int stereo_Fc_start = 22e3;
	int stereo_Fc_end = 54e3;

	// Generate filter coefficients for the carrier filter
	std::vector<float> carr_coeff;
	impulseResponseBPF(if_Fs/U_fact, carr_Fc_start, carr_Fc_end, rf_taps, carr_coeff);

	// Generate filter coefficients for the stereo audio filter
	std::vector<float> stereo_coeff;
	impulseResponseBPF(if_Fs/U_fact, stereo_Fc_start, stereo_Fc_end, rf_taps, stereo_coeff);

	// Carrier Block Processing Variables
	std::vector<float> carr_block;
	std::vector<float> carr_state(carr_coeff.size()-1, 0.0);

	// Pll Variables
	std::vector<float> pll_block;
	float targetfreq = 19e3;
	float ncoScale = 2.0;
	float phaseAdjust = 0.0;
	float normBandwidth = 0.01;
	std::vector<float> state_Pll(6, 0.0);
	state_Pll[2] = 1.0;
	state_Pll[5] = 1.0;

	// Stereo Block Processing Variables
	std::vector<float> stereo_block;
	std::vector<float> stereo_state(stereo_coeff.size()-1, 0.0);

	// Mixing blocks arrays
	std::vector<float> audio_mix;
	std::vector<float> audio_mix_out;
	std::vector<float> audio_mix_state(audio_taps-1, 0.0);


	std::vector<float> mono_block;
	std::vector<float> left_block;
	std::vector<float> right_block;
	
	
	
	// ------------------------  Threading Process Begins ---------------------
	int generated_mono_blocks = 0;
	while(generated_mono_blocks < TOTAL_BLOCKS) {
		
		// ------------------------ Reading data from Queue ---------------------
		std::unique_lock<std::mutex> my_lock(my_IF_mutex);

		//Wait for data
		while(my_IF_queue.empty()) {
			my_queue_cvar.wait(my_lock);
		}

		my_lock.unlock();
		

		// Carrier Recovery
		convolveFIR_DS(carr_block, current_block, carr_coeff, carr_state, 1);
		// Synchronize with f_subcarrier
		Pll(pll_block, carr_block, targetfreq, if_Fs/U_fact, state_Pll, ncoScale, phaseAdjust, normBandwidth);

		// Stereo Channel Extraction
		convolveFIR_DS(stereo_block, current_block, stereo_coeff, stereo_state, 1);

		// Mixing Stereo and Carrier
		audio_mix.clear();
		audio_mix.resize(stereo_block.size(), 0.0);
		for(unsigned int i = 1; i < pll_block.size(); i++){
			audio_mix[i-1] = 2*stereo_block[i-1]*pll_block[i];
		}
		convolveFIR_Resample(audio_mix_out, audio_mix, mono_coeff, audio_mix_state, U_fact, D_fact);
		left_block.clear(); right_block.clear();
		left_block.resize(audio_mix_out.size(), 0.0);
		right_block.resize(audio_mix_out.size(), 0.0);


		// ------------------------ Reading data from Queue ---------------------
		std::unique_lock<std::mutex> my_mono_lock(my_mono_mutex);

		//Wait for data
		while(my_mono_queue.empty()) {
			my_mono_queue_cvar.wait(my_mono_lock);
		}

		//Retireve Data
		mono_block = my_mono_queue.front();
		my_mono_queue.pop();

		//Notify that a queue item has been removed
		my_mono_queue_cvar.notify_one();

		my_mono_lock.unlock();
		//Notify that there is a block ready
		//current_block_cvar.notify_one();

		// ------------------------  Block read, begin processing ---------------------
		//DO MONO Processing using current_block as input data. DO NOT ALTER it, it is shared data Read only************************

		// ------------------------  Stereo Processing  ----------------------------

		for (unsigned int k=0; k<left_block.size(); k++){
			left_block[k] = mono_block[k] + audio_mix_out[k];
			right_block[k] = mono_block[k] - audio_mix_out[k];
		}

		// Write out to aplay
		std::vector<short int> audio_out(left_block.size() + right_block.size(), 0);
		for (unsigned int k=0; k<left_block.size(); k++) {
			if (std::isnan(left_block[k])) {
				audio_out[2*k] = 0;
			}
			else {
				audio_out[2*k] = static_cast<short int>(left_block[k] * 16384);
			}
		}
		for (unsigned int k=0; k<right_block.size(); k++) {
			if (std::isnan(right_block[k])) {
				audio_out[2*k+1] = 0;
			}
			else {
				audio_out[2*k+1] = static_cast<short int>(right_block[k] * 16384);
			}
		}
		fwrite(&audio_out[0], sizeof(short int), audio_out.size(), stdout);

		generated_mono_blocks++;
	}

}

void my_RDS(std::queue<std::vector<float>> &my_IF_queue, \
std::mutex& my_IF_mutex, \
std::condition_variable& my_queue_cvar, \
std::vector<float> &current_block, \
unsigned int mode)
{
	int rf_decim, if_Fs, U_fact, D_fact, rds_U, rds_D, SPS;
	if(mode==0){
		rf_decim = 10;
		U_fact = 1;
		D_fact = 5;
		if_Fs = 2.4e5*U_fact;
		//audio_Fs = 48e3;
		SPS = 19;
		rds_U = 361;
		rds_D = 1920;
	}
	else if(mode == 1){
		rf_decim = 4;
		U_fact = 1;
		D_fact = 5;
		if_Fs = 240e3*U_fact; 	// = 240 Ksamples/sec
		//audio_Fs = 48e3;
	}
	else if(mode == 2){
		rf_decim = 10;
		U_fact = 147;
		D_fact = 800;
		if_Fs = 2.4e5*U_fact; 	// = 240 Ksamples/sec
		//audio_Fs = 44.1e3;
		SPS = 42;
		rds_U = 133;
		rds_D = 320;
	}
	else{
		rf_decim = 4;
		U_fact = 49;
		D_fact = 400;
		if_Fs = 360e3*U_fact; 	// = 360 Ksamples/sec
		//audio_Fs = 44.1e3;
	}
	int block_size = 1024 * rf_decim * static_cast<int>(D_fact/U_fact);

	// ========================= RDS Variables and Arrays ========================
	int rds_decim = 5; //RDS decimation rate
	int rds_taps = 151; //Number of taps for the filter
		//Create coefficents fo the filter to extract RDBS signal
	std::vector<float> extraction_coeff;
	impulseResponseBPF(if_Fs/U_fact , 54000.0, 60000.0, rds_taps, extraction_coeff);
		//Create coefficients for the filter to recover the carrier
	std::vector<float> recovery_coeff;
	impulseResponseBPF(if_Fs/U_fact, 113500.0, 114500.0, rds_taps, recovery_coeff);
		//Create coefficients for the allpass filter (delay)
	std::vector<float> rds_allpass(20, 0.0);
	rds_allpass[19] = 1;
		//Create coefficients for the lowpass demodulating filter
	std::vector<float> rds_lowpass;
	impulseResponseLPF(if_Fs, 3000, rds_taps, rds_lowpass);
	std::vector<float> rds_lowpass_resample;
	impulseResponseLPF(if_Fs, 3000, rds_taps*rds_U, rds_lowpass_resample);
		//Create coefficients for the root-raised cosine filter
	std::vector<float> RRC_coeff;
	impulseResponseRootRaisedCosine(RRC_coeff, 45125.0, 19);
		//Vectors for RDS data
	std::vector<float> rds_extract_block;
	std::vector<float> rds_squared_block;
	std::vector<float> rds_carr_block;
	std::vector<float> rds_carr_pll_block;
	std::vector<float> rds_extract_delay_block;
	//std::vector<float> rds_mixed_block(static_cast<unsigned int>((block_size/2)/rf_decim), 0.0);
	std::vector<float> rds_mixed_block;
	std::vector<float> rds_mixed_block_1;
	std::vector<float> rds_mixed_block_2;
	std::vector<float> rds_RRC_block;
	std::vector<float> rds_cdr_block;
		//State saving vectors for RDS
	std::vector<float> state_rds_extract(rds_taps-1, 0.0);
	std::vector<float> state_rds_allpass(rds_taps-1, 0.0);
	std::vector<float> state_rds_carr(rds_taps-1, 0.0);
	std::vector<float> state_rds_pll(6, 0.0);
	state_rds_pll[2] = 1.0;
	state_rds_pll[4] = 1.0;
	std::vector<float> state_rds_lowpass(rds_taps-1, 0.0);
	std::vector<float> state_rds_lowpass_re(rds_taps-1, 0.0);
	std::vector<float> state_rds_RRC(19-1, 0.0);


	int generated_mono_blocks = 0;
	while((generated_mono_blocks < TOTAL_BLOCKS) && (mode == 0 || mode == 2)) {

		// ------------------------ Reading data from Queue ---------------------
		std::unique_lock<std::mutex> my_lock(my_IF_mutex);
		//Wait for data
		while(my_IF_queue.empty()) {
			my_queue_cvar.wait(my_lock);
		}
		my_lock.unlock();

		// ======================= RDS Processing ======================
		// RDS channel extraction
		convolveFIR_DS(rds_extract_block, current_block, extraction_coeff, state_rds_extract, 1);
		//RDS carrier recovery
			//Squaring
		my_squaring(rds_squared_block, rds_extract_block);
			//Second BPF
		convolveFIR_DS(rds_carr_block, rds_squared_block, recovery_coeff, state_rds_carr, 1);
		Pll(rds_carr_pll_block, rds_carr_block, 114000, if_Fs/U_fact, state_rds_pll, 0.5, 0.0, 0.01);

		//All pass filter delay
		convolveFIR_DS(rds_extract_delay_block, rds_extract_block, rds_allpass, state_rds_allpass, 1);
		//rds_extract_block.clear();
		//RDS mixing
		rds_mixed_block.clear();
		rds_mixed_block.resize(rds_extract_delay_block.size(), 0.0);
		for (unsigned int k=0; k<rds_extract_delay_block.size(); k++) {
			rds_mixed_block[k] = 2*rds_extract_delay_block[k]*rds_carr_pll_block[k];
		}
		//RDS demodulation
		convolveFIR_DS(rds_mixed_block_1, rds_mixed_block, rds_lowpass, state_rds_lowpass, 1);
		convolveFIR_Resample(rds_mixed_block_2, rds_mixed_block_1, rds_lowpass_resample, state_rds_lowpass_re, rds_U, rds_D);
		//rds_mixed_block_1.clear();
		convolveFIR_DS(rds_RRC_block, rds_mixed_block_2, RRC_coeff, state_rds_RRC, 1);
		//rds_mixed_block_2.clear();
		//Clock and data recovery
		my_CDR(rds_cdr_block, rds_RRC_block, SPS);
		
		// Uncomment the print to show the bitstream in the terminal
		for(unsigned int i = 0; i < rds_cdr_block.size(); i++){
			//std::cerr << rds_cdr_block[i];
		}

	}

}


int main(int argc, char* argv[])
{
	// ----------------- Mode Switching ------------------
	unsigned int mode;
	if (argc < 2)
		mode = 0; //default mode
	else if (argc == 2) {
		mode = atoi(argv[1]);
		if (mode > 3) {
			std::cerr << "Mode cannot be greater than 3 \n";
			exit(1);
		}
	}
	else {
		std::cerr << "Program only accepts one argument: an integer representing the mode \n";
		exit(1);
	}

	// ======================== Multithreading Variables and Arrays ======================
	std::queue<std::vector<float>> my_IF_queue;
	std::queue<std::vector<float>> my_mono_queue;
	std::vector<float> current_block;
	std::mutex my_IF_mutex;
	std::mutex my_mono_mutex;
	std::condition_variable my_queue_cvar;
	std::condition_variable my_mono_queue_cvar;


	//Initializing threads
	std::thread thread_RF = std::thread(my_RF_Frontend, std::ref(my_IF_queue), \
		std::ref(my_IF_mutex), std::ref(my_queue_cvar), mode);

	std::thread thread_Mono = std::thread(my_Mono, \ 
		std::ref(my_IF_queue), std::ref(my_mono_queue), \
		std::ref(my_IF_mutex), std::ref(my_queue_cvar), \
		std::ref(my_mono_mutex), std::ref(my_mono_queue_cvar), std::ref(current_block), \
		mode);
	
	std::thread thread_Stereo = std::thread(my_Stereo, std::ref(my_IF_queue), \ 
		std::ref(my_mono_queue), std::ref(my_IF_mutex),\
		std::ref(my_mono_mutex), std::ref(my_queue_cvar), \
		std::ref(my_mono_queue_cvar), std::ref(current_block), \
		mode);
		
	std::thread thread_RDS = std::thread(my_RDS, std::ref(my_IF_queue), \ 
		std::ref(my_IF_mutex),std::ref(my_queue_cvar), \ 
		std::ref(current_block), mode);

	//Joining threads
	thread_RF.join();
	//thread_Mono_Stereo.join();
	thread_Mono.join();
	thread_Stereo.join();
	thread_RDS.join();
	
	return 0;
}
