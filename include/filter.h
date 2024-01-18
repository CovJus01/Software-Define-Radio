/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#ifndef DY4_FILTER_H
#define DY4_FILTER_H

// add headers as needed
#include <iostream>
#include <vector>

// declaration of a function prototypes
void impulseResponseLPF(float, float, unsigned short int, std::vector<float> &);
void impulseResponseBPF(float, float, float, unsigned short int, std::vector<float> &);
void convolveFIR_DS(std::vector<float> &, const std::vector<float> &, const std::vector<float> &, std::vector<float> &, const signed int);
void convolveFIR_Resample(std::vector<float> &, const std::vector<float> &, const std::vector<float> &, std::vector<float> &, const signed int, const signed int);
void fmDemod(std::vector<float> &, const std::vector<float> &, const std::vector<float> &, std::vector<float> &);
void split_audio_into_IQ(const std::vector<float> &, std::vector<float> &, std::vector<float> &);
void IQ_filter_block_processing(const std::vector<float> &, const std::vector<float> &, const std::vector<float> &, int, std::vector<float> &, unsigned int);
void filter_block_processing(const std::vector<float> &, const std::vector<float> &, int, std::vector<float> &, unsigned int, unsigned int);
void Pll(std::vector<float> &, const std::vector<float> &, const float, const float, std::vector<float> &, const float, const float, const float);
void impulseResponseRootRaisedCosine(std::vector<float> &, const float, const int);
void my_squaring(std::vector<float> &, const std::vector<float> &);
void my_CDR(std::vector<float> &, const std::vector<float> &, const int);



#endif // DY4_FILTER_H
