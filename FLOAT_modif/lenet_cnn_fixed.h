/**
  ******************************************************************************
  * @file    lenet_cnn_fixed.h
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @version V1.0
  * @date    04 february 2019
  * @brief   Plain C code for the implementation of Convolutional Neural Networks on FPGA
  * @brief   Designed to support Vivado HLS synthesis
  */

#ifndef LENT_CNN_FIXED_H
#define LENT_CNN_FIXED_H

#include <stdint.h>
#include "lenet_cnn_float.h"

void Fc1_40_400_fixed(
    const int32_t input[restrict POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 
    const int32_t kernel[restrict FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 
    const int32_t bias[restrict FC1_NBOUTPUT], 
    int32_t output[restrict FC1_NBOUTPUT]
);

void Fc2_400_10_fixed(
    const int32_t input[restrict FC1_NBOUTPUT], 
    const int32_t kernel[restrict FC2_NBOUTPUT][FC1_NBOUTPUT], 
    const int32_t bias[restrict FC2_NBOUTPUT], 
    int32_t output[restrict FC2_NBOUTPUT]
);


void Pool1_24x24x20_2x2x20_2_0(	float 	input[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH], 	    // IN
				                float 	output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]);		// OUT

void Conv2_12x12x20_5x5x40_1_0(	float input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH], 	            // IN
				                float kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM], 	// IN
				                float bias[CONV2_NBOUTPUT], 						                    // IN
				                float output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]); 		        // OUT

void Pool2_8x8x40_2x2x40_2_0(	float 	input[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH], 	    // IN
				                float 	output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]);		// OUT



void Softmax_fixed(int32_t vector_in[FC2_NBOUTPUT], float vector_out[FC2_NBOUTPUT]);

#endif
