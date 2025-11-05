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

void Softmax_fixed(int32_t vector_in[FC2_NBOUTPUT], float vector_out[FC2_NBOUTPUT]);

#endif
