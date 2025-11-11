// lenet_cnn_fixed.h
#ifndef LENET_CNN_FIXED_H
#define LENET_CNN_FIXED_H


#include "lenet_cnn_float.h"  // for dimension constants
#include "fixed_point.h"

void Conv1_28x28x1_5x5x20_1_0_fixed(
    fixed16_16_t input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    fixed16_16_t kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    fixed16_16_t bias[CONV1_NBOUTPUT],
    fixed16_16_t output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH]);

void Pool1_24x24x20_2x2x20_2_0_fixed(
    fixed16_16_t input[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
    fixed16_16_t output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]);

void Conv2_12x12x20_5x5x40_1_0_fixed(
    fixed16_16_t input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],
    fixed16_16_t kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    fixed16_16_t bias[CONV2_NBOUTPUT],
    fixed16_16_t output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]);

void Pool2_8x8x40_2x2x40_2_0_fixed(
    fixed16_16_t input[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
    fixed16_16_t output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]);

void Fc1_40_400_fixed(
    fixed16_16_t input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    fixed16_16_t kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    fixed16_16_t bias[FC1_NBOUTPUT],
    fixed16_16_t output[FC1_NBOUTPUT]);

void Fc2_400_10_fixed(
    fixed16_16_t input[FC1_NBOUTPUT],
    fixed16_16_t kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
    fixed16_16_t bias[FC2_NBOUTPUT],
    fixed16_16_t output[FC2_NBOUTPUT]);

void Softmax_fixed(fixed16_16_t input[FC2_NBOUTPUT], fixed16_16_t output[FC2_NBOUTPUT]);

#endif