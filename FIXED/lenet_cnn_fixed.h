// lenet_cnn_fixed.h
#ifndef LENET_CNN_FIXED_H
#define LENET_CNN_FIXED_H

#include "lenet_cnn_float.h"  // for dimension constants

void Conv1_28x28x1_5x5x20_1_0_fixed(
    short input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    short kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    short bias[CONV1_NBOUTPUT],
    short output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH]);

void Pool1_24x24x20_2x2x20_2_0_fixed(
    short input[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
    short output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]);

void Conv2_12x12x20_5x5x40_1_0_fixed(
    short input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],
    short kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    short bias[CONV2_NBOUTPUT],
    short output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]);

void Pool2_8x8x40_2x2x40_2_0_fixed(
    short input[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
    short output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]);

void Fc1_40_400_fixed(
    short input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    short kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    short bias[FC1_NBOUTPUT],
    short output[FC1_NBOUTPUT]);

void Fc2_400_10_fixed(
    short input[FC1_NBOUTPUT],
    short kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
    short bias[FC2_NBOUTPUT],
    short output[FC2_NBOUTPUT]);

void Softmax_fixed(short input[FC2_NBOUTPUT], float output[FC2_NBOUTPUT]);

#endif