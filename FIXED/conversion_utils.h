#ifndef CONVERSION_UTILS_H
#define CONVERSION_UTILS_H

#include "lenet_cnn_float.h"

void convert_input_to_fixed(float input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH], short output[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH]);
void convert_conv1_kernel_to_fixed(float input[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM], short output[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM]);
void convert_conv1_bias_to_fixed(float input[CONV1_NBOUTPUT], short output[CONV1_NBOUTPUT]);
void convert_conv2_kernel_to_fixed(float input[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM], short output[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM]);
void convert_conv2_bias_to_fixed(float input[CONV2_NBOUTPUT], short output[CONV2_NBOUTPUT]);
void convert_fc1_kernel_to_fixed(float input[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], short output[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]);
void convert_fc1_bias_to_fixed(float input[FC1_NBOUTPUT], short output[FC1_NBOUTPUT]);
void convert_fc2_kernel_to_fixed(float input[FC2_NBOUTPUT][FC1_NBOUTPUT], short output[FC2_NBOUTPUT][FC1_NBOUTPUT]);
void convert_fc2_bias_to_fixed(float input[FC2_NBOUTPUT], short output[FC2_NBOUTPUT]);

#endif