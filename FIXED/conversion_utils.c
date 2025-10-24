/**
 * @file conversion_utils.c
 * @brief Utility functions to convert arrays between float and fixed-point formats
 */

#include "lenet_cnn_float.h"
#include "fixed_point.h"

// Convert input image from float to fixed-point
void convert_input_to_fixed(
    float input_float[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    fixed16_16_t input_fixed[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH])
{
    for (int d = 0; d < IMG_DEPTH; d++) {
        for (int h = 0; h < IMG_HEIGHT; h++) {
            for (int w = 0; w < IMG_WIDTH; w++) {
                input_fixed[d][h][w] = float_to_fixed(input_float[d][h][w]);
            }
        }
    }
}

// Convert Conv1 kernel from float to fixed-point
void convert_conv1_kernel_to_fixed(
    float kernel_float[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    fixed16_16_t kernel_fixed[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM])
{
    for (int f = 0; f < CONV1_NBOUTPUT; f++) {
        for (int d = 0; d < IMG_DEPTH; d++) {
            for (int h = 0; h < CONV1_DIM; h++) {
                for (int w = 0; w < CONV1_DIM; w++) {
                    kernel_fixed[f][d][h][w] = float_to_fixed(kernel_float[f][d][h][w]);
                }
            }
        }
    }
}

// Convert Conv1 bias from float to fixed-point
void convert_conv1_bias_to_fixed(
    float bias_float[CONV1_NBOUTPUT],
    fixed16_16_t bias_fixed[CONV1_NBOUTPUT])
{
    for (int i = 0; i < CONV1_NBOUTPUT; i++) {
        bias_fixed[i] = float_to_fixed(bias_float[i]);
    }
}

// Convert Conv2 kernel from float to fixed-point
void convert_conv2_kernel_to_fixed(
    float kernel_float[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    fixed16_16_t kernel_fixed[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM])
{
    for (int f = 0; f < CONV2_NBOUTPUT; f++) {
        for (int d = 0; d < POOL1_NBOUTPUT; d++) {
            for (int h = 0; h < CONV2_DIM; h++) {
                for (int w = 0; w < CONV2_DIM; w++) {
                    kernel_fixed[f][d][h][w] = float_to_fixed(kernel_float[f][d][h][w]);
                }
            }
        }
    }
}

// Convert Conv2 bias from float to fixed-point
void convert_conv2_bias_to_fixed(
    float bias_float[CONV2_NBOUTPUT],
    fixed16_16_t bias_fixed[CONV2_NBOUTPUT])
{
    for (int i = 0; i < CONV2_NBOUTPUT; i++) {
        bias_fixed[i] = float_to_fixed(bias_float[i]);
    }
}

// Convert FC1 kernel from float to fixed-point
void convert_fc1_kernel_to_fixed(
    float kernel_float[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    fixed16_16_t kernel_fixed[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH])
{
    for (int n = 0; n < FC1_NBOUTPUT; n++) {
        for (int c = 0; c < POOL2_NBOUTPUT; c++) {
            for (int h = 0; h < POOL2_HEIGHT; h++) {
                for (int w = 0; w < POOL2_WIDTH; w++) {
                    kernel_fixed[n][c][h][w] = float_to_fixed(kernel_float[n][c][h][w]);
                }
            }
        }
    }
}

// Convert FC1 bias from float to fixed-point
void convert_fc1_bias_to_fixed(
    float bias_float[FC1_NBOUTPUT],
    fixed16_16_t bias_fixed[FC1_NBOUTPUT])
{
    for (int i = 0; i < FC1_NBOUTPUT; i++) {
        bias_fixed[i] = float_to_fixed(bias_float[i]);
    }
}

// Convert FC2 kernel from float to fixed-point
void convert_fc2_kernel_to_fixed(
    float kernel_float[FC2_NBOUTPUT][FC1_NBOUTPUT],
    fixed16_16_t kernel_fixed[FC2_NBOUTPUT][FC1_NBOUTPUT])
{
    for (int n = 0; n < FC2_NBOUTPUT; n++) {
        for (int i = 0; i < FC1_NBOUTPUT; i++) {
            kernel_fixed[n][i] = float_to_fixed(kernel_float[n][i]);
        }
    }
}

// Convert FC2 bias from float to fixed-point
void convert_fc2_bias_to_fixed(
    float bias_float[FC2_NBOUTPUT],
    fixed16_16_t bias_fixed[FC2_NBOUTPUT])
{
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        bias_fixed[i] = float_to_fixed(bias_float[i]);
    }
}

// Convert output from fixed-point to float
void convert_output_to_float(
    fixed16_16_t output_fixed[FC2_NBOUTPUT],
    float output_float[FC2_NBOUTPUT])
{
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        output_float[i] = fixed_to_float(output_fixed[i]);
    }
}
