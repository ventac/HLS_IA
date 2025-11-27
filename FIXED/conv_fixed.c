/**
 * @file conv_fixed.c
 * @brief Fixed-point implementation of convolutional layers for LeNet-5 CNN
 *
 * This file contains the fixed-point (16.16 format) implementation of two 
 * convolutional layers used in the LeNet-5 CNN architecture.
 */

#include "lenet_cnn_fixed.h"
#include "fixed_point.h"

/// @brief First convolution layer using fixed-point arithmetic
/// @param input Input image array of size [1][28][28]
/// @param kernel Convolution filters array of size [20][1][5][5]
/// @param bias Bias terms array of size [20]
/// @param output Output feature maps array of size [20][24][24]
void Conv1_28x28x1_5x5x20_1_0_fixed(
    fixed16_16_t input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    fixed16_16_t kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    fixed16_16_t bias[CONV1_NBOUTPUT],
    fixed16_16_t output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH])
{
    for (int f = 0; f < CONV1_NBOUTPUT; f++) // for each filter
    {
        for (int y = 0; y < CONV1_HEIGHT; y++)
        {
            for (int x = 0; x < CONV1_WIDTH; x++)
            {
                // Use 64-bit accumulator to reduce rounding error from per-multiply shifting
                int64_t acc = 0;

                for (int c = 0; c < IMG_DEPTH; c++)
                {
                    for (int ky = 0; ky < CONV1_DIM; ky++)
                    {
                        for (int kx = 0; kx < CONV1_DIM; kx++)
                        {
                            acc += (int64_t)input[c][y + ky][x + kx] * (int64_t)kernel[f][c][ky][kx];
                        }
                    }
                }

                // Shift once with rounding to convert from Q32 (product) -> Q16 fixed format
                int64_t round = (int64_t)1 << (FIXED_POINT_SHIFT - 1);
                if (acc >= 0) acc += round; else acc -= round;
                int64_t acc_shifted = acc >> FIXED_POINT_SHIFT;
                // Saturate
                if (acc_shifted > INT32_MAX) acc_shifted = INT32_MAX;
                if (acc_shifted < INT32_MIN) acc_shifted = INT32_MIN;

                fixed16_16_t sum = (fixed16_16_t)acc_shifted;
                sum = fixed_add(sum, bias[f]);
                // ReLU activation
                output[f][y][x] = fixed_relu(sum);
            }
        }
    }
}

/// @brief Second convolution layer using fixed-point arithmetic
/// @param input Input feature maps array of size [20][12][12]
/// @param kernel Convolution filters array of size [40][20][5][5]
/// @param bias Bias terms array of size [40]
/// @param output Output feature maps array of size [40][8][8]
void Conv2_12x12x20_5x5x40_1_0_fixed(
    fixed16_16_t input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],
    fixed16_16_t kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    fixed16_16_t bias[CONV2_NBOUTPUT],
    fixed16_16_t output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH])
{
    for (int f = 0; f < CONV2_NBOUTPUT; f++) // for each filter
    {
        for (int y = 0; y < CONV2_HEIGHT; y++)
        {
            for (int x = 0; x < CONV2_WIDTH; x++)
            {
                // Use 64-bit accumulator to reduce rounding error
                int64_t acc = 0;
                for (int c = 0; c < POOL1_NBOUTPUT; c++)
                {
                    for (int ky = 0; ky < CONV2_DIM; ky++)
                    {
                        for (int kx = 0; kx < CONV2_DIM; kx++)
                        {
                            acc += (int64_t)input[c][y + ky][x + kx] * (int64_t)kernel[f][c][ky][kx];
                        }
                    }
                }

                int64_t round = (int64_t)1 << (FIXED_POINT_SHIFT - 1);
                if (acc >= 0) acc += round; else acc -= round;
                int64_t acc_shifted = acc >> FIXED_POINT_SHIFT;
                if (acc_shifted > INT32_MAX) acc_shifted = INT32_MAX;
                if (acc_shifted < INT32_MIN) acc_shifted = INT32_MIN;

                fixed16_16_t sum = (fixed16_16_t)acc_shifted;
                sum = fixed_add(sum, bias[f]);
                // ReLU activation
                output[f][y][x] = fixed_relu(sum);
            }
        }
    }
}
