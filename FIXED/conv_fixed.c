/**
 * @file conv_fixed.c
 * @brief Fixed-point implementation of convolutional layers for LeNet-5 CNN
 *
 * This file contains the fixed-point (16.16 format) implementation of two 
 * convolutional layers used in the LeNet-5 CNN architecture.
 */

#include "lenet_cnn_fixed.h"
//#include "fixed_point.h"


/// @brief First convolution layer using fixed-point arithmetic
/// @param input Input image array of size [IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH]
/// @param kernel Convolution filters array of size [CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM]
/// @param bias Bias terms array of size [CONV1_NBOUTPUT]
/// @param output Output feature maps array of size [CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH]
void Conv1_28x28x1_5x5x20_1_0_fixed(
    short input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    short kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    short bias[CONV1_NBOUTPUT],
    short output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH])
{
    unsigned short f, c, y, x, ky, kx;
    int temp;
    int acc;

    for (f = 0; f < CONV1_NBOUTPUT; f++) {          // for each filter
        for (y = 0; y < CONV1_HEIGHT; y++) {
            for (x = 0; x < CONV1_WIDTH; x++) {
                acc = 0;

                for (c = 0; c < IMG_DEPTH; c++) {    // for each input channel
                    for (ky = 0; ky < CONV1_DIM; ky++) {
                        for (kx = 0; kx < CONV1_DIM; kx++) {
                            temp = kernel[f][c][ky][kx] * input[c][y + ky][x + kx];
                            acc += temp;
                        }
                    }
                }

                // Fixed-point scaling and adding bias
                acc = (acc >> FIXED_POINT) + bias[f];

                // ReLU activation
                output[f][y][x] = (short)(acc > 0 ? acc : 0);
            }
        }
    }
}

/// @brief Second convolution layer using fixed-point arithmetic
/// @param input Input feature maps array of size [POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]
/// @param kernel Convolution filters array of size [CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM]
/// @param bias Bias terms array of size [CONV2_NBOUTPUT]
/// @param output Output feature maps array of size [CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]
void Conv2_12x12x20_5x5x40_1_0_fixed(
    short input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],
    short kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    short bias[CONV2_NBOUTPUT],
    short output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH])
{
    unsigned short f, c, y, x, ky, kx;
    int temp;
    int acc;

    for (f = 0; f < CONV2_NBOUTPUT; f++) {          // for each filter
        for (y = 0; y < CONV2_HEIGHT; y++) {
            for (x = 0; x < CONV2_WIDTH; x++) {
                acc = 0;

                for (c = 0; c < POOL1_NBOUTPUT; c++) {    // for each input channel
                    for (ky = 0; ky < CONV2_DIM; ky++) {
                        for (kx = 0; kx < CONV2_DIM; kx++) {
                            temp = kernel[f][c][ky][kx] * input[c][y + ky][x + kx];
                            acc += temp;
                        }
                    }
                }

                // Fixed-point scaling and adding bias
                acc = (acc >> FIXED_POINT) + bias[f];

                // ReLU activation
                output[f][y][x] = (short)(acc > 0 ? acc : 0);
            }
        }
    }
}

