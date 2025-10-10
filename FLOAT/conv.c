/**
 * @file conv.c
 * @brief Implementation of the convolutional layers for LeNet-5 CNN in floating point
 *
 * This file contains the implementation of two convolutional layers used in the LeNet-5
 * Convolutional Neural Network architecture. The first layer (Conv1) processes the input
 * image, while the second layer (Conv2) processes the feature maps from the first pooling
 * layer. Both layers implement convolution operations with ReLU activation functions.
 *
 * Key features:
 * - Conv1: Transforms 28x28x1 input image into 24x24x20 feature maps
 * - Conv2: Transforms 12x12x20 feature maps into 8x8x40 feature maps
 * - Uses ReLU activation function (max(0,x))
 * - Implements bias addition for each feature map
 */

#include "lenet_cnn_float.h"
#include <math.h>

/// @brief First convolution layer that transforms input image (28x28x1) into feature maps (24x24x20)
/// @param input Input image array of size [1][28][28]
/// @param kernel Convolution filters array of size [20][1][5][5]
/// @param bias Bias terms array of size [20]
/// @param output Output feature maps array of size [20][24][24]
void Conv1_28x28x1_5x5x20_1_0(
    float input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    float kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    float bias[CONV1_NBOUTPUT],
    float output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH])
{
    for (int f = 0; f < CONV1_NBOUTPUT; f++) // for each filter
    {
        for (int y = 0; y < CONV1_HEIGHT; y++)
        {
            for (int x = 0; x < CONV1_WIDTH; x++)
            {
                float sum = 0.0f;

                for (int c = 0; c < IMG_DEPTH; c++)
                {
                    for (int ky = 0; ky < CONV1_DIM; ky++)
                    {
                        for (int kx = 0; kx < CONV1_DIM; kx++)
                        {
                            sum += input[c][y + ky][x + kx] * kernel[f][c][ky][kx];
                        }
                    }
                }

                sum += bias[f];
                // ReLU activation
                output[f][y][x] = (sum > 0) ? sum : 0;
            }
        }
    }
}

/// @brief Second convolution layer that transforms feature maps (12x12x20) into new feature maps (8x8x40)
/// @param input Input feature maps array of size [20][12][12]
/// @param kernel Convolution filters array of size [40][20][5][5]
/// @param bias Bias terms array of size [40]
/// @param output Output feature maps array of size [40][8][8]
void Conv2_12x12x20_5x5x40_1_0(
    float input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],
    float kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    float bias[CONV2_NBOUTPUT],
    float output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH])
{
    for (int f = 0; f < CONV2_NBOUTPUT; f++) // for each filter
    {
        for (int y = 0; y < CONV2_HEIGHT; y++)
        {
            for (int x = 0; x < CONV2_WIDTH; x++)
            {
                float sum = 0.0f;

                for (int c = 0; c < POOL1_NBOUTPUT; c++)
                {
                    for (int ky = 0; ky < CONV2_DIM; ky++)
                    {
                        for (int kx = 0; kx < CONV2_DIM; kx++)
                        {
                            sum += input[c][y + ky][x + kx] * kernel[f][c][ky][kx];
                        }
                    }
                }

                sum += bias[f];
                // ReLU activation
                output[f][y][x] = (sum > 0) ? sum : 0;
            }
        }
    }
}
