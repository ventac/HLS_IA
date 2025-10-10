/**
 * @file pool.c
 * @brief Implementation of pooling layers for LeNet CNN in floating-point arithmetic
 * 
 * This file contains the implementation of two pooling layers used in the LeNet CNN architecture:
 * - Pool1: Performs 2x2 max pooling on a 24x24x20 input feature map, producing a 12x12x20 output
 * - Pool2: Performs 2x2 max pooling on a 8x8x40 input feature map, producing a 4x4x40 output
 * 
 * Both pooling layers use:
 * - Stride of 2
 * - Window size of 2x2
 * - Max pooling operation (selecting maximum value in each 2x2 window)
 * - No padding
 */

#include <stdio.h>
#include "lenet_cnn_float.h"

/// @brief Pool1
/// @param conv1_output entries of conv1 layer
/// @param pool1_output entries of pool1 layer
void Pool1_24x24x20_2x2x20_2_0(float conv1_output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
                                float pool1_output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH])
{
    short ch, oy, ox;
    for (ch = 0; ch < CONV1_NBOUTPUT; ch++) {
        for (oy = 0; oy < POOL1_HEIGHT; oy++) {
            for (ox = 0; ox < POOL1_WIDTH; ox++) {
                short in_y = oy * 2;
                short in_x = ox * 2;

                float m = conv1_output[ch][in_y][in_x];

                if (conv1_output[ch][in_y][in_x + 1] > m)
                    m = conv1_output[ch][in_y][in_x + 1];

                if (conv1_output[ch][in_y + 1][in_x] > m)
                    m = conv1_output[ch][in_y + 1][in_x];

                if (conv1_output[ch][in_y + 1][in_x + 1] > m)
                    m = conv1_output[ch][in_y + 1][in_x + 1];

                pool1_output[ch][oy][ox] = m;
            }
        }
    }
}

/// @brief Pool2
/// @param conv2_output entries of conv2 layer
/// @param pool2_output entries of pool2 layer
void Pool2_8x8x40_2x2x40_2_0(float conv2_output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
                              float pool2_output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH])
{
    short ch, oy, ox;
    for (ch = 0; ch < CONV2_NBOUTPUT; ch++) {
        for (oy = 0; oy < POOL2_HEIGHT; oy++) {
            for (ox = 0; ox < POOL2_WIDTH; ox++) {
                short in_y = oy * 2;
                short in_x = ox * 2;

                float m = conv2_output[ch][in_y][in_x];

                if (conv2_output[ch][in_y][in_x + 1] > m)
                    m = conv2_output[ch][in_y][in_x + 1];
                if (conv2_output[ch][in_y + 1][in_x] > m)
                    m = conv2_output[ch][in_y + 1][in_x];
                if (conv2_output[ch][in_y + 1][in_x + 1] > m)
                    m = conv2_output[ch][in_y + 1][in_x + 1];

                pool2_output[ch][oy][ox] = m;
            }
        }
    }
}
