/**
 * @file pool_fixed.c
 * @brief Fixed-point implementation of pooling layers for LeNet CNN
 * 
 * This file contains the fixed-point (16.16 format) implementation of two 
 * pooling layers using 2x2 max pooling operation.
 */

#include <stdio.h>
#include "lenet_cnn_fixed.h"
#include "fixed_point.h"

/// @brief Pool1 with fixed-point arithmetic
/// @param conv1_output entries of conv1 layer
/// @param pool1_output entries of pool1 layer
void Pool1_24x24x20_2x2x20_2_0_fixed(
    fixed16_16_t conv1_output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
    fixed16_16_t pool1_output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH])
{
    short ch, oy, ox;
    for (ch = 0; ch < CONV1_NBOUTPUT; ch++) {
        for (oy = 0; oy < POOL1_HEIGHT; oy++) {
            for (ox = 0; ox < POOL1_WIDTH; ox++) {
                short in_y = oy * 2;
                short in_x = ox * 2;

                fixed16_16_t m = conv1_output[ch][in_y][in_x];
                m = fixed_max(m, conv1_output[ch][in_y][in_x + 1]);
                m = fixed_max(m, conv1_output[ch][in_y + 1][in_x]);
                m = fixed_max(m, conv1_output[ch][in_y + 1][in_x + 1]);

                pool1_output[ch][oy][ox] = m;
            }
        }
    }
}

/// @brief Pool2 with fixed-point arithmetic
/// @param conv2_output entries of conv2 layer
/// @param pool2_output entries of pool2 layer
void Pool2_8x8x40_2x2x40_2_0_fixed(
    fixed16_16_t conv2_output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
    fixed16_16_t pool2_output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH])
{
    short ch, oy, ox;
    for (ch = 0; ch < CONV2_NBOUTPUT; ch++) {
        for (oy = 0; oy < POOL2_HEIGHT; oy++) {
            for (ox = 0; ox < POOL2_WIDTH; ox++) {
                short in_y = oy * 2;
                short in_x = ox * 2;

                fixed16_16_t m = conv2_output[ch][in_y][in_x];
                m = fixed_max(m, conv2_output[ch][in_y][in_x + 1]);
                m = fixed_max(m, conv2_output[ch][in_y + 1][in_x]);
                m = fixed_max(m, conv2_output[ch][in_y + 1][in_x + 1]);

                pool2_output[ch][oy][ox] = m;
            }
        }
    }
}
