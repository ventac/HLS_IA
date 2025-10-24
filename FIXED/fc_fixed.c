/**
 * @file fc_fixed.c
 * @brief Fixed-point implementation of fully connected layers and softmax for LeNet-5 CNN
 *
 * This file implements the final classification stages using fixed-point (16.16 format)
 * arithmetic, consisting of two fully connected layers followed by a softmax activation.
 */

#include <stdio.h>
#include "lenet_cnn_float.h"
#include "fixed_point.h"

/// @brief First Fully Connected Layer FC1 using fixed-point arithmetic
/// @param input    Layer input from previous pooling layer
/// @param kernel   Weight matrix
/// @param bias     Bias values
/// @param output   Layer output
void Fc1_40_400_fixed(
    const fixed16_16_t input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 
    const fixed16_16_t kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 
    const fixed16_16_t bias[FC1_NBOUTPUT], 
    fixed16_16_t output[FC1_NBOUTPUT]
) {
    for (int n = 0; n < FC1_NBOUTPUT; n++) {
        fixed16_16_t sum = bias[n];
        for (int c = 0; c < POOL2_NBOUTPUT; c++) {
            for (int h = 0; h < POOL2_HEIGHT; h++) {
                for (int w = 0; w < POOL2_WIDTH; w++) {
                    fixed16_16_t prod = fixed_mul(input[c][h][w], kernel[n][c][h][w]);
                    sum = fixed_add(sum, prod);
                }
            }
        }
        output[n] = fixed_relu(sum);
    }
}

/// @brief Second Fully Connected Layer FC2 using fixed-point arithmetic
/// @param input    Layer input (output from FC1)
/// @param kernel   Weight matrix
/// @param bias     Bias values
/// @param output   Layer output
void Fc2_400_10_fixed(
    const fixed16_16_t input[FC1_NBOUTPUT], 
    const fixed16_16_t kernel[FC2_NBOUTPUT][FC1_NBOUTPUT], 
    const fixed16_16_t bias[FC2_NBOUTPUT], 
    fixed16_16_t output[FC2_NBOUTPUT]
) {
    for (int n = 0; n < FC2_NBOUTPUT; n++) {
        fixed16_16_t sum = bias[n];
        for (int i = 0; i < FC1_NBOUTPUT; i++) {
            fixed16_16_t prod = fixed_mul(input[i], kernel[n][i]);
            sum = fixed_add(sum, prod);
        }
        output[n] = sum;
    }
}


/// @brief Softmax layer using fixed-point arithmetic
/// @param vector_in   Input values
/// @param vector_out  Output probabilities
void Softmax_fixed(fixed16_16_t vector_in[FC2_NBOUTPUT], fixed16_16_t vector_out[FC2_NBOUTPUT]) {
    // Find max value for numerical stability
    fixed16_16_t max_val = vector_in[0];
    for (int i = 1; i < FC2_NBOUTPUT; i++) {
        if (fixed_gt(vector_in[i], max_val))
            max_val = vector_in[i];
    }

    // Compute exp(x - max) for each element
    fixed16_16_t sum_exp = FIXED_ZERO;
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        fixed16_16_t diff = fixed_sub(vector_in[i], max_val);
        vector_out[i] = fixed_exp_approx(diff);
        sum_exp = fixed_add(sum_exp, vector_out[i]);
    }

    // Normalize
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        vector_out[i] = fixed_div(vector_out[i], sum_exp);
    }
}
