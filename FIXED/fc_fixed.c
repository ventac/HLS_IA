/**
 * @file fc_fixed.c
 * @brief Fixed-point implementation of fully connected layers and softmax for LeNet-5 CNN
 *
 * This file implements the final classification stages using fixed-point (16.16 format)
 * arithmetic, consisting of two fully connected layers followed by a softmax activation.
 */

#include <stdio.h>
#include <math.h>
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
        // Use 64-bit accumulator to avoid repeated rounding on each multiply
        int64_t acc = 0;
        for (int c = 0; c < POOL2_NBOUTPUT; c++) {
            for (int h = 0; h < POOL2_HEIGHT; h++) {
                for (int w = 0; w < POOL2_WIDTH; w++) {
                    acc += (int64_t)input[c][h][w] * (int64_t)kernel[n][c][h][w];
                }
            }
        }

        // Shift once with rounding
        int64_t round = (int64_t)1 << (FIXED_POINT_SHIFT - 1);
        if (acc >= 0) acc += round; else acc -= round;
        int64_t acc_shifted = acc >> FIXED_POINT_SHIFT;
        if (acc_shifted > INT32_MAX) acc_shifted = INT32_MAX;
        if (acc_shifted < INT32_MIN) acc_shifted = INT32_MIN;

        fixed16_16_t sum = (fixed16_16_t)acc_shifted;
        sum = fixed_add(sum, bias[n]);
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
        int64_t acc = 0;
        for (int i = 0; i < FC1_NBOUTPUT; i++) {
            acc += (int64_t)input[i] * (int64_t)kernel[n][i];
        }
        int64_t round = (int64_t)1 << (FIXED_POINT_SHIFT - 1);
        if (acc >= 0) acc += round; else acc -= round;
        int64_t acc_shifted = acc >> FIXED_POINT_SHIFT;
        if (acc_shifted > INT32_MAX) acc_shifted = INT32_MAX;
        if (acc_shifted < INT32_MIN) acc_shifted = INT32_MIN;

        fixed16_16_t sum = (fixed16_16_t)acc_shifted;
        sum = fixed_add(sum, bias[n]);
        output[n] = sum;
    }
}


/// @brief Softmax layer using fixed-point arithmetic
/// @param vector_in   Input values
/// @param vector_out  Output probabilities
void Softmax_fixed(fixed16_16_t vector_in[FC2_NBOUTPUT], fixed16_16_t vector_out[FC2_NBOUTPUT]) {
    // Compute softmax in floating point for better numerical stability and accuracy,
    // then convert back to fixed-point probabilities.
    float temp[FC2_NBOUTPUT];
    float max_val = (float)fixed_to_float(vector_in[0]);
    for (int i = 1; i < FC2_NBOUTPUT; i++) {
        float v = fixed_to_float(vector_in[i]);
        if (v > max_val) max_val = v;
    }
    float sum = 0.0f;
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        temp[i] = expf(fixed_to_float(vector_in[i]) - max_val);
        sum += temp[i];
    }
    if (sum == 0.0f) sum = 1e-12f;
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        float p = temp[i] / sum;
        vector_out[i] = float_to_fixed(p);
    }
}
