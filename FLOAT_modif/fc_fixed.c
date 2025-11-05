/**
 * @file fc_fixed.c
 * @brief Implementation of the fully connected layers and softmax for LeNet-5 CNN in fixed-point
 *
 * This file implements the final classification stages of the LeNet-5 Convolutional Neural Network,
 * consisting of two fully connected (dense) layers followed by a softmax activation layer.
 * The network structure processes the output from the convolutional layers into final digit
 * classification probabilities.
 *
 * Components:
 * - FC1: First fully connected layer (40 inputs → 400 outputs)
 *        Includes ReLU activation function
 * - FC2: Second fully connected layer (400 inputs → 10 outputs)
 *        Linear activation, no ReLU
 * - Softmax: Final layer that converts raw scores into probabilities
 *           Uses numerical stability optimization by subtracting max value
 *
 * The implementation uses fixed-point arithmetic for efficiency on embedded systems.
 */

#include <stdio.h>
#include <math.h>
#include "lenet_cnn_float.h"
#include "float_int.h"

#define FRAC_BITS 12

/// @brief First Fully Connected Layer FC1: transforms 40 inputs to 400 outputs
/// @param input    Layer input from previous pooling layer
/// @param kernel   Weight matrix
/// @param bias     Bias values
/// @param output   Layer output
void Fc1_40_400_fixed(
    const int32_t input[restrict POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 
    const int32_t kernel[restrict FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 
    const int32_t bias[restrict FC1_NBOUTPUT], 
    int32_t output[restrict FC1_NBOUTPUT]
) {
    for (int n = 0; n < FC1_NBOUTPUT; n++) {
        int64_t sum = (int64_t)bias[n] << FRAC_BITS;
        for (int c = 0; c < POOL2_NBOUTPUT; c++) {
            for (int h = 0; h < POOL2_HEIGHT; h++) {
                for (int w = 0; w < POOL2_WIDTH; w++) {
                    sum += (int64_t)input[c][h][w] * kernel[n][c][h][w];
                }
            }
        }
        sum >>= FRAC_BITS;
        output[n] = (sum > 0) ? (int32_t)sum : 0;
    }
}

/// @brief Second Fully Connected Layer FC2: transforms 400 inputs to 10 outputs
/// @param input    Layer input (output from FC1)
/// @param kernel   Weight matrix
/// @param bias     Bias values
/// @param output   Layer output
void Fc2_400_10_fixed(
    const int32_t input[restrict FC1_NBOUTPUT], 
    const int32_t kernel[restrict FC2_NBOUTPUT][FC1_NBOUTPUT], 
    const int32_t bias[restrict FC2_NBOUTPUT], 
    int32_t output[restrict FC2_NBOUTPUT]
) {
    for (int n = 0; n < FC2_NBOUTPUT; n++) {
        int64_t sum = (int64_t)bias[n] << FRAC_BITS;
        for (int i = 0; i < FC1_NBOUTPUT; i++) {
            sum += (int64_t)input[i] * kernel[n][i];
        }
        sum >>= FRAC_BITS;
        output[n] = (int32_t)sum;
    }
}

/// @brief Softmax layer: normalizes scores into probabilities
/// @param vector_in   Input values
/// @param vector_out  Output probabilities
void Softmax_fixed(int32_t vector_in[FC2_NBOUTPUT], float vector_out[FC2_NBOUTPUT]) {
    float float_in[FC2_NBOUTPUT];
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        float_in[i] = fixed_to_float(vector_in[i], FRAC_BITS);
    }

    float max_val = float_in[0];
    for (int i = 1; i < FC2_NBOUTPUT; i++) {
        if (float_in[i] > max_val)
            max_val = float_in[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        vector_out[i] = expf(float_in[i] - max_val);
        sum_exp += vector_out[i];
    }

    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        vector_out[i] *= inv_sum;
    }
}
