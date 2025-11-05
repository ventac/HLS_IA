/**
 * @file fc.c
 * @brief Implementation of the fully connected layers and softmax for LeNet-5 CNN in floating point
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
 * The implementation uses floating-point arithmetic for maximum accuracy and
 * includes optimizations for numerical stability in the softmax computation.
 */

#include <stdio.h>
#include <math.h>
#include "lenet_cnn_float.h"

/// @brief First Fully Connected Layer FC1: transforms 40 inputs to 400 outputs
/// @param input    Layer input from previous pooling layer
/// @param kernel   Weight matrix
/// @param bias     Bias values
/// @param output   Layer output
void Fc1_40_400(
    const float input[restrict POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 
                 const float kernel[restrict FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 
                 const float bias[restrict FC1_NBOUTPUT], 
                 float output[restrict FC1_NBOUTPUT]
) {
    for (int n = 0; n < FC1_NBOUTPUT; n++) {
        float sum = bias[n];
        for (int c = 0; c < POOL2_NBOUTPUT; c++) {
            for (int h = 0; h < POOL2_HEIGHT; h++) {
                for (int w = 0; w < POOL2_WIDTH; w++) {
                    sum += input[c][h][w] * kernel[n][c][h][w];
                }
            }
        }
        output[n] = fmaxf(0.0f, sum);
    }
}

/// @brief Second Fully Connected Layer FC2: transforms 400 inputs to 10 outputs
/// @param input    Layer input (output from FC1)
/// @param kernel   Weight matrix
/// @param bias     Bias values
/// @param output   Layer output
void Fc2_400_10(
    const float input[restrict FC1_NBOUTPUT], 
                 const float kernel[restrict FC2_NBOUTPUT][FC1_NBOUTPUT], 
                 const float bias[restrict FC2_NBOUTPUT], 
                 float output[restrict FC2_NBOUTPUT]
) {
    for (int n = 0; n < FC2_NBOUTPUT; n++) {
        float sum = bias[n];
        for (int i = 0; i < FC1_NBOUTPUT; i++) {
            sum += input[i] * kernel[n][i];
        }
        output[n] = sum;
    }
}


/// @brief Softmax layer: normalizes scores into probabilities
/// @param vector_in   Input values
/// @param vector_out  Output probabilities
void Softmax(float vector_in[FC2_NBOUTPUT], float vector_out[FC2_NBOUTPUT]) {
    float max_val = vector_in[0];
    for (int i = 1; i < FC2_NBOUTPUT; i++) {
        if (vector_in[i] > max_val)
            max_val = vector_in[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        vector_out[i] = expf(vector_in[i] - max_val);
        sum_exp += vector_out[i];
    }

    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        vector_out[i] *= inv_sum;
    }

}
