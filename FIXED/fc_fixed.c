/**
 * @file fc_fixed.c
 * @brief Fixed-point implementation of fully connected layers and softmax for LeNet-5 CNN
 *
 * This file implements the final classification stages using fixed-point (16.16 format)
 * arithmetic, consisting of two fully connected layers followed by a softmax activation.
 */

#include <math.h>
#include "lenet_cnn_fixed.h"

/// @brief First Fully Connected Layer FC1 using fixed-point arithmetic
/// @param input    Layer input from previous pooling layer [POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]
/// @param kernel   Weight matrix [FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]
/// @param bias     Bias values [FC1_NBOUTPUT]
/// @param output   Layer output [FC1_NBOUTPUT]
void Fc1_40_400_fixed(
    short input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 
    short kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 
    short bias[FC1_NBOUTPUT], 
    short output[FC1_NBOUTPUT]
) {
    unsigned short n, c, h, w;
    int acc;

    for (n = 0; n < FC1_NBOUTPUT; n++) {
        acc = 0;

        for (c = 0; c < POOL2_NBOUTPUT; c++) {
            for (h = 0; h < POOL2_HEIGHT; h++) {
                for (w = 0; w < POOL2_WIDTH; w++) {
                    acc += (int)input[c][h][w] * (int)kernel[n][c][h][w];
                }
            }
        }

        // Fixed-point scaling and bias addition
        acc = (acc >> FIXED_POINT) + bias[n];

        // ReLU activation
        output[n] = (short)(acc > 0 ? acc : 0);
    }
}

/// @brief Second Fully Connected Layer FC2 using fixed-point arithmetic
/// @param input    Layer input (output from FC1) [FC1_NBOUTPUT]
/// @param kernel   Weight matrix [FC2_NBOUTPUT][FC1_NBOUTPUT]
/// @param bias     Bias values [FC2_NBOUTPUT]
/// @param output   Layer output [FC2_NBOUTPUT]
void Fc2_400_10_fixed(
    short input[FC1_NBOUTPUT], 
    short kernel[FC2_NBOUTPUT][FC1_NBOUTPUT], 
    short bias[FC2_NBOUTPUT], 
    short output[FC2_NBOUTPUT]
) {
    unsigned short n, i;
    int sum;

    for (n = 0; n < FC2_NBOUTPUT; n++) {
        sum = 0;

        for (i = 0; i < FC1_NBOUTPUT; i++) {
            sum += (int)input[i] * (int)kernel[n][i];
        }

        // Fixed-point scaling and bias addition
        sum = (sum >> FIXED_POINT) + bias[n];

        // ReLU activation
        output[n] = (short)(sum > 0 ? sum : 0);
    }
}

/// @brief Numerically stable Softmax layer using fixed-point arithmetic
/// @param vector_in   Input values [FC2_NBOUTPUT] in fixed-point
/// @param vector_out  Output probabilities [FC2_NBOUTPUT] as floats
void Softmax_fixed(short vector_in[FC2_NBOUTPUT], float vector_out[FC2_NBOUTPUT]) {
    unsigned short i;
    float soft_sum = 0.0f;
    short max_val = vector_in[0];

    // Find maximum input for numerical stability
    for (i = 1; i < FC2_NBOUTPUT; i++) {
        if (vector_in[i] > max_val) {
            max_val = vector_in[i];
        }
    }

    // Compute exponentials of (input - max) and sum
    for (i = 0; i < FC2_NBOUTPUT; i++) {
        vector_out[i] = expf(SHORT2FLOAT(vector_in[i] - max_val));
        soft_sum += vector_out[i];
    }

    // Normalize to get probabilities
    for (i = 0; i < FC2_NBOUTPUT; i++) {
        vector_out[i] /= soft_sum;
    }
}
