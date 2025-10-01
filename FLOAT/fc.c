#include <stdio.h>
#include "lenet_cnn_float.h"
#include <math.h>

/// @brief 
/// @param vector_in 
/// @param vector_out 
void Softmax(float vector_in[FC2_NBOUTPUT], float vector_out[FC2_NBOUTPUT]) {
    
    // Find the maximum value in the input vector for numerical stability
    float max = vector_in[0];
    for (int i = 1; i < FC2_NBOUTPUT; i++) {
        if (vector_in[i] > max) {
            max = vector_in[i];
        }
    }

    // Calculate the exponentials and sum them
    float sum = 0.0f;
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        vector_out[i] = exp(vector_in[i] - max); // Subtract max for numerical stability
        sum += vector_out[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        vector_out[i] /= sum;
    }
}

void Fc1_40_400(float input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 
                 float kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 
                 float bias[FC1_NBOUTPUT], 
                 float output[FC1_NBOUTPUT]) {
    // Initialize output to zero
    for (int i = 0; i < FC1_NBOUTPUT; i++) {
        output[i] = 0.0f;
    }

    // Perform the fully connected operation
    for (int i = 0; i < FC1_NBOUTPUT; i++) {
        for (int j = 0; j < POOL2_NBOUTPUT; j++) {
            for (int h = 0; h < POOL2_HEIGHT; h++) {
                for (int w = 0; w < POOL2_WIDTH; w++) {
                    output[i] += input[j][h][w] * kernel[i][j][h][w];
                }
            }
        }
        // Add bias
        output[i] += bias[i];
    }
}


/// @brief Prends la sortie de la Fully connected 1
/// @param input La quantité de neurones de fc1
/// @param kernel Le poids 
/// @param bias 
/// @param output 
void Fc2_400_10(float input[FC1_NBOUTPUT], 
                 float kernel[FC2_NBOUTPUT][FC1_NBOUTPUT], 
                 float bias[FC2_NBOUTPUT], 
                 float output[FC2_NBOUTPUT]) {
    // Initialize output to zero
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        output[i] = 0.0f;
    }

    // Perform the fully connected operation
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        for (int j = 0; j < FC1_NBOUTPUT; j++) {
            output[i] += input[j] * kernel[i][j];
        }
        // Add bias
        output[i] += bias[i];
    }
}
