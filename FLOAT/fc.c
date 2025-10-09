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

void Fc1_40_400(const float input[restrict POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 
                 const float kernel[restrict FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 
                 const float bias[restrict FC1_NBOUTPUT], 
                 float output[restrict FC1_NBOUTPUT]) {

    const int INPUT_SIZE = POOL2_NBOUTPUT * POOL2_HEIGHT * POOL2_WIDTH;

    // Initialize output to zero
    for (int i = 0; i < FC1_NBOUTPUT; i++) {
        output[i] = 0.0f;
    }

    // Flatten input
    float input_flat[INPUT_SIZE];
    int idx = 0;
    for (int j = 0; j < POOL2_NBOUTPUT; j++)
        for (int h = 0; h < POOL2_HEIGHT; h++)
            for (int w = 0; w < POOL2_WIDTH; w++)
                input_flat[idx++] = input[j][h][w];

    // FC computation
    for (int i = 0; i < FC1_NBOUTPUT; i++) {
        float sum = bias[i];
        idx = 0;
        for (int j = 0; j < POOL2_NBOUTPUT; j++)
            for (int h = 0; h < POOL2_HEIGHT; h++)
                for (int w = 0; w < POOL2_WIDTH; w++)
                    sum += input_flat[idx++] * kernel[i][j][h][w];
        output[i] = sum;
    }

    // // Perform the fully connected operation
    // for (int i = 0; i < FC1_NBOUTPUT; i++) {
    //     for (int j = 0; j < POOL2_NBOUTPUT; j++) {
    //         for (int h = 0; h < POOL2_HEIGHT; h++) {
    //             for (int w = 0; w < POOL2_WIDTH; w++) {
    //                 output[i] += input[j][h][w] * kernel[i][j][h][w];
    //             }
    //         }
    //     }
    //     // Add bias
    //     output[i] += bias[i];
    // }
}


/// @brief Computes the output of the second fully connected (dense) layer, taking the output of the first fully connected layer as input.
/// @param input Array containing the activations from the first fully connected layer (size FC1_NBOUTPUT).
/// @param kernel 2D array of weights connecting each neuron from the first layer to each neuron in the second layer (size FC2_NBOUTPUT x FC1_NBOUTPUT).
/// @param bias Array of bias values for each neuron in the second fully connected layer (size FC2_NBOUTPUT).
/// @param output Array to store the computed activations for the second fully connected layer (size FC2_NBOUTPUT).
void Fc2_400_10(const float input[restrict FC1_NBOUTPUT], 
                 const float kernel[restrict FC2_NBOUTPUT][FC1_NBOUTPUT], 
                 const float bias[restrict FC2_NBOUTPUT], 
                 float output[restrict FC2_NBOUTPUT]) {
    // // Initialize output to zero
    // for (int i = 0; i < FC2_NBOUTPUT; i++) {
    //     output[i] = 0.0f;
    // }

    // // Perform the fully connected operation
    // for (int i = 0; i < FC2_NBOUTPUT; i++) {
    //     for (int j = 0; j < FC1_NBOUTPUT; j++) {
    //         output[i] += input[j] * kernel[i][j];
    //     }
    //     // Add bias
    //     output[i] += bias[i];
    // }

    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        float sum = bias[i];
        for (int j = 0; j < FC1_NBOUTPUT; j++) {
            sum += input[j] * kernel[i][j];
        }
        output[i] = sum;
    }

}
