#include <stdio.h>
#include <math.h>
#include "lenet_cnn_float.h"

/// @brief Fully Connected Layer FC1 : 40 → 400
/// @param input    entrée du layer
/// @param kernel   poids
/// @param bias     biais
/// @param output   sortie du layer
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

/// @brief Fully Connected Layer FC2 : 400 → 10
/// @param input    entrée du layer (sortie de FC1)
/// @param kernel   poids
/// @param bias     biais
/// @param output   sortie du layer
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


/// @brief Softmax layer : normalise les scores en probabilités
/// @param vector_in   valeurs en entrée
/// @param vector_out  probabilités en sortie
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
