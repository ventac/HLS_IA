#include <stdio.h>
#include <math.h>
#include "lenet_cnn_float.h"

/// ===========================================================
/// @brief Fully Connected Layer FC1 : 40 → 400
/// ===========================================================
/// @param input    [POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH] : entrée du layer
/// @param kernel   [FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH] : poids
/// @param bias     [FC1_NBOUTPUT] : biais
/// @param output   [FC1_NBOUTPUT] : sortie du layer
void Fc1_40_400(
    float input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    float kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    float bias[FC1_NBOUTPUT],
    float output[FC1_NBOUTPUT]
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
        // Activation ReLU
        output[n] = fmaxf(0.0f, sum);
    }
}

/// ===========================================================
/// @brief Fully Connected Layer FC2 : 400 → 10
/// ===========================================================
/// @param input    [FC1_NBOUTPUT] : entrée du layer (sortie de FC1)
/// @param kernel   [FC2_NBOUTPUT][FC1_NBOUTPUT] : poids
/// @param bias     [FC2_NBOUTPUT] : biais
/// @param output   [FC2_NBOUTPUT] : sortie du layer
void Fc2_400_10(
    float input[FC1_NBOUTPUT],
    float kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
    float bias[FC2_NBOUTPUT],
    float output[FC2_NBOUTPUT]
) {
    for (int n = 0; n < FC2_NBOUTPUT; n++) {
        float sum = bias[n];
        for (int i = 0; i < FC1_NBOUTPUT; i++) {
            sum += input[i] * kernel[n][i];
        }
        output[n] = sum; // pas d’activation ici : Softmax ensuite
    }
}


/// ===========================================================
/// @brief Softmax layer : normalise les scores en probabilités
/// ===========================================================
/// @param vector_in   [FC2_NBOUTPUT] : valeurs en entrée
/// @param vector_out  [FC2_NBOUTPUT] : probabilités en sortie
void Softmax(float vector_in[FC2_NBOUTPUT], float vector_out[FC2_NBOUTPUT]) {
    // 1️⃣ Trouver la valeur max pour la stabilité numérique
    float max_val = vector_in[0];
    for (int i = 1; i < FC2_NBOUTPUT; i++) {
        if (vector_in[i] > max_val)
            max_val = vector_in[i];
    }

    // 2️⃣ Calcul des exponentielles décalées
    float sum_exp = 0.0f;
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        vector_out[i] = expf(vector_in[i] - max_val);
        sum_exp += vector_out[i];
    }

    // 3️⃣ Normalisation
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        vector_out[i] *= inv_sum;
    }
}
