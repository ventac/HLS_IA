#include "lenet_cnn_float.h"
#include <math.h>

/// @brief Convolution layer 1 : (28x28x1) -> (24x24x20)
/// @param input [1][28][28]
/// @param kernel [20][1][5][5]
/// @param bias [20]
/// @param output [20][24][24]
void Conv1_28x28x1_5x5x20_1_0(
    float input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    float kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    float bias[CONV1_NBOUTPUT],
    float output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH])
{
    for (int f = 0; f < CONV1_NBOUTPUT; f++) // pour chaque filtre
    {
        for (int y = 0; y < CONV1_HEIGHT; y++)
        {
            for (int x = 0; x < CONV1_WIDTH; x++)
            {
                float sum = 0.0f;

                for (int c = 0; c < IMG_DEPTH; c++)
                {
                    for (int ky = 0; ky < CONV1_DIM; ky++)
                    {
                        for (int kx = 0; kx < CONV1_DIM; kx++)
                        {
                            sum += input[c][y + ky][x + kx] * kernel[f][c][ky][kx];
                        }
                    }
                }

                sum += bias[f];
                // ReLU activation
                output[f][y][x] = (sum > 0) ? sum : 0;
            }
        }
    }
}

/// @brief Convolution layer 2 : (12x12x20) -> (8x8x40)
/// @param input [20][12][12]
/// @param kernel [40][20][5][5]
/// @param bias [40]
/// @param output [40][8][8]
void Conv2_12x12x20_5x5x40_1_0(
    float input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],
    float kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    float bias[CONV2_NBOUTPUT],
    float output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH])
{
    for (int f = 0; f < CONV2_NBOUTPUT; f++) // pour chaque filtre
    {
        for (int y = 0; y < CONV2_HEIGHT; y++)
        {
            for (int x = 0; x < CONV2_WIDTH; x++)
            {
                float sum = 0.0f;

                for (int c = 0; c < POOL1_NBOUTPUT; c++)
                {
                    for (int ky = 0; ky < CONV2_DIM; ky++)
                    {
                        for (int kx = 0; kx < CONV2_DIM; kx++)
                        {
                            sum += input[c][y + ky][x + kx] * kernel[f][c][ky][kx];
                        }
                    }
                }

                sum += bias[f];
                // ReLU activation
                output[f][y][x] = (sum > 0) ? sum : 0;
            }
        }
    }
}
