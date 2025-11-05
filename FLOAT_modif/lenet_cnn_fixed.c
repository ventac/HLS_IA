/**
 ******************************************************************************
 * @file    lenet_cnn_fixed.c
 * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
 * @version V1.0
 * @date    04 february 2019
 * @brief   Plain C code for the implementation of Convolutional Neural Networks on FPGA
 * @brief   Designed to support Vivado HLS synthesis
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "lenet_cnn_float.h"
#include "lenet_cnn_fixed.h"
#include "float_int.h"

#define FRAC_BITS 12

// GLOBAL VARIABLES
unsigned char REF_IMG[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
float INPUT_NORM[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
float CONV1_KERNEL[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM];
float CONV1_BIAS[CONV1_NBOUTPUT];
float CONV2_KERNEL[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM];
float CONV2_BIAS[CONV2_NBOUTPUT];
float FC1_KERNEL_FLOAT[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
float FC1_BIAS_FLOAT[FC1_NBOUTPUT];
float FC2_KERNEL_FLOAT[FC2_NBOUTPUT][FC1_NBOUTPUT];
float FC2_BIAS_FLOAT[FC2_NBOUTPUT];

int32_t FC1_KERNEL[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
int32_t FC1_BIAS[FC1_NBOUTPUT];
int32_t FC2_KERNEL[FC2_NBOUTPUT][FC1_NBOUTPUT];
int32_t FC2_BIAS[FC2_NBOUTPUT];

float FC2_OUTPUT[FC2_NBOUTPUT];
float SOFTMAX_OUTPUT[FC2_NBOUTPUT];

void convert_fc_weights_to_fixed() {
    for (int i = 0; i < FC1_NBOUTPUT; i++) {
        FC1_BIAS[i] = float_to_fixed(FC1_BIAS_FLOAT[i], FRAC_BITS);
        for (int j = 0; j < POOL2_NBOUTPUT; j++) {
            for (int k = 0; k < POOL2_HEIGHT; k++) {
                for (int l = 0; l < POOL2_WIDTH; l++) {
                    FC1_KERNEL[i][j][k][l] = float_to_fixed(FC1_KERNEL_FLOAT[i][j][k][l], FRAC_BITS);
                }
            }
        }
    }

    for (int i = 0; i < FC2_NBOUTPUT; i++) {
        FC2_BIAS[i] = float_to_fixed(FC2_BIAS_FLOAT[i], FRAC_BITS);
        for (int j = 0; j < FC1_NBOUTPUT; j++) {
            FC2_KERNEL[i][j] = float_to_fixed(FC2_KERNEL_FLOAT[i][j], FRAC_BITS);
        }
    }
}

void lenet_cnn_fixed(float input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
                     float conv1_kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
                     float conv1_bias[CONV1_NBOUTPUT],
                     float conv2_kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
                     float conv2_bias[CONV2_NBOUTPUT],
                     int32_t fc1_kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
                     int32_t fc1_bias[FC1_NBOUTPUT],
                     int32_t fc2_kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
                     int32_t fc2_bias[FC2_NBOUTPUT],
                     float output[FC2_NBOUTPUT]) {

    float conv1_output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH];
    float pool1_output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH];
    float conv2_output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH];
    float pool2_output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];

    int32_t pool2_output_fixed[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
    int32_t fc1_output_fixed[FC1_NBOUTPUT];
    int32_t fc2_output_fixed[FC2_NBOUTPUT];

    Conv1_28x28x1_5x5x20_1_0(input, conv1_kernel, conv1_bias, conv1_output);
    Pool1_24x24x20_2x2x20_2_0(conv1_output, pool1_output);
    Conv2_12x12x20_5x5x40_1_0(pool1_output, conv2_kernel, conv2_bias, conv2_output);
    Pool2_8x8x40_2x2x40_2_0(conv2_output, pool2_output);

    for (int i = 0; i < POOL2_NBOUTPUT; i++) {
        for (int j = 0; j < POOL2_HEIGHT; j++) {
            for (int k = 0; k < POOL2_WIDTH; k++) {
                pool2_output_fixed[i][j][k] = float_to_fixed(pool2_output[i][j][k], FRAC_BITS);
            }
        }
    }

    Fc1_40_400_fixed(pool2_output_fixed, fc1_kernel, fc1_bias, fc1_output_fixed);
    Fc2_400_10_fixed(fc1_output_fixed, fc2_kernel, fc2_bias, fc2_output_fixed);
    Softmax_fixed(fc2_output_fixed, output);
}

int main()
{
    char *hdf5_filename = "lenet_weights.weights.h5";
    char *conv1_weights = "/layers/conv2d/vars/0";
    char *conv1_bias = "/layers/conv2d/vars/1";
    char *conv2_weights = "/layers/conv2d_1/vars/0";
    char *conv2_bias = "/layers/conv2d_1/vars/1";
    char *fc1_weights = "/layers/dense/vars/0";
    char *fc1_bias = "/layers/dense/vars/1";
    char *fc2_weights = "/layers/dense_1/vars/0";
    char *fc2_bias = "/layers/dense_1/vars/1";
    char *test_labels_filename = "mnist/t10k-labels-idx1-ubyte";
    FILE *label_file;
    int ret;
    unsigned char label, number;
    unsigned int error;
    char img_filename[120];
    char img_count[10];
    float max;
    struct timeval start, end;
    double tdiff;
    int m;

    printf("\nReading weights \n");
    ReadConv1Weights(hdf5_filename, conv1_weights, CONV1_KERNEL);
    ReadConv1Bias(hdf5_filename, conv1_bias, CONV1_BIAS);
    ReadConv2Weights(hdf5_filename, conv2_weights, CONV2_KERNEL);
    ReadConv2Bias(hdf5_filename, conv2_bias, CONV2_BIAS);
    ReadFc1Weights(hdf5_filename, fc1_weights, FC1_KERNEL_FLOAT);
    ReadFc1Bias(hdf5_filename, fc1_bias, FC1_BIAS_FLOAT);
    ReadFc2Weights(hdf5_filename, fc2_weights, FC2_KERNEL_FLOAT);
    ReadFc2Bias(hdf5_filename, fc2_bias, FC2_BIAS_FLOAT);

    convert_fc_weights_to_fixed();

    printf("\nOpening labels file \n");
    label_file = fopen(test_labels_filename, "r");
    if (!label_file) {
        printf("Error: Unable to open file %s.\n", test_labels_filename);
        exit(1);
    }

    for (int k = 0; k < 8; k++)
        ret = fscanf(label_file, "%c", &label);

    printf("\nProcessing \n");
    m = 0;
    error = 0;

    gettimeofday(&start, NULL);
    while (1) {
        ret = fscanf(label_file, "%c", &label);
        if (feof(label_file))
            break;

        strcpy(img_filename, "mnist/t10k-images-idx3-ubyte[");
        sprintf(img_count, "%d", m);
        if (m < 10) strcat(img_filename, "0000");
        else if (m < 100) strcat(img_filename, "000");
        else if (m < 1000) strcat(img_filename, "00");
        else if (m < 10000) strcat(img_filename, "0");
        strcat(img_filename, img_count);
        strcat(img_filename, "].pgm");

        printf("\033[%d;%dH%s\n", 7, 0, img_filename);

        ReadPgmFile(img_filename, (unsigned char *)REF_IMG);
        NormalizeImg((unsigned char *)REF_IMG, (float *)INPUT_NORM, IMG_WIDTH, IMG_WIDTH);

        lenet_cnn_fixed(INPUT_NORM, CONV1_KERNEL, CONV1_BIAS, CONV2_KERNEL, CONV2_BIAS,
                        FC1_KERNEL, FC1_BIAS, FC2_KERNEL, FC2_BIAS, SOFTMAX_OUTPUT);

        printf("\n\nSoftmax output: \n");
        max = 0;
        number = 0;
        for (int k = 0; k < FC2_NBOUTPUT; k++) {
            printf("%.2f%% ", SOFTMAX_OUTPUT[k] * 100); 
            if (SOFTMAX_OUTPUT[k] > max) {
                max = SOFTMAX_OUTPUT[k];
                number = k;
            }
        }

        printf("\n\nPredicted: %d 	 Actual: %d\n", number, label);
        if (number != label)
            error++;

        m++;
    }
    gettimeofday(&end, NULL);

    tdiff = (double)(end.tv_sec - start.tv_sec);
    printf("TOTAL PROCESSING TIME (gettimeofday): %f s\n", tdiff);

    printf("\n\nErrors : %d / %d", error, m);
    printf("\n\nSuccess rate = %f%%", (1 - ((float)error / m)) * 100);

    printf("\n\n");

    fclose(label_file);

    return 0;
}
