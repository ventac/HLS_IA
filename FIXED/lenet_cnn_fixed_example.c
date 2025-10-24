/**
 * @file lenet_cnn_fixed_example.c
 * @brief Example of how to integrate fixed-point arithmetic into LeNet-5 CNN
 *
 * This file demonstrates:
 * 1. Converting weights and input from float to fixed-point
 * 2. Running the network in fixed-point
 * 3. Converting output back to float for softmax and display
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "globals.h"
#include "lenet_cnn_float.h"
#include "fixed_point.h"
#include "lenet_cnn_fixed.h"
#include "conversion_utils.h"

// Declare fixed-point versions of all network parameters
fixed16_16_t INPUT_NORM_FIXED[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
fixed16_16_t CONV1_KERNEL_FIXED[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM];
fixed16_16_t CONV1_BIAS_FIXED[CONV1_NBOUTPUT];
fixed16_16_t CONV2_KERNEL_FIXED[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM];
fixed16_16_t CONV2_BIAS_FIXED[CONV2_NBOUTPUT];
fixed16_16_t FC1_KERNEL_FIXED[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
fixed16_16_t FC1_BIAS_FIXED[FC1_NBOUTPUT];
fixed16_16_t FC2_KERNEL_FIXED[FC2_NBOUTPUT][FC1_NBOUTPUT];
fixed16_16_t FC2_BIAS_FIXED[FC2_NBOUTPUT];
fixed16_16_t FC2_OUTPUT_FIXED[FC2_NBOUTPUT];
fixed16_16_t SOFTMAX_OUTPUT_FIXED[FC2_NBOUTPUT];

// Keep float versions for loading from HDF5
extern unsigned char REF_IMG[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
extern float INPUT_NORM[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
extern float CONV1_KERNEL[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM];
extern float CONV1_BIAS[CONV1_NBOUTPUT];
extern float CONV2_KERNEL[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM];
extern float CONV2_BIAS[CONV2_NBOUTPUT];
extern float FC1_KERNEL[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
extern float FC1_BIAS[FC1_NBOUTPUT];
extern float FC2_KERNEL[FC2_NBOUTPUT][FC1_NBOUTPUT];
extern float FC2_BIAS[FC2_NBOUTPUT];
extern float FC2_OUTPUT[FC2_NBOUTPUT];
extern float SOFTMAX_OUTPUT[FC2_NBOUTPUT];

// Top Level function using fixed-point arithmetic
void lenet_cnn_fixed(
    fixed16_16_t input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    fixed16_16_t conv1_kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    fixed16_16_t conv1_bias[CONV1_NBOUTPUT],
    fixed16_16_t conv2_kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    fixed16_16_t conv2_bias[CONV2_NBOUTPUT],
    fixed16_16_t fc1_kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    fixed16_16_t fc1_bias[FC1_NBOUTPUT],
    fixed16_16_t fc2_kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
    fixed16_16_t fc2_bias[FC2_NBOUTPUT],
    fixed16_16_t output[FC2_NBOUTPUT])
{
    // Intermediate buffers
    fixed16_16_t conv1_output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH];
    fixed16_16_t pool1_output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH];
    fixed16_16_t conv2_output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH];
    fixed16_16_t pool2_output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
    fixed16_16_t fc1_output[FC1_NBOUTPUT];

    // Execute network layers
    Conv1_28x28x1_5x5x20_1_0_fixed(input, conv1_kernel, conv1_bias, conv1_output);
    Pool1_24x24x20_2x2x20_2_0_fixed(conv1_output, pool1_output);
    Conv2_12x12x20_5x5x40_1_0_fixed(pool1_output, conv2_kernel, conv2_bias, conv2_output);
    Pool2_8x8x40_2x2x40_2_0_fixed(conv2_output, pool2_output);
    Fc1_40_400_fixed(pool2_output, fc1_kernel, fc1_bias, fc1_output);
    Fc2_400_10_fixed(fc1_output, fc2_kernel, fc2_bias, output);
}

/**
 * @brief Main function deploying LeNet inference CNN on MNIST dataset using fixed-point arithmetic
 */
void main()
{
    short x, y, z, k, m;
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
    unsigned char labels_legend[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    char img_filename[120];
    char img_count[10];
    float max;
    struct timeval start, end;
    double tdiff;

    printf("\e[1;1H\e[2J");

    printf("\n========================================\n");
    printf("LeNet-5 CNN with Fixed-Point Arithmetic\n");
    printf("========================================\n");

    printf("\nReading weights from HDF5 file...\n");
    ReadConv1Weights(hdf5_filename, conv1_weights, CONV1_KERNEL);
    ReadConv1Bias(hdf5_filename, conv1_bias, CONV1_BIAS);
    ReadConv2Weights(hdf5_filename, conv2_weights, CONV2_KERNEL);
    ReadConv2Bias(hdf5_filename, conv2_bias, CONV2_BIAS);
    ReadFc1Weights(hdf5_filename, fc1_weights, FC1_KERNEL);
    ReadFc1Bias(hdf5_filename, fc1_bias, FC1_BIAS);
    ReadFc2Weights(hdf5_filename, fc2_weights, FC2_KERNEL);
    ReadFc2Bias(hdf5_filename, fc2_bias, FC2_BIAS);

    printf("\nConverting weights to fixed-point format (16.16)...\n");
    convert_conv1_kernel_to_fixed(CONV1_KERNEL, CONV1_KERNEL_FIXED);
    convert_conv1_bias_to_fixed(CONV1_BIAS, CONV1_BIAS_FIXED);
    convert_conv2_kernel_to_fixed(CONV2_KERNEL, CONV2_KERNEL_FIXED);
    convert_conv2_bias_to_fixed(CONV2_BIAS, CONV2_BIAS_FIXED);
    convert_fc1_kernel_to_fixed(FC1_KERNEL, FC1_KERNEL_FIXED);
    convert_fc1_bias_to_fixed(FC1_BIAS, FC1_BIAS_FIXED);
    convert_fc2_kernel_to_fixed(FC2_KERNEL, FC2_KERNEL_FIXED);
    convert_fc2_bias_to_fixed(FC2_BIAS, FC2_BIAS_FIXED);
    printf("Conversion complete!\n");

    printf("\nOpening labels file...\n");
    label_file = fopen(test_labels_filename, "r");
    if (!label_file)
    {
        printf("Error: Unable to open file %s.\n", test_labels_filename);
        exit(1);
    }

    for (k = 0; k < 8; k++) // Skip 8 first header bytes
        ret = fscanf(label_file, "%c", &label);

    printf("\nProcessing MNIST test images with fixed-point arithmetic...\n");
    printf("========================================\n");
    
    m = 0;      // test image counter
    error = 0;  // number of mispredictions

    // MAIN TEST LOOP
    gettimeofday(&start, NULL);
    while (1)
    {
        ret = fscanf(label_file, "%c", &label);
        if (feof(label_file))
            break;

        // Build filename
        strcpy(img_filename, "mnist/t10k-images-idx3-ubyte[");
        sprintf(img_count, "%d", m);
        if (m < 10)
            strcat(img_filename, "0000");
        else if (m < 100)
            strcat(img_filename, "000");
        else if (m < 1000)
            strcat(img_filename, "00");
        else if (m < 10000)
            strcat(img_filename, "0");
        strcat(img_filename, img_count);
        strcat(img_filename, "].pgm");

        printf("\033[%d;%dH%s\n", 7, 0, img_filename);

        // Read and normalize image (float)
        ReadPgmFile(img_filename, (unsigned char *)REF_IMG);
        NormalizeImg((unsigned char *)REF_IMG, (float *)INPUT_NORM, IMG_WIDTH, IMG_WIDTH);

        // Convert input to fixed-point
        convert_input_to_fixed(INPUT_NORM, INPUT_NORM_FIXED);

        // Run fixed-point inference
        lenet_cnn_fixed(INPUT_NORM_FIXED,
                        CONV1_KERNEL_FIXED,
                        CONV1_BIAS_FIXED,
                        CONV2_KERNEL_FIXED,
                        CONV2_BIAS_FIXED,
                        FC1_KERNEL_FIXED,
                        FC1_BIAS_FIXED,
                        FC2_KERNEL_FIXED,
                        FC2_BIAS_FIXED,
                        FC2_OUTPUT_FIXED);

        // Apply softmax in fixed-point
        Softmax_fixed(FC2_OUTPUT_FIXED, SOFTMAX_OUTPUT_FIXED);

        // Find prediction
        printf("\n\nSoftmax output (Fixed-Point):\n");
        max = 0;
        number = 0;
        for (k = 0; k < FC2_NBOUTPUT; k++)
        {
            float prob = fixed_to_float(SOFTMAX_OUTPUT_FIXED[k]);
            printf("%.2f%% ", prob * 100);
            if (prob > max)
            {
                max = prob;
                number = k;
            }
        }

        printf("\n\nPredicted: %d \t Actual: %d", labels_legend[number], label);
        if (labels_legend[number] != label)
        {
            printf(" [ERROR]");
            error = error + 1;
        }
        else
        {
            printf(" [OK]");
        }
        printf("\n");

        m++;

    } // END MAIN TEST LOOP
    
    gettimeofday(&end, NULL);

    tdiff = (double)(end.tv_sec - start.tv_sec);
    
    printf("\n\n========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    printf("Total images processed: %d\n", m);
    printf("Errors: %d / %d\n", error, m);
    printf("Success rate: %.2f%%\n", (1 - ((float)error / m)) * 100);
    printf("Total processing time: %.3f seconds\n", tdiff);
    printf("Average time per image: %.3f ms\n", (tdiff * 1000) / m);
    printf("========================================\n\n");

    fclose(label_file);
}