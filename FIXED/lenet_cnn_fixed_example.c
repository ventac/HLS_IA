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
#include <unistd.h>
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
    char hdf5_filename[256];
    /* Try several likely locations for the weights file and pick the first
       one that exists. This helps when running the binary from the FIXED
       directory while weights live in FLOAT/ or etc/. */
    const char *candidates[] = {
        "lenet_weights.weights.h5",
        "lenet_weights.hdf5",
        "../FLOAT/lenet_weights.weights.h5",
        "../FLOAT/lenet_weights.hdf5",
        "../FLOAT_modif/lenet_weights.weights.h5",
        "../etc/lenet_weights.hdf5",
        "etc/lenet_weights.hdf5",
        NULL
    };
    int ci = 0;
    while (candidates[ci]) {
        if (access(candidates[ci], F_OK) == 0) {
            strncpy(hdf5_filename, candidates[ci], sizeof(hdf5_filename)-1);
            hdf5_filename[sizeof(hdf5_filename)-1] = '\0';
            break;
        }
        ci++;
    }
    if (!candidates[ci]) {
        /* fallback to default name: program will fail later with clear message */
        strncpy(hdf5_filename, "lenet_weights.weights.h5", sizeof(hdf5_filename)-1);
        hdf5_filename[sizeof(hdf5_filename)-1] = '\0';
    }
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

    /* Clear screen only when stdout is an interactive terminal to avoid
       printing raw escape sequences when output is redirected or the
       terminal doesn't support ANSI codes. */
    if (isatty(STDOUT_FILENO)) {
        printf("\x1b[2J\x1b[H");
    }

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

        /* Intentionally do not clear here; frame will be drawn in one chunk
           later to avoid double-refresh and flicker. */

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

        /* Build a single frame buffer and write it in one operation to
           reduce flicker. We move the cursor to home but do not clear the
           entire screen; lines are padded so remnants from previous
           frames are overwritten. */
        if (isatty(STDOUT_FILENO))
        {
            /* Move cursor to home */
            printf("\x1b[H");
        }

        // Prepare frame in a buffer
        {
            char frame[4096];
            int pos = 0;
            const int LINE_PAD = 120; // pad width to overwrite previous content
            // Image filename header
            pos += snprintf(frame + pos, sizeof(frame) - pos, "%s\n", img_filename);
            pos += snprintf(frame + pos, sizeof(frame) - pos, "\nSoftmax output (Fixed-Point):\n");

            max = 0.0f;
            number = 0;
            for (k = 0; k < FC2_NBOUTPUT; k++)
            {
                float prob = fixed_to_float(SOFTMAX_OUTPUT_FIXED[k]);
                char prob_str[64];
                int n = snprintf(prob_str, sizeof(prob_str), " %d: %.2f%%", k, prob * 100.0f);
                // pad the line to ensure leftover chars are overwritten
                pos += snprintf(frame + pos, sizeof(frame) - pos, "%-*s\n", LINE_PAD, prob_str);
                if (prob > max)
                {
                    max = prob;
                    number = k;
                }
            }

            // Prediction line
            char pred_str[128];
            int pred_n = snprintf(pred_str, sizeof(pred_str), "\nPredicted: %d\tActual: %d", labels_legend[number], label);
            if (labels_legend[number] != label)
            {
                strncat(pred_str, " [ERROR]", sizeof(pred_str) - strlen(pred_str) - 1);
                error = error + 1;
            }
            else
            {
                strncat(pred_str, " [OK]", sizeof(pred_str) - strlen(pred_str) - 1);
            }
            pos += snprintf(frame + pos, sizeof(frame) - pos, "%-*s\n", LINE_PAD, pred_str);

            // Write the whole frame at once
            fwrite(frame, 1, pos, stdout);
            fflush(stdout);
        }

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