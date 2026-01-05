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

#include "lenet_cnn_fixed.h"
#include "weights.h"

void lenet_cnn_fixed(short input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH], 						
	       short 	output[FC2_NBOUTPUT]) 
	{
            short conv1_output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH];
            short pool1_output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH];
            short conv2_output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH];
            short pool2_output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
            short fc1_output[FC1_NBOUTPUT];
            short k, y, x;
            
            // Chaque fonction declaree dans son fichier .h
            Conv1_28x28x1_5x5x20_1_0_fixed(input, CONV1_KERNEL, CONV1_BIAS, conv1_output);
            Pool1_24x24x20_2x2x20_2_0_fixed(conv1_output, pool1_output);
            Conv2_12x12x20_5x5x40_1_0_fixed(pool1_output, CONV2_KERNEL, CONV2_BIAS, conv2_output);
            Pool2_8x8x40_2x2x40_2_0_fixed(conv2_output, pool2_output);
            Fc1_40_400_fixed(pool2_output, FC1_KERNEL, FC1_BIAS, fc1_output);
            Fc2_400_10_fixed(fc1_output, FC2_KERNEL, FC2_BIAS, output);
}


// INFO: Fixed version for reading weights.h
// Fixed version for reading weights.h
unsigned char REF_IMG[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
short INPUT_NORM_FIXED[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
// int32_t CONV1_KERNEL[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM];
// int32_t CONV1_BIAS[CONV1_NBOUTPUT];
// int32_t CONV2_KERNEL[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM];
// int32_t CONV2_BIAS[CONV2_NBOUTPUT];
// int32_t FC1_KERNEL[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
// int32_t FC1_BIAS[FC1_NBOUTPUT];
// int32_t FC2_KERNEL[FC2_NBOUTPUT][FC1_NBOUTPUT];
// int32_t FC2_BIAS[FC2_NBOUTPUT];
short FC2_OUTPUT_FIXED[FC2_NBOUTPUT];
//int32_t SOFTMAX_OUTPUT_FIXED[FC2_NBOUTPUT];
float SOFTMAX_OUTPUT[FC2_NBOUTPUT];




/**
 * @brief Main function deploying LeNet inference CNN on MNIST dataset using fixed-point arithmetic
 */
void main()
{
    short x, y, z, k, m;

    char *test_labels_filename = "mnist/t10k-labels-idx1-ubyte";
    
    FILE *label_file;
    int ret;
    unsigned char label, number = 0;
    unsigned int error;
    unsigned char labels_legend[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    char img_filename[120];
    char img_count[10];
    float max;
    struct timeval start, end;
    double tdiff;

    printf("\e[1;1H\e[2J");

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
        if (m < 10) strcat(img_filename, "0000");
        else if (m < 100) strcat(img_filename, "000");
        else if (m < 1000) strcat(img_filename, "00");
        else if (m < 10000) strcat(img_filename, "0");
        strcat(img_filename, img_count);
        strcat(img_filename, "].pgm");

        // Read and normalize image (float)
        ReadPgmFile(img_filename, (unsigned char *)REF_IMG);
        
        NormalizeImg((unsigned char *)REF_IMG, (short *)INPUT_NORM_FIXED, IMG_WIDTH, IMG_WIDTH);

        // Run fixed-point inference
        lenet_cnn_fixed(INPUT_NORM_FIXED,
                        // CONV1_KERNEL_FIXED,
                        // CONV1_BIAS_FIXED,
                        // CONV2_KERNEL_FIXED,
                        // CONV2_BIAS_FIXED,
                        // FC1_KERNEL_FIXED,
                        // FC1_BIAS_FIXED,
                        // FC2_KERNEL_FIXED,
                        // FC2_BIAS_FIXED,
                        FC2_OUTPUT_FIXED);

        // Apply softmax in fixed-point
        Softmax_fixed(FC2_OUTPUT_FIXED, SOFTMAX_OUTPUT);
        
        printf("\n\nSoftmax output : \n");
        max = 0;
        number = 0;
        
        for (k = 0; k < FC2_NBOUTPUT; k++) {
          printf("%.2f%% ", SOFTMAX_OUTPUT[k]*100);
          if (SOFTMAX_OUTPUT[k] > max) {
            max = SOFTMAX_OUTPUT[k];
            number = k;
          }
        }

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
