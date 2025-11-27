/**
  ****************************************************************************
  * @file    lenet_cnn_fixed.h
  * @brief   Combined header: dimension macros + prototypes for float and fixed versions
  *          of the LeNet CNN functions. This merges the former `lenet_cnn_float.h`
  *          and `lenet_cnn_fixed.h` into a single header to avoid duplication.
  ****************************************************************************
  */

#ifndef LENET_CNN_FIXED_H
#define LENET_CNN_FIXED_H

#include "fixed_point.h"

/* Image dimensions and network layer parameters (from lenet_cnn_float.h) */
#define IMG_WIDTH	28
#define IMG_HEIGHT	28
#define IMG_DEPTH	1

#define CONV1_DIM	    5
#define CONV1_NBOUTPUT	20
#define CONV1_STRIDE	1
#define CONV1_PAD	    0
#define CONV1_WIDTH	    ( ( (IMG_WIDTH - CONV1_DIM + (2*CONV1_PAD) ) / CONV1_STRIDE ) + 1 )
#define CONV1_HEIGHT	( ( (IMG_HEIGHT - CONV1_DIM + (2*CONV1_PAD) ) / CONV1_STRIDE ) + 1 )

#define POOL1_DIM	    2
#define POOL1_NBOUTPUT	CONV1_NBOUTPUT
#define POOL1_STRIDE	2
#define POOL1_PAD	    0
#define POOL1_WIDTH	    ( ( (CONV1_WIDTH - POOL1_DIM + (2*POOL1_PAD) ) / POOL1_STRIDE ) + 1 )
#define POOL1_HEIGHT	( ( (CONV1_HEIGHT - POOL1_DIM + (2*POOL1_PAD) ) / POOL1_STRIDE ) + 1 )

#define CONV2_DIM	    5
#define CONV2_NBOUTPUT	40
#define CONV2_STRIDE	1
#define CONV2_PAD	    0
#define CONV2_WIDTH	    ( ( (POOL1_WIDTH - CONV2_DIM + (2*CONV2_PAD) ) / CONV2_STRIDE ) + 1 )
#define CONV2_HEIGHT	( ( (POOL1_HEIGHT - CONV2_DIM + (2*CONV2_PAD) ) / CONV2_STRIDE ) + 1 )

#define POOL2_DIM	    2
#define POOL2_NBOUTPUT	CONV2_NBOUTPUT
#define POOL2_STRIDE	2
#define POOL2_PAD	    0
#define POOL2_WIDTH	    ( ( (CONV2_WIDTH - POOL2_DIM + (2*POOL2_PAD) ) / POOL2_STRIDE ) + 1 )
#define POOL2_HEIGHT	( ( (CONV2_HEIGHT - POOL2_DIM + (2*POOL2_PAD) ) / POOL2_STRIDE ) + 1 )

#define FC1_NBOUTPUT	400

#define FC2_NBOUTPUT	10

void ReadPgmFile(char *filename, unsigned char *pix); 
void WritePgmFile(char *filename, float *pix, short width, short height); 
void ReadTestLabels(char *filename, short size); 
void RescaleImg(unsigned char *input, short width,short height, float *output, short new_width, short new_height); 
void NormalizeImg(unsigned char *input, float *output, short width, short height); 
void ReadConv1Weights(char *filename, char *datasetname, float weight[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM]); 
void ReadConv1Bias(char *filename, char *datasetname, float *bias); 
void ReadConv2Weights(char *filename, char *datasetname, float weight[CONV2_NBOUTPUT][CONV1_NBOUTPUT][CONV2_DIM][CONV2_DIM]); 
void ReadConv2Bias(char *filename, char *datasetname, float *bias); 
void ReadFc1Weights(char *filename, char *datasetname, float weight[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]); 
void ReadFc1Bias(char *filename, char *datasetname, float *bias); 
void ReadFc2Weights(char *filename, char *datasetname, float weight[FC2_NBOUTPUT][FC1_NBOUTPUT]); 
void ReadFc2Bias(char *filename, char *datasetname, float *bias); 
void WriteWeights(char *filename, short weight[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM]); 

/* ------------------------------------------------------------------------- */
/* Prototypes for fixed-point implementations (kept)                             */
/* ------------------------------------------------------------------------- */
void Conv1_28x28x1_5x5x20_1_0_fixed(
    fixed16_16_t input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    fixed16_16_t kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    fixed16_16_t bias[CONV1_NBOUTPUT],
    fixed16_16_t output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH]);

void Pool1_24x24x20_2x2x20_2_0_fixed(
    fixed16_16_t input[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
    fixed16_16_t output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]);

void Conv2_12x12x20_5x5x40_1_0_fixed(
    fixed16_16_t input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],
    fixed16_16_t kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    fixed16_16_t bias[CONV2_NBOUTPUT],
    fixed16_16_t output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]);

void Pool2_8x8x40_2x2x40_2_0_fixed(
    fixed16_16_t input[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
    fixed16_16_t output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]);

void Fc1_40_400_fixed(
    fixed16_16_t input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    fixed16_16_t kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    fixed16_16_t bias[FC1_NBOUTPUT],
    fixed16_16_t output[FC1_NBOUTPUT]);

void Fc2_400_10_fixed(
    fixed16_16_t input[FC1_NBOUTPUT],
    fixed16_16_t kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
    fixed16_16_t bias[FC2_NBOUTPUT],
    fixed16_16_t output[FC2_NBOUTPUT]);

void Softmax_fixed(fixed16_16_t input[FC2_NBOUTPUT], fixed16_16_t output[FC2_NBOUTPUT]);

#endif /* LENET_CNN_FIXED_H */