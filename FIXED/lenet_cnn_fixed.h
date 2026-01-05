// lenet_cnn_fixed.h
//#ifndef LENET_CNN_FIXED_H
//#define LENET_CNN_FIXED_H

//#include "lenet_cnn_float.h"  // for dimension constants (plus utilise)
//#include "fixed_point.h"

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

// Partie fixed point
#define FIXED_POINT 8
#define FLOAT2SHORT(x) ((short) ((x) * (1 << FIXED_POINT)))
#define SHORT2FLOAT(x) (((float)(x)) / (1 << FIXED_POINT))
#define RELU_F(x) (x > 0)? x : 0

// Fonctions d'utilite
void ReadPgmFile(char *filename, unsigned char *pix); 
void NormalizeImg(unsigned char *input, short *output, short width, short height); 

// New
void Conv1_28x28x1_5x5x20_1_0_fixed(
    short input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
    short kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
    short bias[CONV1_NBOUTPUT],
    short output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH]);

void Pool1_24x24x20_2x2x20_2_0_fixed(
    short input[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
    short output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]);

void Conv2_12x12x20_5x5x40_1_0_fixed(
    short input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],
    short kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
    short bias[CONV2_NBOUTPUT],
    short output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]);

void Pool2_8x8x40_2x2x40_2_0_fixed(
    short input[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
    short output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]);

void Fc1_40_400_fixed(
    short input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    short kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
    short bias[FC1_NBOUTPUT],
    short output[FC1_NBOUTPUT]);

void Fc2_400_10_fixed(
    short input[FC1_NBOUTPUT],
    short kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
    short bias[FC2_NBOUTPUT],
    short output[FC2_NBOUTPUT]);

void Softmax_fixed(short input[FC2_NBOUTPUT], float output[FC2_NBOUTPUT]);
