/**
 * @file pool_fixed.c
 * @brief Fixed-point implementation of pooling layers for LeNet CNN
 * 
 * This file contains the fixed-point (16.16 format) implementation of two 
 * pooling layers using 2x2 max pooling operation.
 */

#include <stdio.h>
#include "lenet_cnn_fixed.h"
#include "fixed_point.h"


short max(short a, short b){
  if(a > b){
    return a;
  }
  else{
    return b;
  }
}

/// @brief Pool1 with fixed-point arithmetic
/// @param conv1_output entries of conv1 layer
/// @param pool1_output entries of pool1 layer
void Pool1_24x24x20_2x2x20_2_0_fixed(
    short conv1_output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
    short pool1_output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH])
{
    unsigned short i,j,k,x,y;
    short max_val = 0;
    
    for(i = 0; i < CONV1_NBOUTPUT; i++){
		for(j = 0; j < CONV1_HEIGHT; j = j + POOL1_DIM){
			for(k = 0; k < CONV1_WIDTH; k = k + POOL1_DIM){
			
			max_val = 0;
				for(x = 0; x < POOL1_DIM; x++){
					for(y = 0; y < POOL1_DIM; y++){
						max_val = max(max_val, conv1_output[i][j+x][k+y]);
					}
				}
                          pool1_output[i][j>>1][k>>1] = max_val;
			}
		}
	}
    
    
}

/// @brief Pool2 with fixed-point arithmetic
/// @param conv2_output entries of conv2 layer
/// @param pool2_output entries of pool2 layer
void Pool2_8x8x40_2x2x40_2_0_fixed(
    short conv2_output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
    short pool2_output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH])
{

    unsigned short i,j,k,x,y;
	short max_val = 0;
	for(i = 0; i < CONV2_NBOUTPUT; i++){
		for(j = 0; j < CONV2_HEIGHT; j = j + POOL2_DIM){
			for(k = 0; k < CONV2_WIDTH; k = k + POOL2_DIM){

			max_val = 0;
				for(x = 0; x < POOL2_DIM; x++){
					for(y = 0; y < POOL2_DIM; y++){

						max_val = max(max_val, conv2_output[i][j+x][k+y]);
					}
				}
		        pool2_output[i][j>>1][k>>1] = max_val;
			}
		}
    }
}
