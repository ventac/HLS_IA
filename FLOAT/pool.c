// pool.c
// Implémentation des couches de pooling utilisées par lenet_cnn_float.c
// Max-pooling 2x2, stride 2, no padding
#include <stdio.h>
#include "lenet_cnn_float.h" // contient les définitions de dimensions (CONV1_* , POOL1_*, ...)

/**
 * Pool1
 * Input  : conv1_output  [CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH]  (ici 20 x 24 x 24)
 * Output : pool1_output  [POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]  (ici 20 x 12 x 12)
 */
void Pool1_24x24x20_2x2x20_2_0(float conv1_output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
                                float pool1_output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH])
{
    short ch, oy, ox;
    for (ch = 0; ch < CONV1_NBOUTPUT; ch++) {
        for (oy = 0; oy < POOL1_HEIGHT; oy++) {
            for (ox = 0; ox < POOL1_WIDTH; ox++) {
                // coordonnées du coin supérieur gauche de la fenêtre 2x2 dans l'entrée
                short in_y = oy * 2;
                short in_x = ox * 2;

                // max pooling 2x2
                float m = conv1_output[ch][in_y][in_x];

                // élément (in_y, in_x+1)
                if (conv1_output[ch][in_y][in_x + 1] > m)
                    m = conv1_output[ch][in_y][in_x + 1];

                // élément (in_y+1, in_x)
                if (conv1_output[ch][in_y + 1][in_x] > m)
                    m = conv1_output[ch][in_y + 1][in_x];

                // élément (in_y+1, in_x+1)
                if (conv1_output[ch][in_y + 1][in_x + 1] > m)
                    m = conv1_output[ch][in_y + 1][in_x + 1];

                pool1_output[ch][oy][ox] = m;
            }
        }
    }
}

/**
 * Pool2
 * Input  : conv2_output  [CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]  (ici 40 x 8 x 8)
 * Output : pool2_output  [POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]  (ici 40 x 4 x 4)
 */
void Pool2_8x8x40_2x2x40_2_0(float conv2_output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
                              float pool2_output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH])
{
    short ch, oy, ox;
    for (ch = 0; ch < CONV2_NBOUTPUT; ch++) {
        for (oy = 0; oy < POOL2_HEIGHT; oy++) {
            for (ox = 0; ox < POOL2_WIDTH; ox++) {
                short in_y = oy * 2;
                short in_x = ox * 2;

                float m = conv2_output[ch][in_y][in_x];

                if (conv2_output[ch][in_y][in_x + 1] > m)
                    m = conv2_output[ch][in_y][in_x + 1];
                if (conv2_output[ch][in_y + 1][in_x] > m)
                    m = conv2_output[ch][in_y + 1][in_x];
                if (conv2_output[ch][in_y + 1][in_x + 1] > m)
                    m = conv2_output[ch][in_y + 1][in_x + 1];

                pool2_output[ch][oy][ox] = m;
            }
        }
    }
}
