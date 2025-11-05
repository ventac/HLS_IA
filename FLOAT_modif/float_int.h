#ifndef FLOAT_INT_H
#define FLOAT_INT_H

#include <stdint.h>

// Converte float para int32_t fixed-point com "frac_bits" bits fracionários
int32_t float_to_fixed(float f, int frac_bits);

// Converte int32_t fixed-point de volta para float
float fixed_to_float(int32_t fixed, int frac_bits);

#endif // FLOAT_INT_H