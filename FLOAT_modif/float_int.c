#include "float_int.h"
#include <stdio.h>

// Converte float para int32_t fixed-point com "frac_bits" bits fracionários
int32_t float_to_fixed(float f, int frac_bits) {
    int32_t int_part = (int32_t)f;
    float frac_part = f - int_part;
    int32_t frac_bits_val = (int32_t)(frac_part * (1 << frac_bits));
    return (int_part << frac_bits) | (frac_bits_val & ((1 << frac_bits) - 1));
}

// Converte int32_t fixed-point de volta para float
float fixed_to_float(int32_t fixed, int frac_bits) {
    int32_t int_part = fixed >> frac_bits;
    int32_t frac_val = fixed & ((1 << frac_bits) - 1);
    float frac_part = frac_val / (float)(1 << frac_bits);
    return int_part + frac_part;
}