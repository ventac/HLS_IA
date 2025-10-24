#include <stdint.h>
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

int main() {
    float f = 182373.11234159;

    int frac_bits = 18;  // Aqui você controla a posição da vírgula
    int32_t fixed = float_to_fixed(f, frac_bits);
    float back = fixed_to_float(fixed, frac_bits);

    printf("Float original: %f\n", f);
    printf("Fixed-point (frac bits = %d): 0x%08X\n", frac_bits, fixed);
    printf("Convertido de volta: %f\n", back);

    return 0;
}