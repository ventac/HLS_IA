#ifndef FIXED_POINT_H
#define FIXED_POINT_H

#include <stdint.h>
#include <limits.h>

// Fixed-point format: 16.16 (16 bits integer, 16 bits fractional)
typedef int32_t fixed16_16_t;

// Use 16 fractional bits to implement 16.16 fixed-point consistently
#define FIXED_POINT_SHIFT 16
#define FIXED_POINT_SCALE (1U << FIXED_POINT_SHIFT)

// Conversion functions
static inline fixed16_16_t float_to_fixed(float f) {
    // Convert with saturation to avoid overflow when casting
    double v = (double)f * (double)FIXED_POINT_SCALE;
    if (v > (double)INT32_MAX) return INT32_MAX;
    if (v < (double)INT32_MIN) return INT32_MIN;
    return (fixed16_16_t)(v);
}

static inline float fixed_to_float(fixed16_16_t fixed) {
    return (float)fixed / FIXED_POINT_SCALE;
}

// Basic arithmetic operations
static inline fixed16_16_t fixed_add(fixed16_16_t a, fixed16_16_t b) {
    return a + b;
}

static inline fixed16_16_t fixed_sub(fixed16_16_t a, fixed16_16_t b) {
    return a - b;
}

static inline fixed16_16_t fixed_mul(fixed16_16_t a, fixed16_16_t b) {
    // Multiply and shift back
    int64_t temp = (int64_t)a * (int64_t)b;
    // Apply rounding: add/subtract 0.5 before shifting
    int64_t round = (int64_t)1 << (FIXED_POINT_SHIFT - 1);
    if (temp >= 0) temp += round; else temp -= round;
    temp >>= FIXED_POINT_SHIFT;
    // Saturate to 32-bit fixed type range
    if (temp > INT32_MAX) return INT32_MAX;
    if (temp < INT32_MIN) return INT32_MIN;
    return (fixed16_16_t)temp;
}

static inline fixed16_16_t fixed_div(fixed16_16_t a, fixed16_16_t b) {
    // Shift up before division
    if (b == 0) {
        return (a >= 0) ? INT32_MAX : INT32_MIN; // avoid div-by-zero
    }
    int64_t temp = ((int64_t)a << FIXED_POINT_SHIFT);
    // Apply rounding: add/subtract |b|/2 before dividing to round toward nearest
    int64_t absb = (b == INT32_MIN) ? (int64_t)INT32_MAX + 1 : (b < 0 ? - (int64_t)b : (int64_t)b);
    int64_t round = absb >> 1;
    if ((temp >= 0 && b > 0) || (temp < 0 && b < 0)) temp += round; else temp -= round;
    int64_t res = temp / b;
    if (res > INT32_MAX) return INT32_MAX;
    if (res < INT32_MIN) return INT32_MIN;
    return (fixed16_16_t)res;
}

// Comparison operations
static inline int fixed_gt(fixed16_16_t a, fixed16_16_t b) {
    return a > b;
}

static inline fixed16_16_t fixed_max(fixed16_16_t a, fixed16_16_t b) {
    return (a > b) ? a : b;
}

static inline fixed16_16_t fixed_min(fixed16_16_t a, fixed16_16_t b) {
    return (a < b) ? a : b;
}

// ReLU activation (max(0, x))
static inline fixed16_16_t fixed_relu(fixed16_16_t x) {
    return (x > 0) ? x : 0;
}

// Constants
#define FIXED_ZERO 0
#define FIXED_ONE (1 << FIXED_POINT_SHIFT)  // 1.0 in fixed-point

// Approximation of exp(x) using Taylor series (for softmax)
// Limited range for stability
static inline fixed16_16_t fixed_exp_approx(fixed16_16_t x) {
    // For values too large or small, return approximation
    if (x > float_to_fixed(10.0f)) return float_to_fixed(22026.0f); // e^10
    if (x < float_to_fixed(-10.0f)) return 0;
    
    // Taylor series: e^x ≈ 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5!
    fixed16_16_t result = FIXED_ONE;  // 1
    fixed16_16_t term = x;             // x
    
    result = fixed_add(result, term);  // 1 + x
    
    term = fixed_mul(term, x);         // x²
    term = fixed_div(term, float_to_fixed(2.0f));  // x²/2!
    result = fixed_add(result, term);
    
    term = fixed_mul(term, x);         // x³/2!
    term = fixed_div(term, float_to_fixed(3.0f));  // x³/3!
    result = fixed_add(result, term);
    
    term = fixed_mul(term, x);         // x⁴/3!
    term = fixed_div(term, float_to_fixed(4.0f));  // x⁴/4!
    result = fixed_add(result, term);
    
    term = fixed_mul(term, x);         // x⁵/4!
    term = fixed_div(term, float_to_fixed(5.0f));  // x⁵/5!
    result = fixed_add(result, term);
    
    return result;
}

#endif // FIXED_POINT_H
