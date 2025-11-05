#include "float_int.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main() {
    float test_values[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 123.456f, -789.012f, 0.125f, 0.25f, 0.75f};
    int num_test_values = sizeof(test_values) / sizeof(test_values[0]);
    int frac_bits = 12;
    float epsilon = 0.001f; // Tolerance for float comparison

    printf("Running tests for float_int conversion...\n");

    for (int i = 0; i < num_test_values; i++) {
        float original_float = test_values[i];
        int32_t fixed_point = float_to_fixed(original_float, frac_bits);
        float converted_back_float = fixed_to_float(fixed_point, frac_bits);

        printf("Original: %f, Fixed: 0x%08X, Converted Back: %f\n", original_float, fixed_point, converted_back_float);

        // Assert that the converted back value is close to the original
        if (fabs(original_float - converted_back_float) > epsilon) {
            printf("Test failed for value: %f\n", original_float);
            return 1; // Indicate failure
        }
    }

    printf("All tests passed!\n");
    return 0;
}
