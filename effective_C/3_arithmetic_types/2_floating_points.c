#include <stdio.h>
#include <math.h>
#include <float.h>

/*
 * Understanding Floating-Point Numbers in C
 * ---------------------------------------
 * Floating-point numbers are used to represent real numbers in computers.
 * They consist of three parts:
 * 1. Sign bit (determines if number is positive or negative)
 * 2. Exponent (determines the magnitude of the number)
 * 3. Significand/Mantissa (determines the precision)
 *
 * C provides three standard floating-point types:
 * - float: typically 32 bits
 * - double: typically 64 bits (more precision than float)
 * - long double: extended precision (implementation dependent)
 */

/* 
 * Function to demonstrate floating-point classifications
 * Shows how different types of floating-point values are handled
 */
const char *show_classification(double x) {
    switch(fpclassify(x)) {
        case FP_INFINITE:  return "Infinite";   // Value is infinity (±∞)
        case FP_NAN:       return "NaN";        // Not a Number
        case FP_NORMAL:    return "Normal";     // Normal floating-point number
        case FP_SUBNORMAL: return "Subnormal";  // Very small number with reduced precision
        case FP_ZERO:      return "Zero";       // Zero value (can be +0 or -0)
        default:           return "Unknown";
    }
}

/*
 * Demonstrates precision limitations and special values
 * Shows common pitfalls and important concepts in floating-point arithmetic
 */
void demonstrate_floating_point_behavior(void) {
    printf("\n=== Floating-Point Behavior Demonstration ===\n");

    /* 
     * Different ways to represent floating-point constants
     * Note: Decimal points are mandatory in these examples
     */
    float f = 10.0F;    // Float constant (suffix F)
    double d = 10.0;    // Double constant (no suffix)
    long double ld = 10.0L;  // Long double constant (suffix L)

    /* Demonstrating precision limitations */
    double precise = 0.1;
    printf("0.1 stored as double: %.20f\n", precise);
    /* 
     * This will show that 0.1 cannot be exactly represented
     * in binary floating-point format
     */

    /* Special values demonstration */
    double inf = 1.0/0.0;  // Creates positive infinity
    double neg_inf = -1.0/0.0;  // Creates negative infinity
    double nan = 0.0/0.0;  // Creates NaN

    printf("\nSpecial value classifications:\n");
    printf("Infinity: %s\n", show_classification(inf));
    printf("Negative Infinity: %s\n", show_classification(neg_inf));
    printf("NaN: %s\n", show_classification(nan));

    /* Demonstrating subnormal numbers */
    double tiny = DBL_MIN/2.0;  // Creates a subnormal number
    printf("Subnormal number: %s\n", show_classification(tiny));
}

/*
 * Demonstrates floating-point arithmetic properties
 * Shows how floating-point math differs from real number math
 */
void demonstrate_floating_point_arithmetic(void) {
    printf("\n=== Floating-Point Arithmetic Properties ===\n");

    /* Association property does not hold */
    double a = 0.1;
    double b = 0.2;
    double c = 0.3;

    double result1 = (a + b) + c;
    double result2 = a + (b + c);

    printf("(0.1 + 0.2) + 0.3 = %.20f\n", result1);
    printf("0.1 + (0.2 + 0.3) = %.20f\n", result2);
    /* 
     * These results may be slightly different due to
     * rounding errors and non-associativity
     */

    /* Demonstrating precision in different types */
    float float_val = 1.23456789f;
    double double_val = 1.23456789;
    printf("\nPrecision comparison:\n");
    printf("Float: %.8f\n", float_val);
    printf("Double: %.8f\n", double_val);
}

int main() {
    /* 
     * Main demonstration of floating-point concepts
     * Shows practical examples of working with floating-point numbers
     */
    demonstrate_floating_point_behavior();
    demonstrate_floating_point_arithmetic();

    /* Demonstrate IEC 60559 limits */
    printf("\n=== Floating-Point Limits ===\n");
    printf("Double Epsilon: %e\n", DBL_EPSILON);
    printf("Double Min: %e\n", DBL_MIN);
    printf("Double Max: %e\n", DBL_MAX);

    return 0;
}