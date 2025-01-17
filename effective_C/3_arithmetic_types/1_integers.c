#include <stdio.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>

/* Function prototypes */
bool is_addition_safe(unsigned int a, unsigned int b);
void demonstrate_number_systems(void);
void demonstrate_integer_limits(void);
void demonstrate_safe_operations(void);
void demonstrate_bit_precise(void);

/* Safe absolute value macro with overflow protection */
#define SAFE_ABS(i, min) ((i) >= 0 ? (i) : ((i) == (min) ? (min) : -(i)))

int main() {
    /* Demonstrate different aspects of integer handling in C */
    demonstrate_number_systems();
    demonstrate_integer_limits();
    demonstrate_safe_operations();
    demonstrate_bit_precise();
    
    return 0;
}

void demonstrate_number_systems(void) {
    printf("\n=== Number System Representations ===\n");
    
    /* Same value (125) represented in different number systems */
    int decimal_value = 125;        // Base 10
    int hex_value = 0x7D;          // Base 16
    int octal_value = 0175;        // Base 8
    int binary_value = 0b1111101;  // Base 2 (C23)
    
    printf("The value 125 represented in different systems:\n");
    printf("Decimal (base 10): %d\n", decimal_value);
    printf("Hexadecimal (base 16): 0x%X\n", hex_value);
    printf("Octal (base 8): 0%o\n", octal_value);
    printf("Binary (base 2): 0b%s\n", "1111101");
    
    /* Demonstrate type suffixes */
    unsigned long long big_number = 18446744073709551615ULL;
    printf("\nLarge number with ULL suffix: %llu\n", big_number);
}

void demonstrate_integer_limits(void) {
    printf("\n=== Integer Limits and Behavior ===\n");
    
    /* Signed integer behavior */
    signed int signed_max = INT_MAX;
    signed int signed_min = INT_MIN;
    
    printf("Signed integer limits:\n");
    printf("Maximum value: %d\n", signed_max);
    printf("Minimum value: %d\n", signed_min);
    
    /* Demonstrate signed overflow (undefined behavior!) */
    printf("\nDemonstrating signed overflow (undefined behavior):\n");
    printf("Before: %d\n", signed_max);
    signed_max++;  // This is undefined behavior
    printf("After increment: %d\n", signed_max);
    
    /* Unsigned integer behavior */
    unsigned int unsigned_max = UINT_MAX;
    printf("\nUnsigned integer wraparound (well-defined behavior):\n");
    printf("Maximum unsigned value: %u\n", unsigned_max);
    printf("After increment: %u\n", unsigned_max + 1);  // Will be 0
}

void demonstrate_safe_operations(void) {
    printf("\n=== Safe Integer Operations ===\n");
    
    /* Demonstrate safe absolute value computation */
    int normal_value = -42;
    int edge_case = INT_MIN;
    
    printf("Safe absolute value examples:\n");
    printf("Regular case |%d| = %d\n", 
           normal_value, SAFE_ABS(normal_value, INT_MIN));
    printf("Edge case |%d| = %d\n", 
           edge_case, SAFE_ABS(edge_case, INT_MIN));
    
    /* Demonstrate safe addition check */
    unsigned int a = UINT_MAX - 2;
    unsigned int b = 5;
    
    printf("\nSafe addition check:\n");
    printf("Checking if %u + %u is safe: %s\n", 
           a, b, is_addition_safe(a, b) ? "Yes" : "No");
}

void demonstrate_bit_precise(void) {
    printf("\n=== Bit-Precise Types (C23) ===\n");
    
    /* Note: This section requires C23 support */
    #ifdef __STDC_VERSION__
    #if __STDC_VERSION__ >= 202311L
        /* Bit-precise integer examples would go here */
        printf("Bit-precise types are supported\n");
    #else
        printf("Bit-precise types require C23\n");
    #endif
    #endif
}

bool is_addition_safe(unsigned int a, unsigned int b) {
    return b <= UINT_MAX - a;
}