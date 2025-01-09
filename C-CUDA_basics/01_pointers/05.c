#include <stdio.h>

int main() {
    int arr[] = {12, 24, 36, 48, 60};

    int* ptr = arr; // ptr points to the first elment of arr (default in C)

    printf("position 1: %d\n", *ptr); // 12

    for (int i = 0; i < 5; i++) {
        printf("%d\n", *ptr);
        printf("%p\n", ptr);
        ptr++;
    }

    // notice that the pointer is incremented by 4 bytes (size of int = 4 bytes * 8 bits/bytes = 32 bits = int32) each time. 
    // ptrs are 64 bits in size (8 bytes). 2**32 = 4,294,967,296 which is too small given how much memory we typically have.
    // arrays are layed out in memory in a contiguous manner (one after the other rather than at random locations in the memory grid)
}