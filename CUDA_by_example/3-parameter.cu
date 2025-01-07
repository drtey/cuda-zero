#include <iostream>
#include "book.h"

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main(void) {
    int c;
    int *dev_c;
    
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int))); // dev_c contains the GPU memory address
    add<<<1,1>>>(2, 7, dev_c);  // the operation is executed on the GPU and the result is stored in the address dev_c
    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost)); // returns the result of the operation to the host (CPU)
    
    printf("Hello, World!\n");
    printf("2 + 7 = %d\n", c);
    
    HANDLE_ERROR(cudaFree(dev_c));
    return 0;
}