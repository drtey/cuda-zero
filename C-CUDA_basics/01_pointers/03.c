#include <stdio.h>

int main() {
    int num = 10;
    float fnum = 3.14;
    void* vptr;

    vptr = &num;
    printf("Integer: %d\n", *(int*)vptr); // output: 10
    // vptr is a memory address "&num" but it is stored as a void pointer (no data type)
    // we can't derference a void pointer, so we cast it to an integer pointer to store the integer value at that memory address "(int*)vptr"
    // then we defrence it with the final asterisk "*" to get the value "*((int*)vptr)"

    vptr = &fnum;
    printf("Float: %.2f\n", *(float*)vptr); // output = 3.14

    return 0;
}

// void pointers are used we don't know the data type of the memory address
// fun fact: malloc() returns a void pointer but we see it as a pointer to a specific data type 
// after the cast (int*)malloc(4) or (float*)malloc(4) etc