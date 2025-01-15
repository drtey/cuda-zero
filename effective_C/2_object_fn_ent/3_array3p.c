#include <stdio.h>

/* First, we declare our functions with identical signatures */
void fn() {
    puts("index 0");
}

void fn1() {
    puts("index 1");
}

void fn2() {
    puts("index 2");
}

int main(int argc, char *argv[]) {
    /* Declare an array of 3 function pointers that return nothing (void)
       and take no parameters () */
    void (*function_array[3])() = {fn, fn1, fn2};
    
    /* We expect the index as a program argument.
       For safety, verify that we received the correct number of arguments */
    if (argc != 2) {
        printf("Usage: %s <index (0-2)>\n", argv[0]);
        return 1;
    }
    
    /* Convert argument string to number */
    int index = atoi(argv[1]);
    
    /* Verify that the index is within valid range */
    if (index >= 0 && index < 3) {
        /* Invoke the corresponding function using the index */
        function_array[index]();
    } else {
        printf("Error: index must be between 0 and 2\n");
        return 1;
    }
    
    return 0;
}
/* 
./program 0  fn()
./program 1  fn1()
./program 2  fn2()
*/