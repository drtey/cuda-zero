#include <stdio.h>
#include <stdlib.h>

int main() {
    // init pointer to null
    int* ptr = NULL;
    printf("1. Initial ptr value: %p\n", (void*)ptr);

    // check for NULL before using
    if (ptr == NULL) {
        printf("2. ptr is NULL cannot deference\n");
    }

    // allocate memory
    ptr = malloc(sizeof(int));
    if (ptr == NULL) {
        printf("3. Memory allocation failed\n");
        return 1;
    }

    printf("4. After allocation ptr value: %p\n", (void*)ptr);

    *ptr = 42;
    printf("5. Value at ptr: %p\n", *ptr);

    // clean memory
    free(ptr);
    ptr = NULL;

    printf("6. After allocation ptr value: %p\n", (void*)ptr);  

    //Demonstrate safety of NULL check after tree
    if (ptr == NULL) {
        printf("7. ptr is null safely avoided after free \n");
    }

    return 0;
}