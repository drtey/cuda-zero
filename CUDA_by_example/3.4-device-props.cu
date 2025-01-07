#include "book.h"

int main(void) {
    cudaDeviceProp prop;
    int dev;

    HANDLE_ERROR( cudaGetDevice(&dev));
    printf("ID current CUDA device: %d\n", dev);

    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 8;
    prop.minor = 9;
    HANDLE_ERROR( cudaChooseDevice(&dev, &prop));
    printf( "ID of CUDA device closest to revision 8.9: %d\n", dev );
    HANDLE_ERROR( cudaSetDevice( dev ) );

}