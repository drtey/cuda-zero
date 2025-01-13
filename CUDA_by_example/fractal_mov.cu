#include "book.h"
#include "cpu_anim.h"

#define DIM 1000

struct cuComplex {
    float r;
    float i;

    __device__ cuComplex(float a, float b): r(a), i(b) { 
    }

    __device__ float magnitude2() {
        return r * r + i * i;
    }

    __device__ cuComplex operator*(cuComplex& c) {
        return cuComplex(r * c.r - i * c.i, i * c.r + r * c.i);
    }

    __device__ cuComplex operator+(cuComplex& c) {
        return cuComplex(r + c.r, i + c.i);
    }
};

__device__ int julia(int x, int y, float time) {
    const float scale = 1.5f;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(cosf(time) * 0.5f, sinf(time) * 0.5f);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000) {
            return i;
        }
    }
    return 0;
}

__global__ void kernel(unsigned char *ptr, float time) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;
    
    int isJulia = julia(x, y, time);

    ptr[offset*4 + 0] = (isJulia * 7) % 256;
    ptr[offset*4 + 1] = (isJulia * 13) % 256; 
    ptr[offset*4 + 2] = (isJulia * 23) % 256; 
    ptr[offset*4 + 3] = 255;
}

struct DataBlock {
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;
};

void cleanup(DataBlock *d) {
    cudaFree(d->dev_bitmap);
}

void generateFrame(DataBlock *d, int ticks) {
    static float time = 0.0f;
    dim3 grid(DIM, DIM);
    kernel<<<grid, 1>>>(d->dev_bitmap, time);
    
    time += 0.02f * sinf(ticks * 0.01f);
    
    HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost));
}

int main(void) {
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    HANDLE_ERROR(cudaMalloc((void**) &data.dev_bitmap, bitmap.image_size()));
    bitmap.anim_and_exit((void (*)(void*, int))generateFrame, (void (*) (void*))cleanup);
}
