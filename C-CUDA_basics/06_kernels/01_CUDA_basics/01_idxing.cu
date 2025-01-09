#include <stdio.h>

// This is a CUDA kernel function that runs on the GPU
// The __global__ keyword means it can be called from CPU code
__global__ void whoami(void) {
    // Calculate unique block ID using a 3D addressing scheme
    // Think of it like finding an apartment in a city:
    int block_id =
        blockIdx.x +    // Position along a floor (like apartment number)
        blockIdx.y * gridDim.x +    // Floor number × apartments per floor
        blockIdx.z * gridDim.x * gridDim.y;   // Building number × (apartments per floor × floors)

    // Calculate the starting thread ID for this block
    // Like finding the first person's ID in an apartment
    int block_offset =
        block_id *      // Take the apartment number
        blockDim.x * blockDim.y * blockDim.z;  // Multiply by total people per apartment

    // Calculate position within the block (thread's local ID)
    // Like finding a person's position within their apartment
    int thread_offset =
        threadIdx.x +  // Position along width
        threadIdx.y * blockDim.x +  // Position along height × width
        threadIdx.z * blockDim.x * blockDim.y;  // Position along depth × (width × height)

    // Calculate global unique ID for this thread
    // Like assigning a unique number to each person in the entire city
    int id = block_offset + thread_offset;

    // Print thread information in a formatted way:
    // - Global ID (4 digits)
    // - Block coordinates (x,y,z) and calculated block_id
    // - Thread coordinates (x,y,z) and calculated thread_offset
    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

int main(int argc, char **argv) {
    // Define grid dimensions (how many blocks in each direction)
    const int b_x = 2, b_y = 3, b_z = 4;  // Creates a 2×3×4 grid of blocks
    
    // Define block dimensions (how many threads per block in each direction)
    const int t_x = 4, t_y = 4, t_z = 4;  // Creates a 4×4×4 cube of threads
    // Note: This gives us 64 threads per block, which will be organized as 2 warps
    // (A warp is a group of 32 threads that execute together)

    // Calculate total number of blocks and threads for reporting
    int blocks_per_grid = b_x * b_y * b_z;        // Total blocks = 2×3×4 = 24
    int threads_per_block = t_x * t_y * t_z;      // Total threads per block = 4×4×4 = 64
    
    // Print configuration summary
    printf("%d blocks/grid\n", blocks_per_grid);
    printf("%d threads/block\n", threads_per_block);
    printf("%d total threads\n", blocks_per_grid * threads_per_block);

    // Create 3D specifications for blocks and threads
    dim3 blocksPerGrid(b_x, b_y, b_z);      // Defines our 2×3×4 grid structure
    dim3 threadsPerBlock(t_x, t_y, t_z);    // Defines our 4×4×4 block structure

    // Launch the kernel with our specified configuration
    // The <<< >>> syntax is CUDA-specific for kernel launches
    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    
    // Wait for all GPU operations to complete before ending program
    cudaDeviceSynchronize();
}