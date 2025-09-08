# CUDA Compression Algorithms Implementation

This directory contains both CPU and GPU implementations of two lossless compression algorithms:

## Algorithms Implemented

### 1. Fixed-Length Compression (`fl_compression.cpp` / `fl_cuda.cu`)
- **Principle**: Analyzes data frames to find minimum bits needed and removes unused high-order bits
- **Best for**: Data with limited value ranges (scientific data, sensor readings, reduced-depth images)
- **Compression**: Up to 87.5% space savings on optimal data

### 2. Run-Length Encoding (`rle_compression.cpp` / `rle_cuda.cu`)
- **Principle**: Counts consecutive identical values and stores value+count pairs
- **Best for**: Repetitive data with long runs of identical values
- **Compression**: Up to 98% space savings on highly repetitive data

## CUDA Implementation Features

### Inter-Warp Communication
Both CUDA implementations use advanced GPU programming techniques:

#### Fixed-Length CUDA (`fl_cuda.cu`)
```cuda
// Warp-level reduction to find maximum value
__device__ __forceinline__ unsigned char warpReduceMax(unsigned char val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Block-level reduction combining multiple warps
__device__ __forceinline__ unsigned char blockReduceMax(unsigned char val) {
    static __shared__ unsigned char shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceMax(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceMax(val);
    
    return val;
}
```

#### RLE CUDA (`rle_cuda.cu`)
```cuda
// Use ballot primitives for efficient run detection
unsigned int start_mask = ballot_sync(0xFFFFFFFF, is_run_start);
unsigned int end_mask = ballot_sync(0xFFFFFFFF, is_run_end);

// Count runs in this warp
int warp_runs = popc(start_mask);
```

### Parallel Processing Strategy

#### Fixed-Length Algorithm
1. **Frame Analysis**: Each thread block processes one frame
2. **Maximum Finding**: Warp-level reductions find max value per frame
3. **Bit Packing**: Parallel bit manipulation with shared memory
4. **Decompression**: Parallel unpacking of bit-packed data

#### RLE Algorithm
1. **Run Detection**: Warp-level ballot operations identify run boundaries
2. **Run Counting**: Inter-warp communication aggregates run counts
3. **Result Compaction**: Remove gaps in sparse result arrays
4. **Parallel Decompression**: Each thread handles one run

### Memory Optimization
- **Coalesced Access**: Memory access patterns optimized for GPU
- **Shared Memory**: Used for intermediate results and reduction operations
- **Atomic Operations**: Minimize conflicts in parallel bit packing
- **Prefix Sums**: Efficient position calculation for decompression

## Performance Characteristics

### Benchmark Results

| Data Type | Size | FL Compression | RLE Compression | Best Algorithm |
|-----------|------|----------------|-----------------|----------------|
| 4-bit Gradient | 50KB | 2:1 (50%) | 0.2:1 (-400%) | **Fixed-Length** |
| 6-bit Gradient | 50KB | 1.33:1 (25%) | 0.2:1 (-400%) | **Fixed-Length** |
| Repetitive | 50KB | 1.87:1 (47%) | 10:1 (90%) | **RLE** |
| Sparse (90% zeros) | 50KB | 6.62:1 (85%) | 2.23:1 (55%) | **Fixed-Length** |
| Random | 50KB | 1:1 (0%) | 0.2:1 (-398%) | **Neither** |

### When to Use Each Algorithm

#### Fixed-Length Compression
✅ **Excellent for:**
- Scientific sensor data
- Medical imaging with limited bit depth
- Embedded system data logs
- Pre-processed images

❌ **Poor for:**
- High-entropy data (photos, encrypted data)
- Already compressed formats
- Random noise

#### Run-Length Encoding
✅ **Excellent for:**
- Simple graphics and logos
- Binary documents and text
- Bitmap fonts
- Pattern-based data

❌ **Poor for:**
- Random or highly varying data
- Natural photographs
- Compressed audio/video

## Compilation and Usage

### C++ Versions
```bash
# Compile
g++ -O3 fl_compression.cpp -o fl_compression
g++ -O3 rle_compression.cpp -o rle_compression

# Usage
./fl_compression -c input.bin -f 256    # Fixed-length with frame size 256
./rle_compression -c input.bin -m 255   # RLE with max run length 255
```

### CUDA Versions (requires NVIDIA GPU + CUDA toolkit)
```bash
# Compile
nvcc -O3 fl_cuda.cu -o fl_cuda
nvcc -O3 rle_cuda.cu -o rle_cuda

# Usage
./fl_cuda input.bin 256        # Fixed-length with frame size 256
./rle_cuda input.bin           # RLE compression
```

### Comprehensive Benchmark
```bash
./comprehensive_benchmark.sh   # Tests both algorithms on various data types
```

## Technical Implementation Details

### Fixed-Length CUDA Kernels

1. **`findRequiredBitsKernel`**: Parallel maximum finding using warp reductions
2. **`packBitsKernel`**: Bit packing with atomic operations for thread safety
3. **`unpackBitsKernel`**: Parallel bit unpacking for decompression

### RLE CUDA Kernels

1. **`rleCompressKernel`**: Run detection using ballot sync primitives
2. **`compactResultsKernel`**: Remove gaps in sparse result arrays
3. **`rleDecompressKernel`**: Parallel run expansion
4. **`calculatePrefixSum`**: Position calculation for efficient decompression

### Memory Management
- Automatic device memory allocation/deallocation
- Error checking for all CUDA operations
- Host-device data transfer optimization
- Memory usage scales linearly with input size

## Future Optimizations

### Potential CUDA Improvements
1. **Cooperative Groups**: Use cooperative groups for more flexible thread coordination
2. **Unified Memory**: Simplify memory management with unified memory
3. **Multi-GPU**: Scale to multiple GPUs for very large datasets
4. **Streams**: Overlap computation and memory transfers
5. **Tensor Cores**: Explore tensor core usage for specific data patterns

### Algorithm Enhancements
1. **Adaptive Frame Size**: Dynamically adjust frame size based on data characteristics
2. **Hybrid Compression**: Combine FL and RLE based on data analysis
3. **Dictionary Encoding**: Add Huffman or arithmetic coding for further compression
4. **Entropy Analysis**: Pre-analyze data to choose optimal algorithm

## Educational Value

This implementation demonstrates:
- **GPU Programming**: Warp primitives, shared memory, atomic operations
- **Parallel Algorithms**: Reduction, prefix sum, load balancing
- **Memory Optimization**: Coalescing, bank conflicts, occupancy
- **Algorithm Analysis**: Time/space complexity, data dependencies
- **Performance Engineering**: Profiling, optimization strategies

## References

1. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
2. [Run-Length Encoding - Wikipedia](https://en.wikipedia.org/wiki/Run-length_encoding)
3. [GPU Gems - Parallel Prefix Sum](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
4. [Warp-Level Primitives](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions)
