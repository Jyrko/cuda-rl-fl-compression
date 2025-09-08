#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

#define WARP_SIZE 32
#define BLOCK_SIZE 256

struct CudaRLEResult {
    unsigned char* symbols;
    int* counts;
    int* positions;
    int num_runs;
    size_t original_size;
    size_t compressed_size;
};

// Warp-level primitive functions
__device__ __forceinline__ unsigned int ballot_sync(unsigned int mask, int predicate) {
    return __ballot_sync(mask, predicate);
}

__device__ __forceinline__ int popc(unsigned int x) {
    return __popc(x);
}

__device__ __forceinline__ int ffs(unsigned int x) {
    return __ffs(x);
}

// Parallel RLE compression kernel using inter-warp communication
__global__ void rleCompressKernel(const unsigned char* input, int input_size,
                                 unsigned char* temp_symbols, int* temp_counts, 
                                 int* temp_positions, int* run_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    __shared__ int warp_run_counts[BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_offsets[BLOCK_SIZE / WARP_SIZE];
    
    if (tid >= input_size) return;
    
    unsigned char current_value = input[tid];
    unsigned char next_value = (tid + 1 < input_size) ? input[tid + 1] : current_value + 1;
    
    // Check if this position starts a new run
    bool is_run_start = (tid == 0) || (input[tid] != input[tid - 1]);
    
    // Check if this position ends a run
    bool is_run_end = (tid == input_size - 1) || (current_value != next_value);
    
    // Use warp-level primitives for efficient communication
    unsigned int start_mask = ballot_sync(0xFFFFFFFF, is_run_start);
    unsigned int end_mask = ballot_sync(0xFFFFFFFF, is_run_end);
    
    // Count runs in this warp
    int warp_runs = popc(start_mask);
    
    // Store warp run count
    if (lane_id == 0) {
        warp_run_counts[warp_id] = warp_runs;
    }
    
    __syncthreads();
    
    // Calculate prefix sum for warp offsets
    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        int offset = 0;
        for (int i = 0; i < threadIdx.x; i++) {
            offset += warp_run_counts[i];
        }
        warp_offsets[threadIdx.x] = offset;
    }
    
    __syncthreads();
    
    // Process runs within the warp
    if (is_run_start) {
        // Find position of this run start within the warp
        unsigned int mask_before = start_mask & ((1u << lane_id) - 1);
        int run_index_in_warp = popc(mask_before);
        int global_run_index = warp_offsets[warp_id] + run_index_in_warp;
        
        // Find the end of this run
        int run_length = 1;
        for (int i = tid + 1; i < input_size && input[i] == current_value; i++) {
            run_length++;
            if (run_length >= 255) break; // Max run length
        }
        
        // Store run information
        temp_symbols[global_run_index] = current_value;
        temp_counts[global_run_index] = run_length;
        temp_positions[global_run_index] = tid;
    }
    
    // Update total run count
    if (threadIdx.x == 0) {
        int block_total = 0;
        for (int i = 0; i < blockDim.x / WARP_SIZE; i++) {
            block_total += warp_run_counts[i];
        }
        atomicAdd(run_count, block_total);
    }
}

// Kernel to compact the results (remove gaps)
__global__ void compactResultsKernel(const unsigned char* temp_symbols, 
                                    const int* temp_counts,
                                    const int* temp_positions,
                                    unsigned char* final_symbols,
                                    int* final_counts,
                                    int total_runs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_runs) return;
    
    final_symbols[tid] = temp_symbols[tid];
    final_counts[tid] = temp_counts[tid];
}

// Parallel RLE decompression kernel
__global__ void rleDecompressKernel(const unsigned char* symbols, const int* counts,
                                   int num_runs, unsigned char* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_runs) return;
    
    // Calculate starting position for this run
    int start_pos = 0;
    for (int i = 0; i < tid; i++) {
        start_pos += counts[i];
    }
    
    unsigned char symbol = symbols[tid];
    int count = counts[tid];
    
    // Fill the output with this run
    for (int i = 0; i < count; i++) {
        output[start_pos + i] = symbol;
    }
}

// Optimized decompression with prefix sum
__global__ void calculatePrefixSum(const int* counts, int* prefix_sums, int num_runs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_runs) return;
    
    int sum = 0;
    for (int i = 0; i <= tid; i++) {
        sum += counts[i];
    }
    prefix_sums[tid] = sum;
}

__global__ void rleDecompressOptimizedKernel(const unsigned char* symbols, 
                                            const int* counts,
                                            const int* prefix_sums,
                                            int num_runs, 
                                            unsigned char* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_runs) return;
    
    int start_pos = (tid == 0) ? 0 : prefix_sums[tid - 1];
    int count = counts[tid];
    unsigned char symbol = symbols[tid];
    
    // Use multiple threads per run for large runs
    int threads_per_run = min(count, WARP_SIZE);
    int elements_per_thread = (count + threads_per_run - 1) / threads_per_run;
    
    for (int i = 0; i < elements_per_thread; i++) {
        int pos = start_pos + i;
        if (pos < start_pos + count) {
            output[pos] = symbol;
        }
    }
}

class CudaRunLengthCompressor {
private:
    int max_run_length;
    
public:
    CudaRunLengthCompressor(int max_run = 255) : max_run_length(max_run) {}
    
    CudaRLEResult compress(const std::vector<unsigned char>& input) {
        CudaRLEResult result = {0};
        result.original_size = input.size();
        
        if (input.empty()) {
            return result;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Device memory allocation
        unsigned char *d_input;
        unsigned char *d_temp_symbols, *d_final_symbols;
        int *d_temp_counts, *d_final_counts;
        int *d_temp_positions, *d_run_count;
        
        size_t input_size = input.size();
        size_t max_runs = input_size; // Worst case: every element is a run
        
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_temp_symbols, max_runs * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_temp_counts, max_runs * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_temp_positions, max_runs * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_run_count, sizeof(int)));
        
        // Copy input to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data(), input_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Initialize run count
        int zero = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(d_run_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
        
        // Launch compression kernel
        int block_size = BLOCK_SIZE;
        int grid_size = (input_size + block_size - 1) / block_size;
        
        rleCompressKernel<<<grid_size, block_size>>>(d_input, input_size,
                                                    d_temp_symbols, d_temp_counts,
                                                    d_temp_positions, d_run_count);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Get number of runs
        int num_runs;
        CHECK_CUDA_ERROR(cudaMemcpy(&num_runs, d_run_count, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Allocate final result arrays
        CHECK_CUDA_ERROR(cudaMalloc(&d_final_symbols, num_runs * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_final_counts, num_runs * sizeof(int)));
        
        // Compact results
        int compact_grid_size = (num_runs + block_size - 1) / block_size;
        compactResultsKernel<<<compact_grid_size, block_size>>>(d_temp_symbols, d_temp_counts,
                                                               d_temp_positions, d_final_symbols,
                                                               d_final_counts, num_runs);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Copy results back to host
        result.symbols = new unsigned char[num_runs];
        result.counts = new int[num_runs];
        result.num_runs = num_runs;
        
        CHECK_CUDA_ERROR(cudaMemcpy(result.symbols, d_final_symbols, num_runs * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(result.counts, d_final_counts, num_runs * sizeof(int), cudaMemcpyDeviceToHost));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        result.compressed_size = num_runs * (sizeof(unsigned char) + sizeof(int));
        
        std::cout << "CUDA RLE Compression completed in " << duration.count() << " microseconds" << std::endl;
        std::cout << "Found " << num_runs << " runs" << std::endl;
        
        // Cleanup device memory
        CHECK_CUDA_ERROR(cudaFree(d_input));
        CHECK_CUDA_ERROR(cudaFree(d_temp_symbols));
        CHECK_CUDA_ERROR(cudaFree(d_temp_counts));
        CHECK_CUDA_ERROR(cudaFree(d_temp_positions));
        CHECK_CUDA_ERROR(cudaFree(d_final_symbols));
        CHECK_CUDA_ERROR(cudaFree(d_final_counts));
        CHECK_CUDA_ERROR(cudaFree(d_run_count));
        
        return result;
    }
    
    std::vector<unsigned char> decompress(const CudaRLEResult& compressed) {
        std::vector<unsigned char> result(compressed.original_size);
        
        if (compressed.num_runs == 0) {
            return result;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Device memory allocation
        unsigned char *d_symbols, *d_output;
        int *d_counts, *d_prefix_sums;
        
        CHECK_CUDA_ERROR(cudaMalloc(&d_symbols, compressed.num_runs * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_counts, compressed.num_runs * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_prefix_sums, compressed.num_runs * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, compressed.original_size * sizeof(unsigned char)));
        
        // Copy compressed data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_symbols, compressed.symbols, compressed.num_runs * sizeof(unsigned char), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_counts, compressed.counts, compressed.num_runs * sizeof(int), cudaMemcpyHostToDevice));
        
        // Calculate prefix sums
        int block_size = BLOCK_SIZE;
        int grid_size = (compressed.num_runs + block_size - 1) / block_size;
        
        calculatePrefixSum<<<grid_size, block_size>>>(d_counts, d_prefix_sums, compressed.num_runs);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Launch decompression kernel
        rleDecompressOptimizedKernel<<<grid_size, block_size>>>(d_symbols, d_counts, d_prefix_sums,
                                                               compressed.num_runs, d_output);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Copy result back to host
        CHECK_CUDA_ERROR(cudaMemcpy(result.data(), d_output, compressed.original_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "CUDA RLE Decompression completed in " << duration.count() << " microseconds" << std::endl;
        
        // Cleanup device memory
        CHECK_CUDA_ERROR(cudaFree(d_symbols));
        CHECK_CUDA_ERROR(cudaFree(d_counts));
        CHECK_CUDA_ERROR(cudaFree(d_prefix_sums));
        CHECK_CUDA_ERROR(cudaFree(d_output));
        
        return result;
    }
    
    void printCompressionStats(const CudaRLEResult& result) {
        double compression_ratio = (double)result.original_size / result.compressed_size;
        double space_savings = (1.0 - (double)result.compressed_size / result.original_size) * 100;
        double avg_run_length = (double)result.original_size / result.num_runs;
        
        std::cout << "\nCUDA RLE Compression Statistics:" << std::endl;
        std::cout << "Original size: " << result.original_size << " bytes" << std::endl;
        std::cout << "Compressed size: " << result.compressed_size << " bytes" << std::endl;
        std::cout << "Number of runs: " << result.num_runs << std::endl;
        std::cout << "Average run length: " << avg_run_length << std::endl;
        std::cout << "Compression ratio: " << compression_ratio << ":1" << std::endl;
        std::cout << "Space savings: " << space_savings << "%" << std::endl;
    }
    
    std::vector<unsigned char> loadImageFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot open image file " << filename << std::endl;
            return {};
        }
        
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<unsigned char> data(file_size);
        file.read(reinterpret_cast<char*>(data.data()), file_size);
        
        std::cout << "Loaded image file: " << filename << " (" << file_size << " bytes)" << std::endl;
        return data;
    }
    
    void cleanup(CudaRLEResult& result) {
        if (result.symbols) {
            delete[] result.symbols;
            result.symbols = nullptr;
        }
        if (result.counts) {
            delete[] result.counts;
            result.counts = nullptr;
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }
    
    CudaRunLengthCompressor compressor;
    
    // Load test data
    auto data = compressor.loadImageFile(argv[1]);
    if (data.empty()) {
        return 1;
    }
    
    std::cout << "Starting CUDA RLE compression..." << std::endl;
    
    // Compress
    auto compressed = compressor.compress(data);
    compressor.printCompressionStats(compressed);
    
    // Decompress
    std::cout << "\nStarting CUDA RLE decompression..." << std::endl;
    auto decompressed = compressor.decompress(compressed);
    
    // Verify correctness
    bool is_correct = (data == decompressed);
    std::cout << "Decompression " << (is_correct ? "SUCCESSFUL" : "FAILED") << std::endl;
    
    if (!is_correct) {
        std::cout << "First difference at position: ";
        for (size_t i = 0; i < std::min(data.size(), decompressed.size()); i++) {
            if (data[i] != decompressed[i]) {
                std::cout << i << " (original: " << (int)data[i] 
                         << ", decompressed: " << (int)decompressed[i] << ")" << std::endl;
                break;
            }
        }
    }
    
    // Cleanup
    compressor.cleanup(compressed);
    
    return 0;
}
