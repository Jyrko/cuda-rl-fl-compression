#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>

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

struct CudaCompressedFrame {
    unsigned char* data;
    int compressed_size;
    int bits_per_value;
    int frame_size;
    int original_size;
};

struct CudaFLResult {
    std::vector<CudaCompressedFrame> frames;
    size_t original_size;
    size_t compressed_size;
};

// Warp-level reduction to find maximum value in a frame
__device__ __forceinline__ unsigned char warpReduceMax(unsigned char val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Block-level reduction to find maximum value
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

// Kernel to find required bits for each frame
__global__ void findRequiredBitsKernel(const unsigned char* input, int* bits_required,
                                      int total_size, int frame_size) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;
    int start_pos = frame_idx * frame_size;
    int end_pos = min(start_pos + frame_size, total_size);
    
    if (start_pos >= total_size) return;
    
    unsigned char max_val = 0;
    
    // Each thread processes multiple elements if needed
    for (int i = start_pos + tid; i < end_pos; i += blockDim.x) {
        max_val = max(max_val, input[i]);
    }
    
    // Find maximum across all threads in block
    max_val = blockReduceMax(max_val);
    
    // Calculate required bits
    if (threadIdx.x == 0) {
        int bits = 0;
        if (max_val == 0) {
            bits = 1;
        } else {
            unsigned char temp = max_val;
            while (temp > 0) {
                temp >>= 1;
                bits++;
            }
        }
        bits_required[frame_idx] = bits;
    }
}

// Simple sequential bit packing kernel 
__global__ void packBitsKernel(const unsigned char* input, unsigned char* output,
                              const int* bits_required, const int* output_offsets,
                              int total_size, int frame_size) {
    int frame_idx = blockIdx.x;
    
    if (threadIdx.x != 0) return; // Only use one thread per block for simplicity
    
    int start_pos = frame_idx * frame_size;
    int end_pos = min(start_pos + frame_size, total_size);
    int frame_elements = end_pos - start_pos;
    
    if (start_pos >= total_size || frame_elements <= 0) return;
    
    int bits_per_value = bits_required[frame_idx];
    int output_offset = output_offsets[frame_idx];
    
    // Initialize output bytes to zero
    int packed_bytes = (frame_elements * bits_per_value + 7) / 8;
    for (int i = 0; i < packed_bytes; i++) {
        output[output_offset + i] = 0;
    }
    
    // Pack bits sequentially
    int current_bit = 0;
    for (int i = 0; i < frame_elements; i++) {
        unsigned char value = input[start_pos + i] & ((1 << bits_per_value) - 1);
        
        for (int bit = 0; bit < bits_per_value; bit++) {
            int byte_idx = current_bit / 8;
            int bit_idx = current_bit % 8;
            
            unsigned char bit_val = (value >> (bits_per_value - 1 - bit)) & 1;
            if (bit_val) {
                output[output_offset + byte_idx] |= (1 << (7 - bit_idx));
            }
            current_bit++;
        }
    }
}

// Kernel to unpack bits for decompression
__global__ void unpackBitsKernel(const unsigned char* packed_data, unsigned char* output,
                                const int* bits_per_frame, const int* frame_sizes,
                                const int* input_offsets, const int* output_offsets,
                                int num_frames) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (frame_idx >= num_frames) return;
    
    int bits_per_value = bits_per_frame[frame_idx];
    int frame_size = frame_sizes[frame_idx];
    int input_offset = input_offsets[frame_idx];
    int output_offset = output_offsets[frame_idx];
    
    // Each thread unpacks multiple values
    for (int i = tid; i < frame_size; i += blockDim.x) {
        unsigned char value = 0;
        int bit_start = i * bits_per_value;
        
        for (int bit = 0; bit < bits_per_value; bit++) {
            int global_bit_pos = bit_start + bit;
            int byte_idx = global_bit_pos / 8;
            int bit_idx = global_bit_pos % 8;
            
            // Fix: Use correct bit indexing (MSB first)
            unsigned char bit_val = (packed_data[input_offset + byte_idx] >> (7 - bit_idx)) & 1;
            value = (value << 1) | bit_val;  // Build value MSB first
        }
        
        output[output_offset + i] = value;
    }
}

class CudaFixedLengthCompressor {
private:
    int frame_size;
    
public:
    CudaFixedLengthCompressor(int frame_size = 256) : frame_size(frame_size) {}
    
    CudaFLResult compress(const std::vector<unsigned char>& input) {
        CudaFLResult result;
        result.original_size = input.size();
        
        if (input.empty()) {
            result.compressed_size = 0;
            return result;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int total_size = input.size();
        int num_frames = (total_size + frame_size - 1) / frame_size;
        
        // Device memory allocation
        unsigned char *d_input, *d_packed_output;
        int *d_bits_required, *d_output_offsets;
        
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, total_size * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_bits_required, num_frames * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output_offsets, num_frames * sizeof(int)));
        
        // Copy input to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data(), total_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Launch kernel to find required bits for each frame
        findRequiredBitsKernel<<<num_frames, BLOCK_SIZE>>>(d_input, d_bits_required, total_size, frame_size);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Copy bits required back to host to calculate output offsets
        std::vector<int> bits_required(num_frames);
        CHECK_CUDA_ERROR(cudaMemcpy(bits_required.data(), d_bits_required, num_frames * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Calculate output offsets and total compressed size
        std::vector<int> output_offsets(num_frames);
        std::vector<int> frame_compressed_sizes(num_frames);
        int total_compressed_size = 0;
        
        for (int i = 0; i < num_frames; i++) {
            output_offsets[i] = total_compressed_size;
            int current_frame_size = min(frame_size, total_size - i * frame_size);
            int compressed_bytes = (current_frame_size * bits_required[i] + 7) / 8;
            frame_compressed_sizes[i] = compressed_bytes;
            total_compressed_size += compressed_bytes;
        }
        
        // Allocate output buffer
        CHECK_CUDA_ERROR(cudaMalloc(&d_packed_output, total_compressed_size * sizeof(unsigned char)));
        
        // Copy offsets to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_output_offsets, output_offsets.data(), num_frames * sizeof(int), cudaMemcpyHostToDevice));
        
        // Launch bit packing kernel
        packBitsKernel<<<num_frames, BLOCK_SIZE>>>(d_input, d_packed_output, d_bits_required, 
                                                  d_output_offsets, total_size, frame_size);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Copy compressed data back to host
        std::vector<unsigned char> compressed_data(total_compressed_size);
        CHECK_CUDA_ERROR(cudaMemcpy(compressed_data.data(), d_packed_output, total_compressed_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        // Create result structure
        int data_offset = 0;
        for (int i = 0; i < num_frames; i++) {
            CudaCompressedFrame frame;
            frame.bits_per_value = bits_required[i];
            frame.frame_size = min(frame_size, total_size - i * frame_size);
            frame.original_size = frame.frame_size;
            frame.compressed_size = frame_compressed_sizes[i];
            frame.data = new unsigned char[frame.compressed_size];
            
            memcpy(frame.data, compressed_data.data() + data_offset, frame.compressed_size);
            data_offset += frame.compressed_size;
            
            result.frames.push_back(frame);
            
            std::cout << "Frame " << i << ": " << frame.frame_size << " -> " 
                      << frame.compressed_size << " bytes (" << frame.bits_per_value << " bits/value)" << std::endl;
        }
        
        result.compressed_size = total_compressed_size;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "CUDA Fixed-Length Compression completed in " << duration.count() << " microseconds" << std::endl;
        
        // Cleanup device memory
        CHECK_CUDA_ERROR(cudaFree(d_input));
        CHECK_CUDA_ERROR(cudaFree(d_packed_output));
        CHECK_CUDA_ERROR(cudaFree(d_bits_required));
        CHECK_CUDA_ERROR(cudaFree(d_output_offsets));
        
        return result;
    }
    
    std::vector<unsigned char> decompress(const CudaFLResult& compressed) {
        if (compressed.frames.empty()) {
            return {};
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int num_frames = compressed.frames.size();
        std::vector<unsigned char> result(compressed.original_size);
        
        // Prepare data for GPU
        std::vector<int> bits_per_frame(num_frames);
        std::vector<int> frame_sizes(num_frames);
        std::vector<int> input_offsets(num_frames);
        std::vector<int> output_offsets(num_frames);
        
        int total_compressed_size = 0;
        int current_output_offset = 0;
        
        for (int i = 0; i < num_frames; i++) {
            bits_per_frame[i] = compressed.frames[i].bits_per_value;
            frame_sizes[i] = compressed.frames[i].frame_size;
            input_offsets[i] = total_compressed_size;
            output_offsets[i] = current_output_offset;
            
            total_compressed_size += compressed.frames[i].compressed_size;
            current_output_offset += compressed.frames[i].frame_size;
        }
        
        // Concatenate all compressed data
        std::vector<unsigned char> all_compressed_data(total_compressed_size);
        int offset = 0;
        for (const auto& frame : compressed.frames) {
            memcpy(all_compressed_data.data() + offset, frame.data, frame.compressed_size);
            offset += frame.compressed_size;
        }
        
        // Device memory allocation
        unsigned char *d_compressed_data, *d_output;
        int *d_bits_per_frame, *d_frame_sizes, *d_input_offsets, *d_output_offsets;
        
        CHECK_CUDA_ERROR(cudaMalloc(&d_compressed_data, total_compressed_size * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, compressed.original_size * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_bits_per_frame, num_frames * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_frame_sizes, num_frames * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_input_offsets, num_frames * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output_offsets, num_frames * sizeof(int)));
        
        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_compressed_data, all_compressed_data.data(), total_compressed_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_bits_per_frame, bits_per_frame.data(), num_frames * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_frame_sizes, frame_sizes.data(), num_frames * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_input_offsets, input_offsets.data(), num_frames * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_output_offsets, output_offsets.data(), num_frames * sizeof(int), cudaMemcpyHostToDevice));
        
        // Launch decompression kernel
        unpackBitsKernel<<<num_frames, BLOCK_SIZE>>>(d_compressed_data, d_output, d_bits_per_frame,
                                                    d_frame_sizes, d_input_offsets, d_output_offsets, num_frames);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Copy result back to host
        CHECK_CUDA_ERROR(cudaMemcpy(result.data(), d_output, compressed.original_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "CUDA Fixed-Length Decompression completed in " << duration.count() << " microseconds" << std::endl;
        
        // Cleanup device memory
        CHECK_CUDA_ERROR(cudaFree(d_compressed_data));
        CHECK_CUDA_ERROR(cudaFree(d_output));
        CHECK_CUDA_ERROR(cudaFree(d_bits_per_frame));
        CHECK_CUDA_ERROR(cudaFree(d_frame_sizes));
        CHECK_CUDA_ERROR(cudaFree(d_input_offsets));
        CHECK_CUDA_ERROR(cudaFree(d_output_offsets));
        
        return result;
    }
    
    void printCompressionStats(const CudaFLResult& result) {
        double compression_ratio = (double)result.original_size / result.compressed_size;
        double space_savings = (1.0 - (double)result.compressed_size / result.original_size) * 100;
        
        std::cout << "\nCUDA Fixed-Length Compression Statistics:" << std::endl;
        std::cout << "Original size: " << result.original_size << " bytes" << std::endl;
        std::cout << "Compressed size: " << result.compressed_size << " bytes" << std::endl;
        std::cout << "Number of frames: " << result.frames.size() << std::endl;
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
    
    void cleanup(CudaFLResult& result) {
        for (auto& frame : result.frames) {
            if (frame.data) {
                delete[] frame.data;
                frame.data = nullptr;
            }
        }
        result.frames.clear();
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <input_file> [frame_size]" << std::endl;
        return 1;
    }
    
    int frame_size = (argc > 2) ? std::atoi(argv[2]) : 256;
    
    CudaFixedLengthCompressor compressor(frame_size);
    
    // Load test data
    auto data = compressor.loadImageFile(argv[1]);
    if (data.empty()) {
        return 1;
    }
    
    std::cout << "Starting CUDA Fixed-Length compression (frame size: " << frame_size << ")..." << std::endl;
    
    // Compress
    auto compressed = compressor.compress(data);
    compressor.printCompressionStats(compressed);
    
    // Decompress
    std::cout << "\nStarting CUDA Fixed-Length decompression..." << std::endl;
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
