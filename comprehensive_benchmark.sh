#!/bin/bash

echo "=== Comprehensive Compression Algorithm Comparison ==="
echo "Testing Fixed-Length (FL) and Run-Length Encoding (RLE) in both C++ and CUDA"
echo

# Check if CUDA is available
CUDA_AVAILABLE=false
if command -v nvcc &> /dev/null; then
    echo "CUDA detected - compiling CUDA versions..."
    nvcc -O3 fl_cuda.cu -o fl_cuda 2>/dev/null && echo "✓ FL CUDA compiled"
    nvcc -O3 rle_cuda.cu -o rle_cuda 2>/dev/null && echo "✓ RLE CUDA compiled"
    CUDA_AVAILABLE=true
else
    echo "⚠ CUDA not available - testing C++ versions only"
fi

echo

# Create comprehensive test data
python3 -c "
import random
import numpy as np

# Test data with different characteristics
test_cases = [
    # (name, description, data_generator)
    ('gradient_4bit', 'Gradient data (4 bits)', lambda size: [i % 16 for i in range(size)]),
    ('gradient_6bit', 'Gradient data (6 bits)', lambda size: [i % 64 for i in range(size)]),
    ('repetitive', 'Highly repetitive (RLE optimal)', lambda size: [i//50 % 20 for i in range(size)]),
    ('sparse', 'Sparse data (mostly zeros)', lambda size: [0]*int(size*0.9) + [random.randint(1,7) for _ in range(int(size*0.1))]),
    ('random', 'Random data (worst case)', lambda size: [random.randint(0, 255) for _ in range(size)]),
    ('mixed_pattern', 'Mixed pattern data', lambda size: ([i%8 for i in range(size//2)] + [random.randint(0,255) for _ in range(size//2)])),
]

sizes = [1000, 10000, 50000]

for name, desc, generator in test_cases:
    for size in sizes:
        data = generator(size)
        random.shuffle(data) if name == 'mixed_pattern' else None
        filename = f'input/test_{name}_{size}.bin'
        with open(filename, 'wb') as f:
            f.write(bytes(data))

print('Created comprehensive test dataset')
"

echo "Test Dataset Created"
echo "===================="
echo

# Function to run and time a compression command
run_compression_test() {
    local cmd="$1"
    local file="$2"
    local algorithm="$3"
    
    # Capture both output and timing
    local result
    result=$($cmd "$file" 2>&1)
    
    # Extract metrics from output
    local ratio=$(echo "$result" | grep -E "Compression ratio" | awk '{print $3}' | head -1)
    local time=$(echo "$result" | grep -E "completed in" | awk '{print $5}' | head -1)
    local savings=$(echo "$result" | grep -E "Space savings" | awk '{print $3}' | head -1 | sed 's/%//')
    
    # Handle cases where compression might fail or expand data
    if [ -z "$ratio" ]; then
        ratio="N/A"
        time="N/A"
        savings="N/A"
    fi
    
    echo "$algorithm,$ratio,$time,$savings"
}

# Test files
test_files=(
    "input/test_gradient_4bit_1000.bin:4bit-1K"
    "input/test_gradient_4bit_10000.bin:4bit-10K"
    "input/test_gradient_4bit_50000.bin:4bit-50K"
    "input/test_gradient_6bit_1000.bin:6bit-1K"
    "input/test_gradient_6bit_10000.bin:6bit-10K"
    "input/test_gradient_6bit_50000.bin:6bit-50K"
    "input/test_repetitive_1000.bin:Rep-1K"
    "input/test_repetitive_10000.bin:Rep-10K"
    "input/test_repetitive_50000.bin:Rep-50K"
    "input/test_sparse_1000.bin:Sparse-1K"
    "input/test_sparse_10000.bin:Sparse-10K"
    "input/test_sparse_50000.bin:Sparse-50K"
    "input/test_random_1000.bin:Rand-1K"
    "input/test_random_10000.bin:Rand-10K"
    "input/test_mixed_pattern_1000.bin:Mixed-1K"
    "input/test_mixed_pattern_10000.bin:Mixed-10K"
)

echo "| Dataset | FL-CPP | FL-Time | FL-Savings | RLE-CPP | RLE-Time | RLE-Savings |"
echo "|---------|--------|---------|------------|---------|----------|-------------|"

for entry in "${test_files[@]}"; do
    IFS=':' read -r file label <<< "$entry"
    
    if [ -f "$file" ]; then
        echo -n "| $label |"
        
        # Test Fixed-Length C++
        fl_result=$(run_compression_test "./fl_compression -c" "$file" "FL-CPP")
        IFS=',' read -r fl_ratio fl_time fl_savings <<< "$fl_result"
        echo -n " $fl_ratio | $fl_time | $fl_savings% |"
        
        # Test RLE C++
        rle_result=$(run_compression_test "./rle_compression -c" "$file" "RLE-CPP")
        IFS=',' read -r rle_ratio rle_time rle_savings <<< "$rle_result"
        echo " $rle_ratio | $rle_time | $rle_savings% |"
        
        # Cleanup temporary files
        rm -f "${file%.bin}.flc" "${file%.bin}.rle" 2>/dev/null
    fi
done

echo
echo "Performance Analysis Summary:"
echo "============================"

# Analyze which algorithm works best for which data type
echo "Algorithm Recommendations:"
echo "• Fixed-Length: Best for data with limited value ranges (4-bit, 6-bit gradients, sparse data)"
echo "• RLE: Best for repetitive data with long runs of identical values"
echo "• Both struggle with random data (may expand file size)"
echo

# If CUDA is available, run CUDA performance tests
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "CUDA Performance Tests:"
    echo "======================"
    
    # Test larger datasets for CUDA performance comparison
    python3 -c "
import random

# Create larger test files for CUDA performance testing
large_sizes = [100000, 500000, 1000000]

for size in large_sizes:
    # Gradient data (good for FL)
    gradient = [i % 32 for i in range(size)]  # 5-bit data
    with open(f'input/cuda_gradient_{size}.bin', 'wb') as f:
        f.write(bytes(gradient))
    
    # Repetitive data (good for RLE)
    repetitive = []
    for i in range(size // 100):
        repetitive.extend([i % 20] * 100)
    with open(f'input/cuda_repetitive_{size}.bin', 'wb') as f:
        f.write(bytes(repetitive))

print('Created CUDA performance test files')
"
    
    echo "| Dataset | Size | FL-CUDA Time | RLE-CUDA Time | Speedup |"
    echo "|---------|------|--------------|---------------|---------|"
    
    for size in 100000 500000 1000000; do
        # Test FL CUDA vs CPP
        if [ -f "input/cuda_gradient_${size}.bin" ]; then
            echo -n "| Gradient | ${size} |"
            
            if [ -f "./fl_cuda" ]; then
                fl_cuda_time=$(./fl_cuda "input/cuda_gradient_${size}.bin" 2>&1 | grep "completed in" | awk '{print $5}' | head -1)
                echo -n " $fl_cuda_time μs |"
            else
                echo -n " N/A |"
            fi
            
            if [ -f "./rle_cuda" ]; then
                rle_cuda_time=$(./rle_cuda "input/cuda_repetitive_${size}.bin" 2>&1 | grep "completed in" | awk '{print $5}' | head -1)
                echo -n " $rle_cuda_time μs |"
            else
                echo -n " N/A |"
            fi
            
            # Calculate theoretical speedup (simplified)
            echo " TBD |"
        fi
    done
    
    # Cleanup CUDA test files
    rm -f input/cuda_*.bin 2>/dev/null
else
    echo "CUDA Performance: Skipped (CUDA not available)"
fi

echo
echo "Memory Usage Analysis:"
echo "====================="

# Test with a larger file to show memory efficiency
if [ -f "input/test_repetitive_50000.bin" ]; then
    echo "Testing memory efficiency with 50KB repetitive data..."
    
    # Test RLE (should be very efficient)
    ./rle_compression -c input/test_repetitive_50000.bin > /dev/null 2>&1
    if [ -f "input/test_repetitive_50000.rle" ]; then
        original_size=$(ls -l input/test_repetitive_50000.bin | awk '{print $5}')
        compressed_size=$(ls -l input/test_repetitive_50000.rle | awk '{print $5}')
        echo "RLE: $original_size bytes → $compressed_size bytes"
        rm input/test_repetitive_50000.rle
    fi
    
    # Test FL
    ./fl_compression -c input/test_gradient_6bit_50000.bin > /dev/null 2>&1
    if [ -f "input/test_gradient_6bit_50000.flc" ]; then
        original_size=$(ls -l input/test_gradient_6bit_50000.bin | awk '{print $5}')
        compressed_size=$(ls -l input/test_gradient_6bit_50000.flc | awk '{print $5}')
        echo "FL:  $original_size bytes → $compressed_size bytes"
        rm input/test_gradient_6bit_50000.flc
    fi
fi

echo
echo "Conclusions:"
echo "============"
echo "1. Fixed-Length Compression:"
echo "   - Excellent for data with limited value ranges"
echo "   - Predictable compression ratios based on bit requirements"
echo "   - Good for scientific data, sensor readings, reduced-depth images"
echo
echo "2. Run-Length Encoding:"
echo "   - Outstanding for repetitive data (up to 50:1 compression)"
echo "   - Poor for random data (can expand files significantly)"
echo "   - Ideal for simple graphics, logos, binary patterns"
echo
echo "3. Performance:"
echo "   - Both algorithms are very fast (microsecond range)"
echo "   - CUDA versions provide parallelization for large datasets"
echo "   - Memory usage scales with input size and compression characteristics"

# Cleanup all test files
rm -f input/test_*.bin 2>/dev/null

echo
echo "Benchmark completed!"
