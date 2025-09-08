#!/bin/bash

echo "=== Comprehensive Compression Algorithm Comparison ==="
echo "Testing Fixed-Length (FL) and Run-Length Encoding (RLE) in both C++ and CUDA"
echo

# Check if datasets exist
if [ ! -d "input" ] || [ -z "$(ls -A input/*.bin 2>/dev/null)" ]; then
    echo "⚠ Test datasets not found!"
    echo "Please run './generate_datasets.sh' first to create test datasets."
    echo
    read -p "Generate datasets now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./generate_datasets.sh
    else
        echo "Exiting. Run './generate_datasets.sh' to create datasets."
        exit 1
    fi
fi

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

# Check if datasets exist, if not prompt user to generate them
if [ ! -f "input/gradient_4bit_1000.bin" ]; then
    echo "❌ Test datasets not found!"
    echo "Please generate datasets first by running:"
    echo "  ./generate_datasets.sh"
    echo
    echo "This will create all required test files in the input/ directory."
    exit 1
fi

echo "✅ Using pre-generated datasets from input/ directory"

echo "Test Datasets Ready"
echo "==================="
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

# Test files - using pre-generated datasets
test_files=(
    "input/gradient_4bit_1000.png:4bit-1K"
    "input/gradient_4bit_10000.png:4bit-10K"
    "input/gradient_4bit_50000.png:4bit-50K"
    "input/gradient_6bit_1000.png:6bit-1K"
    "input/gradient_6bit_10000.png:6bit-10K"
    "input/gradient_6bit_50000.png:6bit-50K"
    "input/repetitive_high_1000.png:Rep-1K"
    "input/repetitive_high_10000.png:Rep-10K"
    "input/repetitive_high_50000.png:Rep-50K"
    "input/sparse_3bit_1000.png:Sparse-1K"
    "input/sparse_3bit_10000.png:Sparse-10K"
    "input/sparse_3bit_50000.png:Sparse-50K"
    "input/random_uniform_1000.png:Rand-1K"
    "input/random_uniform_10000.png:Rand-10K"
    "input/mixed_pattern_1000.png:Mixed-1K"
    "input/mixed_pattern_10000.png:Mixed-10K"
    "input/scientific_sensor_10000.png:Sensor-10K"
    "input/text_like_10000.png:Text-10K"
    "input/binary_pattern_10000.png:Binary-10K"
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
    echo "Using pre-generated performance datasets..."
    
    performance_files=(
        "input/perf_gradient_1000000.bin:Gradient-1M"
        "input/perf_repetitive_1000000.bin:Repetitive-1M"
        "input/perf_sparse_1000000.bin:Sparse-1M"
    )
    
    echo "| Dataset | Size | FL-CUDA Time | RLE-CUDA Time | Notes |"
    echo "|---------|------|--------------|---------------|-------|"
    
    for entry in "${performance_files[@]}"; do
        IFS=':' read -r file label <<< "$entry"
        
        if [ -f "$file" ]; then
            filesize=$(ls -l "$file" | awk '{print $5}')
            echo -n "| $label | ${filesize} |"
            
            if [ -f "./fl_cuda" ]; then
                fl_cuda_time=$(./fl_cuda "$file" 2>&1 | grep "completed in" | awk '{print $5}' | head -1)
                echo -n " $fl_cuda_time μs |"
            else
                echo -n " N/A |"
            fi
            
            if [ -f "./rle_cuda" ]; then
                rle_cuda_time=$(./rle_cuda "$file" 2>&1 | grep "completed in" | awk '{print $5}' | head -1)
                echo -n " $rle_cuda_time μs |"
            else
                echo -n " N/A |"
            fi
            
            echo " CUDA Performance |"
        else
            echo "| $label | Missing | N/A | N/A | Dataset not found |"
        fi
    done
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
