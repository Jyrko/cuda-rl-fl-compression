#!/bin/bash

echo "=== RLE vs Fixed-Length Compression Comparison ==="
echo

# Test both algorithms on various data types
test_files=(
    "input/rle_repetitive.bin"
    "input/rle_large_runs.bin" 
    "input/rle_mixed.bin"
    "input/pattern_4bit.bin"
    "input/gradient_6bit.bin"
    "input/sparse_3bit.bin"
)

echo "| File | Size | RLE Ratio | RLE Time | FL Ratio | FL Time |"
echo "|------|------|-----------|----------|----------|---------|"

for file in "${test_files[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        filesize=$(ls -l "$file" | awk '{print $5}')
        
        echo -n "| $filename | ${filesize}B |"
        
        # Test RLE
        rle_output=$(./rle_compression -c "$file" 2>/dev/null | grep -E "(Compression ratio|completed in)")
        rle_ratio=$(echo "$rle_output" | grep "Compression ratio" | awk '{print $3}')
        rle_time=$(echo "$rle_output" | grep "completed in" | awk '{print $5}')
        
        echo -n " $rle_ratio | ${rle_time}μs |"
        
        # Test Fixed-Length
        fl_output=$(./fl_compression -c "$file" 2>/dev/null | grep -E "(Compression ratio|Original size|Compressed size)")
        original_size=$(echo "$fl_output" | grep "Original size" | awk '{print $3}')
        compressed_size=$(echo "$fl_output" | grep "Compressed size" | awk '{print $3}')
        
        if [ ! -z "$original_size" ] && [ ! -z "$compressed_size" ] && [ "$compressed_size" -gt 0 ]; then
            fl_ratio=$(echo "scale=3; $original_size / $compressed_size" | bc 2>/dev/null || echo "1.0")
        else
            fl_ratio="N/A"
        fi
        
        echo " $fl_ratio | N/A |"
        
        # Cleanup
        rm -f "${file%.bin}.rle" "${file%.bin}.flc" 2>/dev/null
    fi
done

echo
echo "=== Performance on Different Data Sizes ==="
echo

# Create files of different sizes for performance testing
python3 -c "
import random

sizes = [1000, 10000, 100000, 1000000]
for size in sizes:
    # Repetitive data
    data = []
    for i in range(size // 10):
        data.extend([i % 50] * 10)
    with open(f'input/perf_rep_{size}.bin', 'wb') as f:
        f.write(bytes(data))
    
    # Random data  
    data = [random.randint(0, 255) for _ in range(size)]
    with open(f'input/perf_rand_{size}.bin', 'wb') as f:
        f.write(bytes(data))

print('Created performance test files')
"

echo "| Size | Type | RLE Ratio | RLE Time (μs) |"
echo "|------|------|-----------|---------------|"

for size in 1000 10000 100000 1000000; do
    for type in "rep" "rand"; do
        file="input/perf_${type}_${size}.bin"
        if [ -f "$file" ]; then
            echo -n "| ${size} | $type |"
            
            output=$(./rle_compression -c "$file" 2>/dev/null | grep -E "(Compression ratio|completed in)")
            ratio=$(echo "$output" | grep "Compression ratio" | awk '{print $3}')
            time=$(echo "$output" | grep "completed in" | awk '{print $5}')
            
            echo " $ratio | $time |"
            
            # Cleanup
            rm -f "${file%.bin}.rle" 2>/dev/null
        fi
    done
done

echo
echo "=== Memory Usage Analysis ==="
echo

# Test with the largest file to show scalability
largest_file="input/perf_rep_1000000.bin"
if [ -f "$largest_file" ]; then
    echo "Testing RLE on 1MB repetitive data..."
    ./rle_compression -c "$largest_file"
    
    # Show file sizes
    echo
    echo "File size comparison:"
    ls -lh input/perf_rep_1000000.* | awk '{print $5 "\t" $9}'
    
    # Test decompression
    echo
    echo "Testing decompression..."
    ./rle_compression -d input/perf_rep_1000000.rle
    
    # Verify correctness
    echo
    echo -n "Verification: "
    if cmp -s input/perf_rep_1000000.bin input/perf_rep_1000000_decompressed.bin; then
        echo "PASSED - Files are identical"
    else
        echo "FAILED - Files differ"
    fi
    
    # Cleanup
    rm -f input/perf_rep_1000000.rle input/perf_rep_1000000_decompressed.bin
fi

# Cleanup performance test files
rm -f input/perf_*.bin

echo
echo "=== Conclusion ==="
echo "RLE is best for:"
echo "- Data with long runs of identical values"
echo "- Simple images, logos, text documents"
echo "- Binary data with repetitive patterns"
echo
echo "Fixed-Length is best for:"
echo "- Data with limited value ranges"
echo "- Scientific data, sensor readings"
echo "- Images with reduced bit depth"
echo
echo "RLE is worst for:"
echo "- Random data (can expand file size significantly)"
echo "- Highly compressed formats (JPEG, PNG)"
echo "- Encrypted or highly entropic data"
