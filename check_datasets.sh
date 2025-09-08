#!/bin/bash

echo "=== Checking Dataset Status ==="

cd input

echo "Files with 0 bytes:"
ls -la *.bin | awk '$5 == 0 {print $9 " - " $5 " bytes"}'

echo ""
echo "Valid files (>0 bytes):"
ls -la *.bin | awk '$5 > 0 {print $9 " - " $5 " bytes"}' | head -10

echo ""
echo "Total dataset files:"
echo "Binary files: $(ls -1 *.bin 2>/dev/null | wc -l)"
echo "PNG files: $(ls -1 *.png 2>/dev/null | wc -l)"

# Test one that works
echo ""
echo "Testing with a valid file:"
if [ -f "sparse_5bit_50000.bin" ] && [ -s "sparse_5bit_50000.bin" ]; then
    echo "sparse_5bit_50000.bin exists and has data ($(stat -c%s sparse_5bit_50000.bin 2>/dev/null || stat -f%z sparse_5bit_50000.bin) bytes)"
    cd ..
    ./rle_cuda input/sparse_5bit_50000.bin
else
    echo "sparse_5bit_50000.bin is empty or missing"
    # Try another file
    for file in repetitive_high_1000.bin gradient_4bit_1000.bin binary_pattern_1000.bin; do
        if [ -f "$file" ] && [ -s "$file" ]; then
            echo "Using $file instead ($(stat -c%s $file 2>/dev/null || stat -f%z $file) bytes)"
            cd ..
            ./rle_cuda input/$file
            break
        fi
    done
fi
