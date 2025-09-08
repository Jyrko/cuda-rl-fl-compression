#!/bin/bash

echo "=== Simple Dataset Generator ==="
echo "Creating basic test datasets for CUDA compression testing"

mkdir -p input

# Generate simple test files using Python
python3 -c "
import random
import os

print('Creating simple binary datasets...')

# Set random seed for reproducibility
random.seed(42)

def create_file(name, data):
    with open(f'input/{name}', 'wb') as f:
        f.write(bytes(data))
    print(f'Created input/{name} ({len(data)} bytes)')

# 1. Repetitive data (good for RLE)
print('1. Creating repetitive data...')
repetitive_data = []
for i in range(1000):
    value = i // 50  # Each value repeats 50 times
    repetitive_data.append(value % 256)
create_file('repetitive_high_1000.bin', repetitive_data)

# Scale up
repetitive_large = []
for i in range(50000):
    value = i // 100  # Each value repeats 100 times  
    repetitive_large.append(value % 256)
create_file('repetitive_high_50000.bin', repetitive_large)

# 2. Gradient data (good for Fixed-Length)
print('2. Creating gradient data...')
gradient_4bit = [i % 16 for i in range(1000)]  # 4-bit values
create_file('gradient_4bit_1000.bin', gradient_4bit)

gradient_4bit_large = [i % 16 for i in range(50000)]
create_file('gradient_4bit_50000.bin', gradient_4bit_large)

gradient_6bit = [i % 64 for i in range(1000)]  # 6-bit values
create_file('gradient_6bit_1000.bin', gradient_6bit)

# 3. Sparse data (mostly zeros)
print('3. Creating sparse data...')
sparse_data = [0] * 900 + [random.randint(1, 7) for _ in range(100)]
random.shuffle(sparse_data)
create_file('sparse_3bit_1000.bin', sparse_data)

sparse_large = [0] * 45000 + [random.randint(1, 31) for _ in range(5000)]
random.shuffle(sparse_large)
create_file('sparse_5bit_50000.bin', sparse_large)

# 4. Binary pattern data  
print('4. Creating binary pattern data...')
binary_data = [random.choice([0, 1]) for _ in range(1000)]
create_file('binary_pattern_1000.bin', binary_data)

binary_large = [random.choice([0, 1]) for _ in range(10000)]
create_file('binary_pattern_10000.bin', binary_large)

# 5. Mixed pattern data
print('5. Creating mixed pattern data...')
mixed_data = ([i % 8 for i in range(500)] + 
              [random.randint(0, 255) for _ in range(500)])
random.shuffle(mixed_data)
create_file('mixed_pattern_1000.bin', mixed_data)

# 6. Random data (worst case)
print('6. Creating random data...')
random_data = [random.randint(0, 255) for _ in range(1000)]
create_file('random_uniform_1000.bin', random_data)

# 7. Performance datasets
print('7. Creating performance datasets...')
perf_repetitive = [i // 1000 % 50 for i in range(1000000)]
create_file('perf_repetitive_1000000.bin', perf_repetitive)

perf_gradient = [i % 32 for i in range(1000000)]
create_file('perf_gradient_1000000.bin', perf_gradient)

perf_sparse = [0] * 950000 + [random.randint(1, 15) for _ in range(50000)]
random.shuffle(perf_sparse)
create_file('perf_sparse_1000000.bin', perf_sparse)

print('\\nDataset generation completed!')
print('Created simple datasets optimized for compression testing.')
"

echo ""
echo "Dataset generation completed!"
echo "Files created in input/ directory:"
ls -la input/*.bin | head -15
