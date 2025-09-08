#!/bin/bash

echo "=== Compression Algorithm Dataset Generator ==="
echo "Creating test datasets for Fixed-Length and Run-Length Encoding algorithms"
echo

# Create input directory if it doesn't exist
mkdir -p input

# Generate comprehensive test datasets using Python
python3 -c "
import random
import numpy as np
import os

print('Generating test datasets...')

# Ensure reproducible results
random.seed(42)
np.random.seed(42)

# Test data characteristics and sizes
test_cases = [
    # (name, description, data_generator, sizes)
    ('gradient_4bit', 'Gradient data requiring 4 bits per value', 
     lambda size: [i % 16 for i in range(size)], 
     [1000, 10000, 50000, 100000]),
    
    ('gradient_6bit', 'Gradient data requiring 6 bits per value', 
     lambda size: [i % 64 for i in range(size)], 
     [1000, 10000, 50000, 100000]),
    
    ('gradient_8bit', 'Full 8-bit gradient data', 
     lambda size: [i % 256 for i in range(size)], 
     [1000, 10000, 50000]),
    
    ('repetitive_high', 'Highly repetitive data (optimal for RLE)', 
     lambda size: [i//50 % 20 for i in range(size)], 
     [1000, 10000, 50000, 100000, 500000]),
    
    ('repetitive_medium', 'Medium repetitive data', 
     lambda size: [i//10 % 50 for i in range(size)], 
     [1000, 10000, 50000]),
    
    ('sparse_3bit', 'Sparse data (90% zeros, max 7)', 
     lambda size: [0]*int(size*0.9) + [random.randint(1,7) for _ in range(int(size*0.1))], 
     [1000, 10000, 50000, 100000]),
    
    ('sparse_5bit', 'Sparse data (80% zeros, max 31)', 
     lambda size: [0]*int(size*0.8) + [random.randint(1,31) for _ in range(int(size*0.2))], 
     [1000, 10000, 50000]),
    
    ('random_uniform', 'Random uniform data (worst case)', 
     lambda size: [random.randint(0, 255) for _ in range(size)], 
     [1000, 10000, 50000]),
    
    ('random_limited', 'Random data with limited range (0-15)', 
     lambda size: [random.randint(0, 15) for _ in range(size)], 
     [1000, 10000, 50000]),
    
    ('mixed_pattern', 'Mixed repetitive and random sections', 
     lambda size: ([i%8 for i in range(size//2)] + [random.randint(0,255) for _ in range(size//2)]), 
     [1000, 10000, 50000]),
    
    ('text_like', 'Text-like data (ASCII printable)', 
     lambda size: [random.randint(32, 126) for _ in range(size)], 
     [1000, 10000, 50000]),
    
    ('binary_pattern', 'Binary pattern data (0s and 1s)', 
     lambda size: [random.choice([0, 1]) for _ in range(size)], 
     [1000, 10000, 50000]),
    
    ('scientific_sensor', 'Simulated sensor data (scaled to 8-bit)', 
     lambda size: [int((2048 + 1000 * np.sin(i/100) + random.gauss(0, 50)) % 4096 / 16) for i in range(size)], 
     [1000, 10000, 50000]),
]

# Performance test datasets (larger sizes)
performance_cases = [
    ('perf_repetitive', 'Performance test - repetitive data', 
     lambda size: [i//100 % 50 for i in range(size)], 
     [1000000, 5000000]),
    
    ('perf_gradient', 'Performance test - gradient data', 
     lambda size: [i % 32 for i in range(size)], 
     [1000000, 5000000]),
    
    ('perf_sparse', 'Performance test - sparse data', 
     lambda size: [0]*int(size*0.95) + [random.randint(1,15) for _ in range(int(size*0.05))], 
     [1000000, 2000000]),
]

# Create regular test datasets
total_files = 0
total_size = 0

print('Creating standard test datasets:')
for name, desc, generator, sizes in test_cases:
    print(f'  {name}: {desc}')
    for size in sizes:
        data = generator(size)
        
        # Shuffle mixed pattern data to make it more realistic
        if name == 'mixed_pattern':
            random.shuffle(data)
        
        # Ensure sparse data is properly distributed
        if 'sparse' in name:
            random.shuffle(data)
        
        filename = f'input/{name}_{size}.bin'
        with open(filename, 'wb') as f:
            f.write(bytes(data))
        
        file_size = len(data)
        total_files += 1
        total_size += file_size
        print(f'    Created {filename} ({file_size:,} bytes)')

print()
print('Creating performance test datasets:')
for name, desc, generator, sizes in performance_cases:
    print(f'  {name}: {desc}')
    for size in sizes:
        data = generator(size)
        
        filename = f'input/{name}_{size}.bin'
        with open(filename, 'wb') as f:
            f.write(bytes(data))
        
        file_size = len(data)
        total_files += 1
        total_size += file_size
        print(f'    Created {filename} ({file_size:,} bytes)')

print()
print('Creating special test cases:')

# Real-world simulation datasets
special_cases = [
    # Bitmap font simulation
    ('bitmap_font', lambda: [0]*800 + [255]*50 + [0]*150, 'Bitmap font pattern'),
    
    # Logo-like data
    ('logo_pattern', lambda: ([0]*100 + [255]*20 + [0]*100 + [128]*10)*5, 'Logo-like pattern'),
    
    # Medical imaging simulation (16-bit values in 8-bit space)
    ('medical_sim', lambda: [int(x) for x in np.random.normal(128, 30, 10000) if 0 <= x <= 255], 'Medical imaging simulation'),
    
    # Network packet simulation
    ('network_sim', lambda: [0]*500 + [random.randint(1,255) for _ in range(500)], 'Network packet simulation'),
]

for name, generator, desc in special_cases:
    data = generator()
    filename = f'input/{name}.bin'
    with open(filename, 'wb') as f:
        f.write(bytes(data))
    
    file_size = len(data)
    total_files += 1
    total_size += file_size
    print(f'  Created {filename} ({file_size:,} bytes) - {desc}')

print()
print(f'Dataset generation complete!')
print(f'Total files created: {total_files}')
print(f'Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)')
print()
print('Dataset categories:')
print('  • Gradient data: Tests Fixed-Length compression efficiency')
print('  • Repetitive data: Tests RLE compression efficiency') 
print('  • Sparse data: Tests both algorithms on data with many zeros')
print('  • Random data: Tests worst-case scenarios')
print('  • Performance data: Large datasets for speed testing')
print('  • Special cases: Real-world data simulations')
"

echo
echo "Dataset generation completed!"
echo "Files are stored in the 'input/' directory"
echo
echo "Usage:"
echo "  Run this script once to generate test datasets"
echo "  Then run ./comprehensive_benchmark.sh to test compression algorithms"
echo
echo "To regenerate datasets:"
echo "  rm -rf input/*.bin && ./generate_datasets.sh"
