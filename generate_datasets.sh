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
from PIL import Image
import math

print('Generating test datasets...')
print('Installing PIL if needed...')

# Try to install Pillow if not available
try:
    from PIL import Image
except ImportError:
    print('Installing Pillow for image generation...')
    import subprocess
    subprocess.check_call(['pip3', 'install', 'Pillow'])
    from PIL import Image

# Ensure reproducible results
random.seed(42)
np.random.seed(42)

# Function to save data as both binary and image formats
def save_dataset_with_image(data, name, size, description):
    # Save binary file for compression testing
    filename_bin = f'input/{name}_{size}.bin'
    with open(filename_bin, 'wb') as f:
        f.write(bytes(data))
    
    # Create image visualization for gradient and pattern data
    if any(pattern in name for pattern in ['gradient', 'sparse', 'repetitive', 'binary_pattern', 'mixed_pattern']):
        try:
            # Calculate image dimensions (try to make roughly square)
            total_pixels = len(data)
            width = int(math.sqrt(total_pixels))
            height = (total_pixels + width - 1) // width  # Ceiling division
            
            # Pad data if necessary to fill rectangle
            padded_data = data + [0] * (width * height - len(data))
            
            # Convert to numpy array and reshape
            img_array = np.array(padded_data[:width * height], dtype=np.uint8)
            img_array = img_array.reshape((height, width))
            
            # Scale values to full 0-255 range for better visibility
            if img_array.max() > 0:
                img_array = (img_array.astype(float) / img_array.max() * 255).astype(np.uint8)
            
            # Create and save image
            img = Image.fromarray(img_array, mode='L')  # Grayscale
            img_filename = f'input/{name}_{size}.png'
            img.save(img_filename)
            
            print(f'    Created {filename_bin} ({len(data):,} bytes) + {img_filename} ({width}x{height})')
            
        except Exception as e:
            print(f'    Created {filename_bin} ({len(data):,} bytes) - Image generation failed: {e}')
    else:
        print(f'    Created {filename_bin} ({len(data):,} bytes)')

# Function to create a simple bitmap manually (fallback if PIL fails)
def create_simple_bitmap(data, filename, width, height):
    # Simple BMP header for grayscale image
    file_size = 54 + width * height  # Header + pixel data
    
    # BMP file header (14 bytes)
    bmp_header = bytearray([
        0x42, 0x4D,  # 'BM'
        file_size & 0xFF, (file_size >> 8) & 0xFF, (file_size >> 16) & 0xFF, (file_size >> 24) & 0xFF,  # File size
        0x00, 0x00, 0x00, 0x00,  # Reserved
        0x36, 0x00, 0x00, 0x00   # Pixel data offset
    ])
    
    # BMP info header (40 bytes)
    info_header = bytearray([
        0x28, 0x00, 0x00, 0x00,  # Header size
        width & 0xFF, (width >> 8) & 0xFF, (width >> 16) & 0xFF, (width >> 24) & 0xFF,  # Width
        height & 0xFF, (height >> 8) & 0xFF, (height >> 16) & 0xFF, (height >> 24) & 0xFF,  # Height
        0x01, 0x00,  # Planes
        0x08, 0x00,  # Bits per pixel
        0x00, 0x00, 0x00, 0x00,  # Compression
        0x00, 0x00, 0x00, 0x00,  # Image size
        0x00, 0x00, 0x00, 0x00,  # X pixels per meter
        0x00, 0x00, 0x00, 0x00,  # Y pixels per meter
        0x00, 0x01, 0x00, 0x00,  # Colors used
        0x00, 0x00, 0x00, 0x00   # Important colors
    ])
    
    # Color palette (256 grayscale colors)
    palette = bytearray()
    for i in range(256):
        palette.extend([i, i, i, 0])  # B, G, R, A
    
    # Pixel data (bottom-up)
    pixel_data = bytearray()
    padded_data = data + [0] * (width * height - len(data))
    
    for y in range(height-1, -1, -1):  # BMP is bottom-up
        for x in range(width):
            pixel_index = y * width + x
            if pixel_index < len(padded_data):
                # Scale to 0-255 range
                value = min(255, max(0, int(padded_data[pixel_index] * 255 / max(1, max(padded_data)))))
                pixel_data.append(value)
            else:
                pixel_data.append(0)
    
    # Write BMP file
    with open(filename, 'wb') as f:
        f.write(bmp_header)
        f.write(info_header)
        f.write(palette)
        f.write(pixel_data)

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
        
        file_size = len(data)
        total_files += 1
        total_size += file_size
        save_dataset_with_image(data, name, size, desc)

print()
print('Creating performance test datasets:')
for name, desc, generator, sizes in performance_cases:
    print(f'  {name}: {desc}')
    for size in sizes:
        data = generator(size)
        
        file_size = len(data)
        total_files += 1
        total_size += file_size
        save_dataset_with_image(data, name, size, desc)

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
