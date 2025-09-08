#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstring>
#include <string>

struct CompressedFrame {
    std::vector<unsigned char> data;
    int bits_per_value;
    int frame_size;
};

class FixedLengthCompressor {
private:
    int frame_size;
    
    int findRequiredBits(const std::vector<unsigned char>& frame) {
        if (frame.empty()) return 0;
        
        unsigned char max_val = *std::max_element(frame.begin(), frame.end());
        if (max_val == 0) return 1; // Need at least 1 bit for zero
        
        int bits = 0;
        while (max_val > 0) {
            max_val >>= 1;
            bits++;
        }
        return bits;
    }
    
    std::vector<unsigned char> packBits(const std::vector<unsigned char>& frame, int bits_per_value) {
        std::vector<unsigned char> packed;
        int current_byte = 0;
        int bits_filled = 0;
        
        for (unsigned char value : frame) {
            // Mask to only keep required bits
            value &= (1 << bits_per_value) - 1;
            
            int bits_to_process = bits_per_value;
            while (bits_to_process > 0) {
                int space_in_byte = 8 - bits_filled;
                int bits_to_add = std::min(bits_to_process, space_in_byte);
                
                // Shift value to fit in current position
                int shifted_value = (value >> (bits_to_process - bits_to_add)) & ((1 << bits_to_add) - 1);
                current_byte |= shifted_value << (space_in_byte - bits_to_add);
                
                bits_filled += bits_to_add;
                bits_to_process -= bits_to_add;
                
                if (bits_filled == 8) {
                    packed.push_back(current_byte);
                    current_byte = 0;
                    bits_filled = 0;
                }
            }
        }
        
        // Add remaining bits if any
        if (bits_filled > 0) {
            packed.push_back(current_byte);
        }
        
        return packed;
    }
    
    std::vector<unsigned char> unpackBits(const std::vector<unsigned char>& packed, 
                                         int bits_per_value, int original_size) {
        std::vector<unsigned char> unpacked;
        int bit_position = 0;
        
        for (int i = 0; i < original_size; i++) {
            unsigned char value = 0;
            
            for (int bit = 0; bit < bits_per_value; bit++) {
                int byte_index = bit_position / 8;
                int bit_index = bit_position % 8;
                
                if (byte_index < packed.size()) {
                    int bit_value = (packed[byte_index] >> (7 - bit_index)) & 1;
                    value |= bit_value << (bits_per_value - 1 - bit);
                }
                bit_position++;
            }
            
            unpacked.push_back(value);
        }
        
        return unpacked;
    }
    
public:
    FixedLengthCompressor(int frame_size = 256) : frame_size(frame_size) {}
    
    std::vector<CompressedFrame> compress(const std::vector<unsigned char>& input) {
        std::vector<CompressedFrame> compressed_frames;
        
        for (size_t i = 0; i < input.size(); i += frame_size) {
            // Extract frame
            std::vector<unsigned char> frame;
            for (size_t j = i; j < std::min(i + frame_size, input.size()); j++) {
                frame.push_back(input[j]);
            }
            
            // Find required bits
            int bits_required = findRequiredBits(frame);
            
            // Pack the frame
            std::vector<unsigned char> packed = packBits(frame, bits_required);
            
            // Store compressed frame
            CompressedFrame cf;
            cf.data = packed;
            cf.bits_per_value = bits_required;
            cf.frame_size = frame.size();
            compressed_frames.push_back(cf);
            
            std::cout << "Frame " << compressed_frames.size() - 1 
                      << ": " << frame.size() << " bytes -> " << packed.size() 
                      << " bytes (" << bits_required << " bits/value)" << std::endl;
        }
        
        return compressed_frames;
    }
    
    std::vector<unsigned char> decompress(const std::vector<CompressedFrame>& compressed_frames) {
        std::vector<unsigned char> decompressed;
        
        for (const auto& frame : compressed_frames) {
            std::vector<unsigned char> unpacked = unpackBits(frame.data, 
                                                           frame.bits_per_value, 
                                                           frame.frame_size);
            decompressed.insert(decompressed.end(), unpacked.begin(), unpacked.end());
        }
        
        return decompressed;
    }
    
    void saveCompressedToFile(const std::vector<CompressedFrame>& compressed_frames, 
                             const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot create output file " << filename << std::endl;
            return;
        }
        
        // Write number of frames
        int num_frames = compressed_frames.size();
        file.write(reinterpret_cast<const char*>(&num_frames), sizeof(num_frames));
        
        // Write frame information and data
        for (const auto& frame : compressed_frames) {
            file.write(reinterpret_cast<const char*>(&frame.bits_per_value), sizeof(frame.bits_per_value));
            file.write(reinterpret_cast<const char*>(&frame.frame_size), sizeof(frame.frame_size));
            
            int data_size = frame.data.size();
            file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
            file.write(reinterpret_cast<const char*>(frame.data.data()), data_size);
        }
        
        std::cout << "Compressed data saved to: " << filename << std::endl;
    }
    
    std::vector<CompressedFrame> loadCompressedFromFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot open compressed file " << filename << std::endl;
            return {};
        }
        
        std::vector<CompressedFrame> compressed_frames;
        
        // Read number of frames
        int num_frames;
        file.read(reinterpret_cast<char*>(&num_frames), sizeof(num_frames));
        
        // Read frame information and data
        for (int i = 0; i < num_frames; i++) {
            CompressedFrame frame;
            file.read(reinterpret_cast<char*>(&frame.bits_per_value), sizeof(frame.bits_per_value));
            file.read(reinterpret_cast<char*>(&frame.frame_size), sizeof(frame.frame_size));
            
            int data_size;
            file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
            
            frame.data.resize(data_size);
            file.read(reinterpret_cast<char*>(frame.data.data()), data_size);
            
            compressed_frames.push_back(frame);
        }
        
        std::cout << "Loaded " << num_frames << " compressed frames from: " << filename << std::endl;
        return compressed_frames;
    }
    
    std::vector<unsigned char> loadImageFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot open image file " << filename << std::endl;
            return {};
        }
        
        // Get file size
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        // Read file data
        std::vector<unsigned char> data(file_size);
        file.read(reinterpret_cast<char*>(data.data()), file_size);
        
        std::cout << "Loaded image file: " << filename << " (" << file_size << " bytes)" << std::endl;
        return data;
    }
    
    void saveDecompressedImage(const std::vector<unsigned char>& data, const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot create output file " << filename << std::endl;
            return;
        }
        
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        std::cout << "Decompressed data saved to: " << filename << " (" << data.size() << " bytes)" << std::endl;
    }
    
    void printCompressionStats(const std::vector<unsigned char>& original, 
                              const std::vector<CompressedFrame>& compressed) {
        size_t original_size = original.size();
        size_t compressed_size = 0;
        
        for (const auto& frame : compressed) {
            compressed_size += frame.data.size();
        }
        
        double compression_ratio = (double)original_size / compressed_size;
        double space_savings = (1.0 - (double)compressed_size / original_size) * 100;
        
        std::cout << "\nCompression Statistics:" << std::endl;
        std::cout << "Original size: " << original_size << " bytes" << std::endl;
        std::cout << "Compressed size: " << compressed_size << " bytes" << std::endl;
        std::cout << "Compression ratio: " << compression_ratio << ":1" << std::endl;
        std::cout << "Space savings: " << space_savings << "%" << std::endl;
    }
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  -c, --compress <input_file>   Compress image file\n";
    std::cout << "  -d, --decompress <comp_file>  Decompress file\n";
    std::cout << "  -o, --output <output_file>    Output file (default: auto-generated)\n";
    std::cout << "  -f, --frame-size <size>       Frame size for compression (default: 256)\n";
    std::cout << "  -h, --help                    Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " -c input/image.jpg\n";
    std::cout << "  " << program_name << " -c input/image.jpg -o compressed.flc\n";
    std::cout << "  " << program_name << " -d compressed.flc -o output.jpg\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string mode;
    std::string input_file;
    std::string output_file;
    int frame_size = 256;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-c" || arg == "--compress") {
            mode = "compress";
            if (i + 1 < argc) {
                input_file = argv[++i];
            } else {
                std::cerr << "Error: Missing input file for compression\n";
                return 1;
            }
        } else if (arg == "-d" || arg == "--decompress") {
            mode = "decompress";
            if (i + 1 < argc) {
                input_file = argv[++i];
            } else {
                std::cerr << "Error: Missing input file for decompression\n";
                return 1;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            } else {
                std::cerr << "Error: Missing output file\n";
                return 1;
            }
        } else if (arg == "-f" || arg == "--frame-size") {
            if (i + 1 < argc) {
                frame_size = std::atoi(argv[++i]);
                if (frame_size <= 0) {
                    std::cerr << "Error: Invalid frame size\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: Missing frame size\n";
                return 1;
            }
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Error: Unknown option " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (mode.empty()) {
        std::cerr << "Error: Must specify either --compress or --decompress\n";
        printUsage(argv[0]);
        return 1;
    }
    
    FixedLengthCompressor compressor(frame_size);
    
    if (mode == "compress") {
        // Load image file
        auto image_data = compressor.loadImageFile(input_file);
        if (image_data.empty()) {
            return 1;
        }
        
        // Generate output filename if not specified
        if (output_file.empty()) {
            size_t dot_pos = input_file.find_last_of('.');
            if (dot_pos != std::string::npos) {
                output_file = input_file.substr(0, dot_pos) + ".flc";
            } else {
                output_file = input_file + ".flc";
            }
        }
        
        std::cout << "Compressing " << input_file << " (frame size: " << frame_size << ")..." << std::endl;
        
        // Compress the data
        auto compressed = compressor.compress(image_data);
        
        // Save compressed data
        compressor.saveCompressedToFile(compressed, output_file);
        
        // Print statistics
        compressor.printCompressionStats(image_data, compressed);
        
    } else if (mode == "decompress") {
        // Load compressed file
        auto compressed = compressor.loadCompressedFromFile(input_file);
        if (compressed.empty()) {
            return 1;
        }
        
        // Generate output filename if not specified
        if (output_file.empty()) {
            size_t dot_pos = input_file.find_last_of('.');
            if (dot_pos != std::string::npos) {
                output_file = input_file.substr(0, dot_pos) + "_decompressed";
                // Try to guess original extension or use .bin
                std::string base_name = input_file.substr(0, dot_pos);
                size_t prev_dot = base_name.find_last_of('.');
                if (prev_dot != std::string::npos) {
                    output_file += base_name.substr(prev_dot);
                } else {
                    output_file += ".bin";
                }
            } else {
                output_file = input_file + "_decompressed.bin";
            }
        }
        
        std::cout << "Decompressing " << input_file << "..." << std::endl;
        
        // Decompress the data
        auto decompressed = compressor.decompress(compressed);
        
        // Save decompressed data
        compressor.saveDecompressedImage(decompressed, output_file);
        
        std::cout << "Decompression completed successfully!" << std::endl;
    }
    
    return 0;
}