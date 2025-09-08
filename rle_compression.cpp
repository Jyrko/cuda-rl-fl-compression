#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>
#include <chrono>

struct RLEResult {
    std::vector<unsigned char> symbols;
    std::vector<int> counts;
    size_t original_size;
    size_t compressed_size;
};

class RunLengthCompressor {
private:
    int max_run_length;
    
public:
    RunLengthCompressor(int max_run = 255) : max_run_length(max_run) {}
    
    RLEResult compress(const std::vector<unsigned char>& input) {
        RLEResult result;
        result.original_size = input.size();
        
        if (input.empty()) {
            result.compressed_size = 0;
            return result;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        unsigned char current_symbol = input[0];
        int count = 1;
        
        for (size_t i = 1; i < input.size(); i++) {
            if (input[i] == current_symbol && count < max_run_length) {
                count++;
            } else {
                // Store current run
                result.symbols.push_back(current_symbol);
                result.counts.push_back(count);
                
                // Start new run
                current_symbol = input[i];
                count = 1;
            }
        }
        
        // Store final run
        result.symbols.push_back(current_symbol);
        result.counts.push_back(count);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        result.compressed_size = result.symbols.size() + result.counts.size() * sizeof(int);
        
        std::cout << "RLE Compression completed in " << duration.count() << " microseconds" << std::endl;
        std::cout << "Found " << result.symbols.size() << " runs" << std::endl;
        
        return result;
    }
    
    std::vector<unsigned char> decompress(const RLEResult& compressed) {
        std::vector<unsigned char> decompressed;
        decompressed.reserve(compressed.original_size);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < compressed.symbols.size(); i++) {
            unsigned char symbol = compressed.symbols[i];
            int count = compressed.counts[i];
            
            for (int j = 0; j < count; j++) {
                decompressed.push_back(symbol);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "RLE Decompression completed in " << duration.count() << " microseconds" << std::endl;
        
        return decompressed;
    }
    
    void saveCompressedToFile(const RLEResult& result, const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot create output file " << filename << std::endl;
            return;
        }
        
        // Write header
        size_t num_runs = result.symbols.size();
        file.write(reinterpret_cast<const char*>(&result.original_size), sizeof(result.original_size));
        file.write(reinterpret_cast<const char*>(&num_runs), sizeof(num_runs));
        
        // Write symbols and counts
        file.write(reinterpret_cast<const char*>(result.symbols.data()), num_runs);
        file.write(reinterpret_cast<const char*>(result.counts.data()), num_runs * sizeof(int));
        
        std::cout << "RLE compressed data saved to: " << filename << std::endl;
    }
    
    RLEResult loadCompressedFromFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        RLEResult result;
        
        if (!file) {
            std::cerr << "Error: Cannot open compressed file " << filename << std::endl;
            return result;
        }
        
        // Read header
        size_t num_runs;
        file.read(reinterpret_cast<char*>(&result.original_size), sizeof(result.original_size));
        file.read(reinterpret_cast<char*>(&num_runs), sizeof(num_runs));
        
        // Read symbols and counts
        result.symbols.resize(num_runs);
        result.counts.resize(num_runs);
        
        file.read(reinterpret_cast<char*>(result.symbols.data()), num_runs);
        file.read(reinterpret_cast<char*>(result.counts.data()), num_runs * sizeof(int));
        
        result.compressed_size = result.symbols.size() + result.counts.size() * sizeof(int);
        
        std::cout << "Loaded RLE compressed data from: " << filename << std::endl;
        std::cout << "Original size: " << result.original_size << " bytes, " 
                  << num_runs << " runs" << std::endl;
        
        return result;
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
    
    void printCompressionStats(const RLEResult& result) {
        double compression_ratio = (double)result.original_size / result.compressed_size;
        double space_savings = (1.0 - (double)result.compressed_size / result.original_size) * 100;
        double avg_run_length = (double)result.original_size / result.symbols.size();
        
        std::cout << "\nRLE Compression Statistics:" << std::endl;
        std::cout << "Original size: " << result.original_size << " bytes" << std::endl;
        std::cout << "Compressed size: " << result.compressed_size << " bytes" << std::endl;
        std::cout << "Number of runs: " << result.symbols.size() << std::endl;
        std::cout << "Average run length: " << avg_run_length << std::endl;
        std::cout << "Compression ratio: " << compression_ratio << ":1" << std::endl;
        std::cout << "Space savings: " << space_savings << "%" << std::endl;
    }
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  -c, --compress <input_file>   Compress image file using RLE\n";
    std::cout << "  -d, --decompress <comp_file>  Decompress RLE file\n";
    std::cout << "  -o, --output <output_file>    Output file (default: auto-generated)\n";
    std::cout << "  -m, --max-run <size>          Maximum run length (default: 255)\n";
    std::cout << "  -h, --help                    Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " -c input/image.bin\n";
    std::cout << "  " << program_name << " -c input/image.bin -o compressed.rle\n";
    std::cout << "  " << program_name << " -d compressed.rle -o output.bin\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string mode;
    std::string input_file;
    std::string output_file;
    int max_run_length = 255;
    
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
        } else if (arg == "-m" || arg == "--max-run") {
            if (i + 1 < argc) {
                max_run_length = std::atoi(argv[++i]);
                if (max_run_length <= 0) {
                    std::cerr << "Error: Invalid max run length\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: Missing max run length\n";
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
    
    RunLengthCompressor compressor(max_run_length);
    
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
                output_file = input_file.substr(0, dot_pos) + ".rle";
            } else {
                output_file = input_file + ".rle";
            }
        }
        
        std::cout << "Compressing " << input_file << " (max run length: " << max_run_length << ")..." << std::endl;
        
        // Compress the data
        auto compressed = compressor.compress(image_data);
        
        // Save compressed data
        compressor.saveCompressedToFile(compressed, output_file);
        
        // Print statistics
        compressor.printCompressionStats(compressed);
        
    } else if (mode == "decompress") {
        // Load compressed file
        auto compressed = compressor.loadCompressedFromFile(input_file);
        if (compressed.symbols.empty()) {
            return 1;
        }
        
        // Generate output filename if not specified
        if (output_file.empty()) {
            size_t dot_pos = input_file.find_last_of('.');
            if (dot_pos != std::string::npos) {
                output_file = input_file.substr(0, dot_pos) + "_decompressed.bin";
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
