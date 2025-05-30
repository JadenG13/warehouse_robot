#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cstdlib>
#include <sstream>

namespace fs = std::filesystem;

// Global constants
const fs::path TEMPLATE_PATH = "../template.csp";
const fs::path PAT_EXE = "../MONO-PAT-v3.6.0/PAT3.Console.exe";
const fs::path OUTPUT_FILE = "../output.txt";
const fs::path RUN_CSP = "../run.csp";

// Utility function to replace all occurrences of a substring
std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

// Function to build CSP from template CSP
fs::path build_csp(const std::vector<int>& grid, 
                   std::pair<int, int> start_rc,
                   std::pair<int, int> goal_rc,
                   int M,
                   const std::string& out_path) {
    // Get positions
    auto [start_r, start_c] = start_rc;
    auto [goal_r, goal_c] = goal_rc;

    // Read template text
    std::ifstream template_file(TEMPLATE_PATH);
    if (!template_file.is_open()) {
        throw std::runtime_error("Could not open template file");
    }

    std::stringstream buffer;
    buffer << template_file.rdbuf();
    std::string template_content = buffer.str();

    // Convert grid to string
    std::stringstream grid_str;
    grid_str << "[";
    for (size_t i = 0; i < grid.size(); ++i) {
        grid_str << grid[i];
        if (i < grid.size() - 1) grid_str << ",";
    }
    grid_str << "]";

    // Replace placeholders with updated parameters
    std::string filled = template_content;
    filled = replace_all(filled, "{{M}}", std::to_string(M));
    filled = replace_all(filled, "{{GRID}}", grid_str.str());
    filled = replace_all(filled, "{{START_R}}", std::to_string(start_r));
    filled = replace_all(filled, "{{START_C}}", std::to_string(start_c));
    filled = replace_all(filled, "{{GOAL_R}}", std::to_string(goal_r));
    filled = replace_all(filled, "{{GOAL_C}}", std::to_string(goal_c));

    // Write to output file
    fs::path out_file = out_path;
    std::ofstream output(out_file);
    if (!output.is_open()) {
        throw std::runtime_error("Could not open output file for writing");
    }
    output << filled;
    output.close();

    return out_file;
}

// Function to run PAT
void run_pat(const fs::path& csp_path) {
    std::cout << PAT_EXE.string() << " " << csp_path.string() << " " << OUTPUT_FILE.string() << std::endl;
    
    // Construct command
    std::string command = "mono " + PAT_EXE.string() + " " + csp_path.string() + " " + OUTPUT_FILE.string();
    
    // Execute command
    int return_code = system(command.c_str());
    
    if (return_code != 0) {
        throw std::runtime_error("PAT exited with error code: " + std::to_string(return_code));
    }

    // Read and print verification result
    if (fs::exists(OUTPUT_FILE)) {
        std::ifstream result_file(OUTPUT_FILE);
        if (result_file.is_open()) {
            std::string line;
            std::cout << "Verification Result:" << std::endl;
            while (std::getline(result_file, line)) {
                std::cout << line << std::endl;
            }
        }
    }
}

int main() {
    // Example data
    std::vector<int> grid = {
        1, 1, 1,-1, 1, 1, 1, 1,
        1, 1, 1,-1,-1, 1, 1, 1,
        1,-1,-1, 1, 1, 1, 1, 1,
        1,-1, 1, 1, 1, 1, 1,-1,
        1, 1, 1, 1, 1, 1,-1,-1,
        1, 1, 1, 1, 1,-1, 1,-1,
        1, 1, 1, 1, 1,-1, 1, 1,
        1, 1, 1, 1, 1,-1, 1, 1
    };

    std::pair<int, int> start_pos = {4, 1};
    std::pair<int, int> goal_pos = {0, 4};
    int grid_size = 8;

    try {
        // Get CSP file and run PAT
        fs::path csp_file = build_csp(grid, start_pos, goal_pos, grid_size, RUN_CSP.string());
        run_pat(csp_file);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
