#include <ros/ros.h>
#include <warehouse_robot/ValidatePathWithPat.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_datatypes.h>
#include <filesystem>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

// ValidatorAgent: A ROS node that validates robot paths using PAT model checker
//
// This agent converts costmap data to a CSP model and uses PAT to check
// if a valid path exists between start and goal positions. It works by:
// 1. Converting world coordinates to grid coordinates
// 2. Cropping the costmap to the usable area (-7,-7) to (7,7)
// 3. Generating a CSP model from the grid
// 4. Using PAT to verify path existence
class ValidatorAgent {
private:
    ros::NodeHandle nh_;
    ros::Subscriber costmap_sub_;
    ros::ServiceServer validate_pat_srv_;

    nav_msgs::OccupancyGrid::ConstPtr global_costmap_;

    // Configuration parameters
    std::string world_name_;
    XmlRpc::XmlRpcValue config_;
    double cell_size_;
    double grid_origin_x_;
    double grid_origin_y_;

    // PAT model checker paths and configuration
    std::string PAT_DIR = std::string(getenv("HOME")) + "/catkin_ws/src/warehouse_robot/pat/";
    const fs::path TEMPLATE_PATH = PAT_DIR + "template.csp";
    const fs::path PAT_EXE = PAT_DIR + "MONO-PAT-v3.6.0/PAT3.Console.exe";
    const fs::path OUTPUT_FILE = PAT_DIR + "output.txt";
    const fs::path RUN_CSP = PAT_DIR + "run.csp";

    // Convert world coordinates to grid coordinates.
    // x, y: World coordinates
    // map: Occupancy grid pointer
    // Returns: (x, y) grid coordinates
    std::pair<int, int> worldToGrid(double x, double y, const nav_msgs::OccupancyGrid::ConstPtr& map) {
        int grid_x = static_cast<int>(std::round((x - map->info.origin.position.x) / cell_size_));
        int grid_y = static_cast<int>(std::round((y - map->info.origin.position.y) / cell_size_));
        return {grid_x, grid_y};
    }

    // Build CSP model from grid data.
    // grid: Occupancy grid as vector
    // start_rc: Start (row, col)
    // goal_rc: Goal (row, col)
    // M: Grid size
    // out_path: Output CSP file path
    // Returns: Path to generated CSP file
    fs::path buildCsp(const std::vector<int>& grid, 
                      std::pair<int, int> start_rc,
                      std::pair<int, int> goal_rc,
                      int M,
                      const std::string& out_path) {
        // Get positions
        auto [start_c, start_r] = start_rc;
        auto [goal_c, goal_r] = goal_rc;

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
        int row = 0;
        int grid_width = M;  // M is the width/height of the square grid
        for (size_t i = 0; i < grid.size(); ++i) {
            grid_str << grid[i];
            if (i < grid.size() - 1) {
                grid_str << ",";
                if ((i + 1) % grid_width == 0) {
                    grid_str << "\n ";  // Add newline and space for alignment
                    row++;
                }
            }
        }
        grid_str << "]";

        // Replace placeholders with updated parameters
        template_content = replace_all(template_content, "{{GRID}}", grid_str.str());
        template_content = replace_all(template_content, "{{START_R}}", std::to_string(start_r));
        template_content = replace_all(template_content, "{{START_C}}", std::to_string(start_c));
        template_content = replace_all(template_content, "{{GOAL_R}}", std::to_string(goal_r));
        template_content = replace_all(template_content, "{{GOAL_C}}", std::to_string(goal_c));
        template_content = replace_all(template_content, "{{M}}", std::to_string(M));

        // Write to output file
        std::ofstream out_file(out_path);
        if (!out_file.is_open()) {
            throw std::runtime_error("Could not open output file");
        }
        out_file << template_content;
        out_file.close();

        return fs::path(out_path);
    }

    // Replace all occurrences of a string
    std::string replace_all(std::string str, const std::string& from, const std::string& to) {
        size_t start_pos = 0;
        while((start_pos = str.find(from, start_pos)) != std::string::npos) {
            str.replace(start_pos, from.length(), to);
            start_pos += to.length();
        }
        return str;
    }

    // Run PAT verification on the generated CSP model
    // csp_path: Path to CSP file
    // Returns: true if a path exists, false otherwise
    bool verifyWithPat(const fs::path& csp_path) {
        std::string command = "mono " + PAT_EXE.string() + " " + csp_path.string() + " " + OUTPUT_FILE.string();
        int result = system(command.c_str());
        
        if (result != 0) {
            ROS_ERROR("PAT verification failed");
            return false;
        }

        // Parse verification result from output file
        std::ifstream output_file(OUTPUT_FILE);
        std::string line;
        bool valid = true;

        while (std::getline(output_file, line)) {
            if (line.find("NOT") != std::string::npos) {
                valid = false;
                break;
            }
        }

        return valid;
    }

public:
    ValidatorAgent() : nh_("~") {
        // Load configuration from parameters
        if (!nh_.getParam("world_name", world_name_)) {
            ROS_ERROR("[Validator] Failed to get world_name parameter");
            throw std::runtime_error("Failed to get world_name parameter");
        }
        
        if (!nh_.getParam("/" + world_name_ + "/cell_size", cell_size_)) {
            ROS_ERROR_STREAM("Failed to load " << world_name_ << "/cell_size");
        }
        if (!nh_.getParam("/" + world_name_ + "/grid_origin_x", grid_origin_x_)) {
            ROS_ERROR_STREAM("Failed to load " << world_name_ << "/grid_origin_x");
        }
        if (!nh_.getParam("/" + world_name_ + "/grid_origin_y", grid_origin_y_)) {
            ROS_ERROR_STREAM("Failed to load " << world_name_ << "/grid_origin_y");
        }
        
        // Initialize ROS components
        costmap_sub_ = nh_.subscribe("/move_base/global_costmap/costmap", 10, 
            &ValidatorAgent::costmapCallback, this);
        validate_pat_srv_ = nh_.advertiseService("validate_path_with_pat",
            &ValidatorAgent::handleValidatePathWithPat, this);

        ROS_INFO("[Validator] Ready.");
    }

    // Store the latest costmap data for path validation
    void costmapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        global_costmap_ = msg;
    }

    bool handleValidatePathWithPat(warehouse_robot::ValidatePathWithPat::Request &req,
                                 warehouse_robot::ValidatePathWithPat::Response &res) {
        ROS_INFO_STREAM("[Validator] Received request to validate path with PAT");
        if (!global_costmap_) {
            res.exists = false;
            res.message = "No costmap data available";
            return true;
        }
        ROS_INFO_STREAM("[Validator] Global costmap received with size: " 
                        << global_costmap_->info.width << "x" 
                        << global_costmap_->info.height);

        try {
            // Validate start and goal poses
            ROS_INFO("[Validator] Validating start and goal poses...");
            
            // Get the costmap resolution and info
            int costmap_width = global_costmap_->info.width;
            int costmap_height = global_costmap_->info.height;
            double origin_x = global_costmap_->info.origin.position.x;
            double origin_y = global_costmap_->info.origin.position.y;

            
            // Use grid coordinates from constructor
            ROS_DEBUG_STREAM("[Validator] Using grid origin: (" << grid_origin_x_ << ", " << grid_origin_y_ << ")");
            int min_x = static_cast<int>(std::round((grid_origin_x_ - origin_x) / cell_size_));
            int max_x = static_cast<int>(std::round((-grid_origin_x_ - origin_x) / cell_size_));
            int min_y = static_cast<int>(std::round((grid_origin_y_ - origin_y) / cell_size_));
            int max_y = static_cast<int>(std::round((-grid_origin_y_ - origin_y) / cell_size_));

            // Convert poses to grid coordinates
            auto start = worldToGrid(req.start_pose.pose.position.x,
                                   req.start_pose.pose.position.y,
                                   global_costmap_);
            auto goal = worldToGrid(req.goal_pose.pose.position.x,
                                  req.goal_pose.pose.position.y,
                                  global_costmap_);

            ROS_INFO_STREAM("[Validator] Start: (" << start.first << ", " << start.second << 
                          "), Goal: (" << goal.first << ", " << goal.second << ")");

            // Create cropped grid for PAT
            int crop_size = (max_x - min_x + 1);
            std::vector<int> grid;
            grid.reserve(crop_size * crop_size);

            ROS_INFO_STREAM("[Validator] Cropping costmap to area: (" << min_x << ", " << min_y << 
                          ") to (" << max_x << ", " << max_y << ")");

            // Copy only the cropped area
            for (int y = min_y; y <= max_y; y++) {
                for (int x = min_x; x <= max_x; x++) {
                    if (x >= 0 && x < costmap_width && y >= 0 && y < costmap_height) {
                        int idx = y * costmap_width + x;
                        grid.push_back(global_costmap_->data[idx] > 0 ? -1 : 1);
                    } else {
                        grid.push_back(-1); // Mark out of bounds as obstacles
                    }
                }
            }
            
            ROS_INFO_STREAM("[Validator] Cropped grid size: " << crop_size << "x" << crop_size);

            // Adjust start/goal coordinates to cropped grid
            start.first -= min_x;
            start.second -= min_y;
            goal.first -= min_x;
            goal.second -= min_y;

            // Add debug logging
            ROS_INFO_STREAM("[Validator] Map bounds: (" << origin_x << "," << origin_y << ") to (" 
                          << (origin_x + costmap_width * cell_size_) << "," 
                          << (origin_y + costmap_height * cell_size_) << ")");
            ROS_INFO_STREAM("[Validator] Crop area: (" << min_x << "," << min_y << ") to ("
                          << max_x << "," << max_y << ")");
            ROS_INFO_STREAM("[Validator] Adjusted start: (" << start.first << "," << start.second 
                          << "), goal: (" << goal.first << "," << goal.second << ")");

            // Verify coordinates are in cropped grid
            if (start.first < 0 || start.first >= crop_size ||
                start.second < 0 || start.second >= crop_size ||
                goal.first < 0 || goal.first >= crop_size ||
                goal.second < 0 || goal.second >= crop_size) {
                res.exists = false;
                res.message = "Start or goal position outside valid area";
                return true;
            }

            // Build CSP file with cropped grid
            auto csp_path = buildCsp(grid, start, goal, crop_size, RUN_CSP.string());
            ROS_INFO_STREAM("[Validator] CSP file created at: " << csp_path.string());

            // Verify with PAT
            bool path_exists = verifyWithPat(csp_path);
            ROS_INFO_STREAM("[Validator] Path verification result: " << (path_exists ? "Exists" : "Does not exist"));

            res.exists = path_exists;
            res.message = path_exists ? "Path exists" : "No path exists";

        } catch (const std::exception& e) {
            ROS_ERROR_STREAM("[Validator] Error in PAT verification: " << e.what());
            res.exists = false;
            res.message = std::string("Error: ") + e.what();
        }
        ROS_INFO_STREAM("[Validator] Validation complete. Result: " << res.message);

        return true;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "validator_agent");
    ROS_INFO("[Validator] Starting validator agent...");
    ValidatorAgent validator;
    ros::spin();
    return 0;
}
