# Warehouse Robot

*(3806ICT) Robotics Agents and Reasoning*

## Setup Instructions

1. Create the workspace directory:
   ```bash
   mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws/src
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/JadenG13/warehouse_robot.git
   cd warehouse_robot
   ```

3. Install dependencies:
   ```bash
   sudo apt install -y ros-noetic-move-base-msgs ros-noetic-turtlebot3* ros-noetic-gazebo-ros* mono-complete libmono-system-windows-forms4.0-cil python3-pip
   curl -fsSL https://ollama.com/install.sh | sh
   pip install ollama
   echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   export DISABLE_ROS1_EOL_WARNINGS=1
   source ~/.bashrc
   ```

4. Install llama3.2 by doing the following:
   - In the first terminal, start the Ollama server:
     ```bash
     ollama serve
     ```
   - In the second terminal, run the llama3.2 model:
     ```bash
     ollama run llama3.2
     ```
     Once the installation is complete, you may exit the terminal by pressing `Ctrl+C`.

## Usage

### Build the Package

Run the following command to build the package:
```bash
catkin_make && source ~/catkin_ws/devel/setup.sh
```

### Start the Ollama Server

Ensure the Ollama server is running:
```bash
ollama serve
```

### Launch the Package

Launch the package with the specified world:
```bash
roslaunch warehouse_robot grid_navigation.launch world_name:=warehouse_1
```