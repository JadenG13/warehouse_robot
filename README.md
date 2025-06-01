# warehouse_robot
*(3806ICT) Robotics Agents and Reasoning*

## Usage

### catkin_make package
```
catkin_make && source ~/catkin_ws/devel/setup.sh
```

### Launch the Grid Navigation Node
```
export DISABLE_ROS1_EOL_WARNINGS=1 && roslaunch warehouse_robot grid_navigation.launch world_name:=warehouse_1
```

## Logging (rosbag)

### Run Command for Logging
```
rosbag record -O performance_capture.bag /rosout_agg
```

Ctrl + C to exit

This captures a bag in the root directory `/catkin_ws/`, which you can then convert to a log file using `rostobag.py`.

More robust than previous version, make sure you use `rostobag.py` with `python3`.

You will need to install dependencies:
```
pip install pycryptodomex
pip install python-gnupg
```

Then run `logging.py`.

Make sure to point to the log file — update the code (path is hardcoded).

## Requirements

- Ollama 3.2

## Installing Ollama

### Option 1: Install via Shell Script
```
curl -fsSL https://ollama.com/install.sh | sh
```

### Option 2: Install via pip (Python wrapper)
```
pip install ollama
```

> ⚠️ Note: The pip version installs the Python client. You still need to install and run the native Ollama backend using Option 1.

### Run the Ollama model
```
ollama run llama3.2
```

### Start the Ollama server (if not already running)
```
ollama serve
```
