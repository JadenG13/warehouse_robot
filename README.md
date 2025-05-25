# warehouse_robot  
*(3806ICT) Robotics Agents and Reasoning*

## Usage

### Launch the Grid Navigation Node
```
roslaunch warehouse_robot grid_navigation.launch world_name:=warehouse_1 x:=1 y:=-5
```

### Test Goal Sending
```
rosrun warehouse_robot send_goal.py charging_station
```

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
