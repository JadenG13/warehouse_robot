# Required ros packages
source /opt/ros/noetic/setup.bash
sudo apt install -y ros-noetic-move-base-msgs ros-noetic-turtlebot3*

export DISABLE_ROS1_EOL_WARNINGS=1
source ~/.bashrc

# Check and install Mono
if ! dpkg -l | grep -q mono-complete; then
  sudo apt install -y mono-complete libmono-system-windows-forms4.0-cil
fi

# Ensure pip is installed
if ! command -v pip &> /dev/null; then
  sudo apt install -y python3-pip
fi

# Install Ollama
if ! command -v ollama &> /dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh || {
    echo "Failed to install Ollama. Please check the URL or your internet connection.";
    exit 1;
  }
else
  echo "Ollama is already installed. Skipping installation."
fi

# Install Ollama Python package
if ! pip show ollama &> /dev/null; then
  pip install ollama || {
    echo "Failed to install Ollama Python package. Please check your pip installation.";
    exit 1;
  }
else
  echo "Ollama Python package is already installed. Skipping installation."
fi


# Build the catkin workspace
cd ~/catkin_ws && catkin_make && source ~/catkin_ws/devel/setup.sh || {
  echo "Failed to build the catkin workspace. Please check for errors.";
  exit 1;
}
