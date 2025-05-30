cmake_minimum_required(VERSION 3.10)
project(trigger_pat)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(trigger_pat trigger_PAT.cpp)

# Add filesystem library if needed (for older compilers)
target_link_libraries(trigger_pat stdc++fs)
