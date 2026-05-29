cmake_minimum_required(VERSION 3.10)
project(VisionPipeline)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(PCL 1.12 REQUIRED COMPONENTS common io filters registration segmentation sample_consensus visualization)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(vision_pipeline vision_pipeline.cpp)

target_link_libraries(vision_pipeline ${PCL_LIBRARIES})