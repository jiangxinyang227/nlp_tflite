cmake_minimum_required(VERSION 3.15)
project(nlp_tflite)
set(CMAKE_CXX_STANDARD 14)

# 添加头文件
set(INC_DIR /Users/jiangxinyang/Desktop/inference_tflite/include)
include_directories(${INC_DIR})

# 添加动态
set(LINK_DIR /Users/jiangxinyang/Desktop/inference_tflite/lib)
link_directories(${LINK_DIR})
link_libraries(tensorflowlite flatbuffers)

add_executable(nlp_tflite src/inference.cpp src/inference.h)
target_link_libraries(nlp_tflite tensorflowlite flatbuffers)
