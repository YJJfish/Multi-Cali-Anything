cmake_minimum_required(VERSION 3.8)
project(calibration LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Ceres REQUIRED)
find_package(SQLite3 REQUIRED)

file(GLOB SOURCES  *.h *.hpp *.cuh *.cpp *.cu)

add_executable(calibration ${SOURCES})

set_target_properties(calibration PROPERTIES CUDA_ARCHITECTURES "86")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(
	calibration
	Ceres::ceres
	SQLite::SQLite3
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -lineinfo")