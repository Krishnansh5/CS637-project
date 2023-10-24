cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fast_clarans)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Debug)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(fast_clarans test.cpp python_wrapper.cpp pam.cpp ssim.cpp)
# add_library(fast_clarans SHARED pythonWrapper.cpp pythonWrapper.hpp pam.cpp pam.h ssim.cpp)
target_link_libraries(fast_clarans "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET fast_clarans PROPERTY CXX_STANDARD 17)