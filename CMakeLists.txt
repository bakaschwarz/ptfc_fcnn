cmake_minimum_required(VERSION 3.4)
project(fcnn_train)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(util/fcnn)
add_subdirectory(util/tclap)

set(SOURCE_FILES main.cpp main.h)
add_executable(fcnn_train ${SOURCE_FILES})

target_link_libraries(fcnn_train LINK_PUBLIC fcnn tclap)
