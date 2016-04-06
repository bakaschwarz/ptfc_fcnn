cmake_minimum_required(VERSION 3.4)
project(ptfc_fcnn)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
if(NOT UNIX)
    set(CMAKE_C_COMPILER cl.exe) #TODO Very rudimental Windows VS Support. Needs way more than that
endif()

add_subdirectory(util/fcnn)
add_subdirectory(util/tclap)

set(SOURCE_FILES main.cpp main.h)
add_executable(ptfc_fcnn ${SOURCE_FILES})

target_link_libraries(ptfc_fcnn LINK_PUBLIC fcnn tclap)