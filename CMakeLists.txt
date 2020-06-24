cmake_minimum_required(VERSION 2.8)

project(helloAR)



SET(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/ )

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")



# Required packages

find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

# Adding local ARUco Library

include_directories(C:/Users/mayay/source/repos/3d-scanning/libs/aruco-lib/include/)

link_directories(C:/Users/mayay/source/repos/3d-scanning/libs/aruco-lib/lib/)



add_executable(helloAR ARVoxelCarving.cpp)

target_link_libraries(helloAR ${OpenCV_LIBS} aruco3112)