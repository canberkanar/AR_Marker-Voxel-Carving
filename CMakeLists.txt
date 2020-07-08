cmake_minimum_required(VERSION 2.8)

project(helloAR)


SET(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/ )
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


# Required packages
find_package(OpenCV REQUIRED)

link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(marker_recognition marker_recognition.cpp)
add_executable(background_subtraction backgroud_subtraction.cpp)

add_executable(pose_estimation pose_estimation/pose_estimation.cpp)
add_executable(camera_calibration camera_calibration/camera_calibration.cpp)

# Adding local ARUco Library
IF (WIN32)
    include_directories(C:/Users/mayay/source/repos/3d-scanning/libs/aruco-lib/include/)
    link_directories(C:/Users/mayay/source/repos/3d-scanning/libs/aruco-lib/lib/)
    target_link_libraries(marker_recognition ${OpenCV_LIBS} aruco3112)
    target_link_libraries(background_subtraction ${OpenCV_LIBS})
    target_link_libraries(pose_estimation ${OpenCV_LIBS})
    target_link_libraries(camera_calibration ${OpenCV_LIBS})
ENDIF()

IF(UNIX)
    target_link_libraries(marker_recognition ${OpenCV_LIBS} aruco)
    target_link_libraries(background_subtraction ${OpenCV_LIBS})
    target_link_libraries(pose_estimation ${OpenCV_LIBS})
    target_link_libraries(camera_calibration ${OpenCV_LIBS})
ENDIF()

