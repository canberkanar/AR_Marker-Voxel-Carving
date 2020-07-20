cmake_minimum_required(VERSION 2.8)

project(helloAR)


SET(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/ )
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")


# Required packages
find_package(OpenCV REQUIRED)
find_package(aruco REQUIRED)
find_package(VTK 8.2 REQUIRED)
#find_package(Eigen3 REQUIRED)



link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${aruco_LIBRARY_DIRS})
link_directories(${VTK_LIBRARY_DIRS})

#add_executable(marker_recognition marker_recognition.cpp utils.cpp)
add_executable(marker_recognition_img marker_recognition_img.cpp utils.cpp)
#add_executable(background_subtraction backgroud_subtraction.cpp utils.cpp)
#add_executable(pose_estimation pose_estimation.cpp)
add_executable(camera_calibration extern/camera_calibration.cpp)
add_executable(marker_generation marker_generation.cpp)

#add_executable(vacancy_main vacancy.cpp utils.cpp)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${aruco_INCLUDE_DIRS} )
#include_directories( ${EIGEN3_INCLUDE_DIRS} )



IF (WIN32)
    include_directories( C:/Users/TheaK/Documents/TUM/Master/3DScanning/project/libs/aruco/out/install/x64-Debug/include)
    # Adding local ARUco Library
    include_directories(C:/Users/mayay/source/repos/3d-scanning/libs/aruco-lib/include/)
    #link_directories(C:/Users/mayay/source/repos/3d-scanning/libs/aruco-lib/lib/)
   # target_link_libraries(marker_recognition aruco3112 ${OpenCV_LIBS})
    target_link_libraries(marker_recognition_img aruco3112 ${OpenCV_LIBS} ${VTK_LIBRARIES})
    target_link_libraries(marker_generation aruco3112 ${OpenCV_LIBS})
    #target_link_libraries(background_subtraction ${OpenCV_LIBS})
    #target_link_libraries(pose_estimation ${OpenCV_LIBS} ${aruco_LIBS})
    target_link_libraries(camera_calibration ${OpenCV_LIBS} ${aruco_LIBS})
ENDIF()

IF(UNIX)
    target_link_libraries(marker_recognition ${OpenCV_LIBS} aruco)
    target_link_libraries(background_subtraction ${OpenCV_LIBS})
#    target_link_libraries(pose_estimation ${OpenCV_LIBS} aruco)
    target_link_libraries(camera_calibration ${OpenCV_LIBS} aruco)
    target_link_libraries(marker_generation ${OpenCV_LIBS} aruco)
    target_link_libraries(marker_recognition_img aruco ${OpenCV_LIBS} ${VTK_LIBRARIES})
#    target_link_libraries(vacancy_main ${OpenCV_LIBS} vacancy)
ENDIF()

