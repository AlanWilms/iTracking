cmake_minimum_required(VERSION 3.9)
project(Open_CV_Eye_Detection)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV HINTS /usr/local/opt/opencv3 )

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

set(SOURCE_FILES eye_detector.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(iTracking ${SOURCE_FILES})
target_link_libraries( iTracking ${OpenCV_LIBS} )
