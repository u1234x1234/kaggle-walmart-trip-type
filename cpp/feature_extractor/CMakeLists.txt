project(feature_extractor)
cmake_minimum_required(VERSION 2.8)


include_directories(
	${CMAKE_SOURCE_DIR}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O2 -g")

file(GLOB SRC_LIST main.cpp)


set(LINK_LIBRARIES
	opencv_core
	opencv_imgproc
	opencv_highgui
	opencv_features2d
	opencv_calib3d
)

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(
	${PROJECT_NAME} -Wl,-E
	${LINK_LIBRARIES}
)

