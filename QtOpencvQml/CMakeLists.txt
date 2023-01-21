cmake_minimum_required(VERSION 3.10)
project(FaceDetectionQtYocto LANGUAGES CXX)

# thread
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# OpenCV setup
find_package(OpenCV REQUIRED)
message("Opencv version ${OpenCV_VERSION}")
if(OpenCV_VERSION VERSION_LESS "4.4.0")
        message("Opencv version may be not suitable")
endif()

message("OpenCV libs ${OpenCV_LIBS}")
message("OpenCV directory ${OpenCV_DIR}")

# Qt oackage
find_package(QT NAMES Qt5 COMPONENTS Core Quick Gui REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core Quick Gui REQUIRED)
qt5_add_resources(QT_RESOURCES qml.qrc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -std=c++14 -fstrict-aliasing -pedantic-errors -pedantic -Wno-deprecated-declarations -Wno-unused-variable")

add_executable(${PROJECT_NAME} main.cpp qmlType.cpp qmlType.h ../darknet.h ${QT_RESOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE Qt5::Gui Qt5::Quick ${OpenCV_LIBS} /home/ulas/FaceDetection-YOLO4-jetsonNano/libdarknet.so)
