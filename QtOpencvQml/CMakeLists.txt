cmake_minimum_required(VERSION 3.10)
project(FaceDetectionQtYocto LANGUAGES CXX)

add_executable(${PROJECT_NAME} main.cpp qmlType.cpp qml.qrc qmlType.h ../darknet.h)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

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
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Qt oackage

find_package(QT NAMES Qt5 COMPONENTS Core Quick Gui REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core Quick Gui REQUIRED)
qt5_add_resources(${PROJECT_NAME} qml.qrc)


set_target_properties(${PROJECT_NAME} PROPERTIES AUTOMOC TRUE)


target_link_libraries(${PROJECT_NAME} PRIVATE Qt5::Gui Qt5::Quick ${OpenCV_LIBS} /home/ulas/FaceDetection-YOLO4-jetsonNano/libdarknet.so)
