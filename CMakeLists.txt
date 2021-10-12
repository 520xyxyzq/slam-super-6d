cmake_minimum_required(VERSION 2.8.3)
project(slam-super-6D)

find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)


add_executable(test_load src/test.cpp)

target_link_libraries(test_load gtsam)

target_include_directories(test_load PUBLIC ${GTSAM_INCLUDE_DIR})
