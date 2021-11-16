cmake_minimum_required(VERSION 2.8.3)
project(slam-super-6D)

find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)


add_executable(solve src/GraphSolver.cpp)

target_link_libraries(solve gtsam)

target_include_directories(solve PUBLIC ${GTSAM_INCLUDE_DIR})
