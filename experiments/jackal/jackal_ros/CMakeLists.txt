cmake_minimum_required(VERSION 3.0.2)
project(jackal_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  vicon_bridge
  apriltag_ros
  std_msgs
  sensor_msgs
)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES jackal_ros
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
CATKIN_DEPENDS
message_runtime
vicon_bridge
apriltag_ros
std_msgs
sensor_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
