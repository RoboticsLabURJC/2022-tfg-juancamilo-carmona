cmake_minimum_required(VERSION 2.8.3)
project(carla_teleop)

find_package(ros_environment REQUIRED)
set(ROS_VERSION $ENV{ROS_VERSION})


find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
ament_export_dependencies(rclpy)
# Install launch files.
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

ament_package()


