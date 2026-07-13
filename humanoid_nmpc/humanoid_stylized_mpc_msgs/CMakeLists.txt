cmake_minimum_required(VERSION 3.14)
project(humanoid_stylized_mpc_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ocs2_ros2_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CurveChannel.msg"
  "msg/CartesianCurve.msg"
  "msg/Footstep.msg"
  "msg/StylizedGaitCommand.msg"
  "msg/StylizedGaitStatus.msg"
  DEPENDENCIES std_msgs ocs2_ros2_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
