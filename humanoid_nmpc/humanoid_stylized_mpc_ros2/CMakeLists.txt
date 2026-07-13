cmake_minimum_required(VERSION 3.14)
project(humanoid_stylized_mpc_ros2)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)
endif ()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem log log_setup)
find_package(PkgConfig REQUIRED)
pkg_check_modules(pinocchio REQUIRED pinocchio)
find_package(rclcpp REQUIRED)
find_package(humanoid_stylized_mpc REQUIRED)
find_package(humanoid_stylized_mpc_msgs REQUIRED)
find_package(humanoid_common_mpc REQUIRED)
find_package(humanoid_wb_mpc REQUIRED)
find_package(ocs2_ros2_interfaces REQUIRED)
find_package(ocs2_sqp REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(dependencies
        rclcpp
        humanoid_stylized_mpc
        humanoid_stylized_mpc_msgs
        humanoid_common_mpc
        humanoid_wb_mpc
        ocs2_ros2_interfaces
        ocs2_sqp
)

include_directories(
        include
        ${EIGEN3_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
        src/Ros2StylizedMotionManager.cpp
)
ament_target_dependencies(${PROJECT_NAME} ${dependencies})

add_executable(stylized_wb_mpc_sqp_node src/StylizedWBMpcSqpNode.cpp)
target_link_libraries(stylized_wb_mpc_sqp_node ${PROJECT_NAME})
ament_target_dependencies(stylized_wb_mpc_sqp_node ${dependencies})

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)

install(TARGETS stylized_wb_mpc_sqp_node
        DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/${PROJECT_NAME}
        DESTINATION include/
)

install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${dependencies})

ament_package()
