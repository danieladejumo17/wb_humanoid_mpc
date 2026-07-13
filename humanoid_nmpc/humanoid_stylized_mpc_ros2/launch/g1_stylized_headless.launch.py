"""Headless stylized whole-body MPC for the Unitree G1: the stylized solver
node plus the model-integrating dummy sim. No RViz, no teleop GUIs — meant for
the gait-studio backend and the integration tests."""

import os

from ament_index_python.packages import get_package_share_directory

import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    g1_wb_mpc_dir = get_package_share_directory("g1_wb_mpc")
    g1_description_dir = get_package_share_directory("g1_description")
    common_mpc_dir = get_package_share_directory("humanoid_common_mpc")

    args = [
        DeclareLaunchArgument("robot_name", default_value="g1"),
        DeclareLaunchArgument(
            "config_name",
            default_value=os.path.join(g1_wb_mpc_dir, "config/mpc/task_stylized.info"),
        ),
        DeclareLaunchArgument(
            "target_command_file",
            default_value=os.path.join(g1_wb_mpc_dir, "config/command/reference.info"),
        ),
        DeclareLaunchArgument(
            "description_name",
            default_value=os.path.join(g1_description_dir, "urdf/g1_29dof.urdf"),
        ),
        DeclareLaunchArgument(
            "target_gait_file",
            default_value=os.path.join(common_mpc_dir, "config/command/gait.info"),
        ),
    ]

    node_args = [
        LaunchConfiguration("robot_name"),
        LaunchConfiguration("config_name"),
        LaunchConfiguration("target_command_file"),
        LaunchConfiguration("description_name"),
        LaunchConfiguration("target_gait_file"),
    ]

    mpc_node = launch_ros.actions.Node(
        package="humanoid_stylized_mpc_ros2",
        executable="stylized_wb_mpc_sqp_node",
        name="stylized_wb_mpc_sqp_node",
        output="screen",
        arguments=node_args,
    )

    dummy_sim_node = launch_ros.actions.Node(
        package="humanoid_wb_mpc_ros2",
        executable="humanoid_wb_mpc_dummy_sim_node",
        name="humanoid_wb_mpc_dummy_sim_node",
        output="screen",
        arguments=node_args,
    )

    return launch.LaunchDescription(args + [mpc_node, dummy_sim_node])
