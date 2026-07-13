/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#pragma once

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <humanoid_stylized_mpc/StylizedMotionManager.h>
#include <humanoid_stylized_mpc_msgs/msg/stylized_gait_command.hpp>
#include <humanoid_stylized_mpc_msgs/msg/stylized_gait_status.hpp>

namespace ocs2::humanoid {

/** Message <-> data conversions (exposed for the tests and the backend docs). */
StylizedGaitCommandData fromMsg(const humanoid_stylized_mpc_msgs::msg::StylizedGaitCommand& msg);
humanoid_stylized_mpc_msgs::msg::StylizedGaitStatus toMsg(const StylizedGaitStatusData& status);

/**
 * StylizedMotionManager wired to ROS2: subscribes
 * <robot>/stylized_gait_command and publishes <robot>/stylized_gait_status.
 */
class Ros2StylizedMotionManager : public StylizedMotionManager {
 public:
  using StylizedMotionManager::StylizedMotionManager;

  void subscribe(const rclcpp::Node::SharedPtr& node, const std::string& robotName);

 private:
  rclcpp::Subscription<humanoid_stylized_mpc_msgs::msg::StylizedGaitCommand>::SharedPtr commandSubscriber_;
  rclcpp::Publisher<humanoid_stylized_mpc_msgs::msg::StylizedGaitStatus>::SharedPtr statusPublisher_;
};

}  // namespace ocs2::humanoid
