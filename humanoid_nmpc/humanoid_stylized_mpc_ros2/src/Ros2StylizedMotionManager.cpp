/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#include "humanoid_stylized_mpc_ros2/Ros2StylizedMotionManager.h"

namespace ocs2::humanoid {

StylizedGaitCommandData fromMsg(const humanoid_stylized_mpc_msgs::msg::StylizedGaitCommand& msg) {
  StylizedGaitCommandData data;
  data.commandId = msg.command_id;
  data.enable = msg.enable;
  data.cycleDuration = msg.cycle_duration;
  data.timeGrid.assign(msg.time_grid.begin(), msg.time_grid.end());

  data.modeSwitchingTimes.assign(msg.cycle_mode_schedule.event_times.begin(), msg.cycle_mode_schedule.event_times.end());
  data.modeSequence.reserve(msg.cycle_mode_schedule.mode_sequence.size());
  for (const auto mode : msg.cycle_mode_schedule.mode_sequence) {
    data.modeSequence.push_back(static_cast<size_t>(mode));
  }

  auto convertChannel = [](const humanoid_stylized_mpc_msgs::msg::CurveChannel& c) {
    StylizedCurveChannel out;
    out.name = c.name;
    out.weight = c.weight;
    if (c.active) {
      out.values.assign(c.values.begin(), c.values.end());
    }
    return out;
  };
  for (const auto& c : msg.joint_curves) {
    if (c.active) data.jointCurves.push_back(convertChannel(c));
  }
  for (const auto& c : msg.base_curves) {
    if (c.active) data.baseCurves.push_back(convertChannel(c));
  }

  for (const auto& c : msg.cartesian_curves) {
    StylizedCartesianCurve out;
    out.frameName = c.frame_name;
    for (size_t i = 0; i < 3; ++i) {
      out.activeXyz[i] = c.active_xyz[i];
      out.weightsXyz[i] = c.weights_xyz[i];
    }
    out.x.assign(c.x.begin(), c.x.end());
    out.y.assign(c.y.begin(), c.y.end());
    out.z.assign(c.z.begin(), c.z.end());
    data.cartesianCurves.push_back(std::move(out));
  }

  for (const auto& s : msg.footsteps) {
    StylizedFootstep step;
    step.foot = s.foot;
    step.touchdownTime = s.touchdown_time;
    step.x = s.position_xy[0];
    step.y = s.position_xy[1];
    data.footsteps.push_back(step);
  }

  data.footstepWeight = msg.footstep_weight;
  data.swingApexHeights.assign(msg.swing_apex_heights.begin(), msg.swing_apex_heights.end());
  return data;
}

humanoid_stylized_mpc_msgs::msg::StylizedGaitStatus toMsg(const StylizedGaitStatusData& status) {
  humanoid_stylized_mpc_msgs::msg::StylizedGaitStatus msg;
  msg.last_command_id = status.lastCommandId;
  msg.accepted = status.accepted;
  msg.message = status.message;
  msg.active = status.active;
  msg.anchor_time = status.anchorTime;
  msg.cycle_duration = status.cycleDuration;
  return msg;
}

void Ros2StylizedMotionManager::subscribe(const rclcpp::Node::SharedPtr& node, const std::string& robotName) {
  statusPublisher_ = node->create_publisher<humanoid_stylized_mpc_msgs::msg::StylizedGaitStatus>(
      robotName + "/stylized_gait_status", rclcpp::QoS(10).reliable());

  setStatusCallback([publisher = statusPublisher_](const StylizedGaitStatusData& status) {
    publisher->publish(toMsg(status));
  });

  auto callback = [this](const humanoid_stylized_mpc_msgs::msg::StylizedGaitCommand::SharedPtr msg) {
    this->setCommand(fromMsg(*msg));
  };
  commandSubscriber_ = node->create_subscription<humanoid_stylized_mpc_msgs::msg::StylizedGaitCommand>(
      robotName + "/stylized_gait_command", rclcpp::QoS(1).reliable(), callback);
}

}  // namespace ocs2::humanoid
