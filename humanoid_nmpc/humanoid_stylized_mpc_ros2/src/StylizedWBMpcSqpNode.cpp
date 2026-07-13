/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.

Stylized whole-body MPC solver node. Mirrors WBMpcSqpNode but is driven by
user-authored gait cycles (<robot>/stylized_gait_command) instead of velocity
teleop, and disables the procedural arm-swing reference.
******************************************************************************/

#include <rclcpp/rclcpp.hpp>

#include <ocs2_ros2_interfaces/mpc/MPC_ROS_Interface.h>
#include <ocs2_sqp/SqpMpc.h>

#include <humanoid_wb_mpc/WBMpcInterface.h>

#include <humanoid_stylized_mpc/cost/FootstepPlacementCost.h>
#include <humanoid_stylized_mpc/cost/StylizedTaskSpaceCost.h>

#include "humanoid_stylized_mpc_ros2/Ros2StylizedMotionManager.h"

using namespace ocs2;
using namespace ocs2::humanoid;

int main(int argc, char** argv) {
  std::vector<std::string> programArgs = rclcpp::remove_ros_arguments(argc, argv);
  if (programArgs.size() < 5) {
    throw std::runtime_error("Usage: stylized_wb_mpc_sqp_node robot_name task_file reference_file urdf_file [gait_file]");
  }

  const std::string robotName(programArgs[1]);
  const std::string taskFile(programArgs[2]);
  const std::string referenceFile(programArgs[3]);
  const std::string urdfFile(programArgs[4]);

  rclcpp::init(argc, argv);

  // Robot interface
  WBMpcInterface interface(taskFile, urdfFile, referenceFile);

  // The stylized joint curves fully define the arm references; the built-in
  // velocity-scaled arm swing would fight them.
  interface.getSwitchedModelReferenceManagerPtr()->setArmSwingReferenceActive(false);

  rclcpp::Node::SharedPtr nodeHandle = std::make_shared<rclcpp::Node>(robotName + "_stylized_wb_mpc");

  auto stylizedMotionManager = std::make_shared<Ros2StylizedMotionManager>(
      interface.getSwitchedModelReferenceManagerPtr(), interface.getMpcRobotModel(), referenceFile);
  stylizedMotionManager->subscribe(nodeHandle, robotName);

  // Register the styling costs on a copy of the OCP (the stock problem stays
  // untouched inside the interface). References/weights arrive via runtime
  // parameters from the shared reference buffer.
  OptimalControlProblem problem = interface.getOptimalControlProblem();
  const auto buffer = stylizedMotionManager->getReferenceBuffer();

  for (const std::string frameName : {"foot_l_contact", "foot_r_contact", "left_rubber_hand", "right_rubber_hand"}) {
    problem.costPtr->add("stylized_pos_" + frameName,
                         std::make_unique<StylizedTaskSpaceCost>(buffer, interface.getPinocchioInterface(),
                                                                 interface.getMpcRobotModelAD(), frameName,
                                                                 interface.modelSettings()));
  }
  for (size_t contactIndex : {0u, 1u}) {
    problem.costPtr->add("footstep_placement_" + interface.modelSettings().contactNames6DoF[contactIndex],
                         std::make_unique<FootstepPlacementCost>(buffer, interface.getPinocchioInterface(),
                                                                 interface.getMpcRobotModelAD(), contactIndex,
                                                                 interface.modelSettings()));
  }

  // MPC
  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(), problem, interface.getInitializer());

  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());
  mpc.getSolverPtr()->addSynchronizedModule(stylizedMotionManager);

  MPC_ROS_Interface mpcNode(mpc, robotName);
  auto qos = rclcpp::QoS(1);
  qos.best_effort();
  mpcNode.launchNodes(nodeHandle, qos);

  return 0;
}
