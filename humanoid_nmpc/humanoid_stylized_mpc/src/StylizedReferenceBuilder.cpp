/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#include "humanoid_stylized_mpc/StylizedReferenceBuilder.h"

#include <cmath>

namespace ocs2::humanoid::stylized_reference {

vector2_t rotateToWorld(const vector2_t& v, scalar_t yaw) {
  const scalar_t c = std::cos(yaw);
  const scalar_t s = std::sin(yaw);
  return vector2_t(c * v.x() - s * v.y(), s * v.x() + c * v.y());
}

vector3_t anchorAfterCycles(const StylizedGaitCommandData& command, const vector3_t& anchorPose, size_t numCycles) {
  vector3_t anchor = anchorPose;
  const vector2_t displacement = command.cycleDisplacement();
  for (size_t k = 0; k < numCycles; ++k) {
    const vector2_t worldDelta = rotateToWorld(displacement, anchor.z());
    anchor.x() += worldDelta.x();
    anchor.y() += worldDelta.y();
    // Yaw is unchanged: cycles do not rotate the gait frame (v1 limitation).
  }
  return anchor;
}

vector_t buildStateAtPhase(const MpcRobotModelBase<scalar_t>& model,
                           const vector_t& defaultJointState,
                           scalar_t defaultBaseHeight,
                           const StylizedGaitCommandData& command,
                           scalar_t tau,
                           const vector3_t& cycleAnchor) {
  const scalar_t T = command.cycleDuration;
  const vector2_t displacement = command.cycleDisplacement();

  vector_t state = vector_t::Zero(model.getStateDim());

  // --- base pose defaults: linear progress along the footstep displacement.
  vector2_t xyGait = displacement * (tau / T);
  scalar_t z = defaultBaseHeight;
  scalar_t yawOffset = 0.0;
  scalar_t pitch = 0.0;
  scalar_t roll = 0.0;

  vector2_t vXyGait = displacement / T;
  scalar_t vz = 0.0;
  scalar_t yawRate = 0.0;

  vector_t jointAngles = defaultJointState;

  for (const auto& c : command.baseCurves) {
    const scalar_t v = command.samplePeriodic(c.values, tau);
    const scalar_t dv = command.sampleDerivativePeriodic(c.values, tau);
    if (c.name == "base/px") {
      xyGait.x() = v;
      vXyGait.x() = dv;
    } else if (c.name == "base/py") {
      xyGait.y() = v;
      vXyGait.y() = dv;
    } else if (c.name == "base/pz") {
      z = v;
      vz = dv;
    } else if (c.name == "base/yaw") {
      yawOffset = v;
      yawRate = dv;
    } else if (c.name == "base/pitch") {
      pitch = v;
    } else if (c.name == "base/roll") {
      roll = v;
    }
  }

  for (const auto& c : command.jointCurves) {
    constexpr size_t prefixLen = 6;  // "joint/"
    if (c.name.size() <= prefixLen) continue;
    const size_t jointIndex = model.getJointIndex(c.name.substr(prefixLen));
    jointAngles(jointIndex) = command.samplePeriodic(c.values, tau);
  }

  const scalar_t anchorYaw = cycleAnchor.z();
  const vector2_t xyWorld = vector2_t(cycleAnchor.x(), cycleAnchor.y()) + rotateToWorld(xyGait, anchorYaw);
  const vector2_t vXyWorld = rotateToWorld(vXyGait, anchorYaw);

  vector6_t basePose;
  basePose << xyWorld.x(), xyWorld.y(), z, anchorYaw + yawOffset, pitch, roll;
  model.setBasePose(state, basePose);
  model.setJointAngles(state, jointAngles);

  // Base velocity block [v_xyz, dEuler_zyx] sits right after the joint angles
  // in the whole-body state layout.
  const size_t velStart = model.getJointStartindex() + model.getJointDim();
  state(velStart + 0) = vXyWorld.x();
  state(velStart + 1) = vXyWorld.y();
  state(velStart + 2) = vz;
  state(velStart + 3) = yawRate;

  return state;
}

TargetTrajectories buildTargetTrajectories(const MpcRobotModelBase<scalar_t>& model,
                                           const vector_t& defaultJointState,
                                           scalar_t defaultBaseHeight,
                                           const StylizedGaitCommandData& command,
                                           scalar_t anchorTime,
                                           const vector3_t& anchorPose,
                                           scalar_t initTime,
                                           scalar_t finalTime,
                                           scalar_t dt) {
  const scalar_t T = command.cycleDuration;

  scalar_array_t timeTrajectory;
  vector_array_t stateTrajectory;

  scalar_t t = initTime;
  while (true) {
    const scalar_t clampedT = std::min(t, finalTime);
    const scalar_t sinceAnchor = clampedT - anchorTime;
    vector_t state;
    if (sinceAnchor < 0.0) {
      state = buildStateAtPhase(model, defaultJointState, defaultBaseHeight, command, 0.0, anchorPose);
    } else {
      const auto cycles = static_cast<size_t>(std::floor(sinceAnchor / T));
      const scalar_t tau = sinceAnchor - static_cast<scalar_t>(cycles) * T;
      state = buildStateAtPhase(model, defaultJointState, defaultBaseHeight, command, tau,
                                anchorAfterCycles(command, anchorPose, cycles));
    }
    timeTrajectory.push_back(clampedT);
    stateTrajectory.push_back(std::move(state));
    if (clampedT >= finalTime) break;
    t += dt;
  }

  const vector_array_t inputTrajectory(timeTrajectory.size(), vector_t::Zero(model.getInputDim()));
  return TargetTrajectories(timeTrajectory, stateTrajectory, inputTrajectory);
}

TargetTrajectories buildStanceTargetTrajectories(const MpcRobotModelBase<scalar_t>& model,
                                                 const vector_t& defaultJointState,
                                                 scalar_t defaultBaseHeight,
                                                 scalar_t initTime,
                                                 scalar_t finalTime,
                                                 const vector_t& initState) {
  const vector6_t currentPose = model.getBasePose(initState);
  vector6_t targetPose;
  targetPose << currentPose(0), currentPose(1), defaultBaseHeight, currentPose(3), 0.0, 0.0;

  vector_t state = vector_t::Zero(model.getStateDim());
  model.setBasePose(state, targetPose);
  model.setJointAngles(state, defaultJointState);

  const scalar_array_t timeTrajectory{initTime, finalTime};
  const vector_array_t stateTrajectory(2, state);
  const vector_array_t inputTrajectory(2, vector_t::Zero(model.getInputDim()));
  return TargetTrajectories(timeTrajectory, stateTrajectory, inputTrajectory);
}

}  // namespace ocs2::humanoid::stylized_reference
