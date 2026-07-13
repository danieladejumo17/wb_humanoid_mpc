/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#pragma once

#include <ocs2_core/reference/TargetTrajectories.h>

#include <humanoid_common_mpc/common/MpcRobotModelBase.h>

#include "humanoid_stylized_mpc/StylizedReference.h"

namespace ocs2::humanoid::stylized_reference {

/** Rotates a gait-frame planar vector into world frame by yaw. */
vector2_t rotateToWorld(const vector2_t& v, scalar_t yaw);

/**
 * Gait-frame anchor pose (x, y, yaw) advanced numCycles cycles by the
 * command's per-cycle footstep displacement (rotated by the anchor yaw).
 */
vector3_t anchorAfterCycles(const StylizedGaitCommandData& command, const vector3_t& anchorPose, size_t numCycles);

/**
 * Reference state at cycle phase tau in [0, T), composed onto cycleAnchor:
 * defaults (linear planar progress, defaultBaseHeight, defaultJointState)
 * overridden by the command's active base/joint curves. Velocities of active
 * base channels come from curve derivatives; joint velocities stay zero.
 */
vector_t buildStateAtPhase(const MpcRobotModelBase<scalar_t>& model,
                           const vector_t& defaultJointState,
                           scalar_t defaultBaseHeight,
                           const StylizedGaitCommandData& command,
                           scalar_t tau,
                           const vector3_t& cycleAnchor);

/**
 * Tiles the cycle over [initTime, finalTime] on a dt grid. Times before
 * anchorTime hold the cycle-start pose (pre-roll before the gait begins).
 */
TargetTrajectories buildTargetTrajectories(const MpcRobotModelBase<scalar_t>& model,
                                           const vector_t& defaultJointState,
                                           scalar_t defaultBaseHeight,
                                           const StylizedGaitCommandData& command,
                                           scalar_t anchorTime,
                                           const vector3_t& anchorPose,
                                           scalar_t initTime,
                                           scalar_t finalTime,
                                           scalar_t dt);

/** Standing-still target holding the current planar pose and default posture. */
TargetTrajectories buildStanceTargetTrajectories(const MpcRobotModelBase<scalar_t>& model,
                                                 const vector_t& defaultJointState,
                                                 scalar_t defaultBaseHeight,
                                                 scalar_t initTime,
                                                 scalar_t finalTime,
                                                 const vector_t& initState);

}  // namespace ocs2::humanoid::stylized_reference
