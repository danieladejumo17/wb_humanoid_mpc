/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "humanoid_stylized_mpc/StylizedReference.h"

namespace ocs2::humanoid {

/** Footstep placement activation window (absolute cycle phases). */
struct FootstepWindow {
  int foot{0};
  scalar_t touchdownPhase{0.0};  // cycle phase of touchdown
  scalar_t windowStart{0.0};     // cycle phase where the placement cost turns on
  vector2_t positionGaitFrame{vector2_t::Zero()};
};

/**
 * Immutable per-solve snapshot of the stylized reference, in a form the
 * cartesian/footstep costs can query by absolute solver time. Published by
 * StylizedMotionManager::preSolverRun; read by cost getParameters()/isActive()
 * during the same solve.
 */
struct StylizedReferenceSnapshot {
  bool active{false};
  scalar_t anchorTime{0.0};
  vector3_t anchorPose{vector3_t::Zero()};  // gait frame (x, y, yaw)
  StylizedGaitCommandData command;
  std::vector<FootstepWindow> footstepWindows;

  /** Cycle phase in [0, T) for an absolute solver time (times before the
   * anchor clamp to phase 0). */
  scalar_t phaseAt(scalar_t time) const;

  /** Gait-frame anchor of the cycle containing `time`. */
  vector3_t cycleAnchorAt(scalar_t time) const;

  const StylizedCartesianCurve* findCurve(const std::string& frameName) const;
};

/** Computes per-footstep activation windows from the command's mode template:
 * the window spans the last `windowFraction` of the swing preceding each
 * touchdown (capped at maxWindow seconds). */
std::vector<FootstepWindow> computeFootstepWindows(const StylizedGaitCommandData& command,
                                                   scalar_t windowFraction = 0.8,
                                                   scalar_t maxWindow = 0.5);

/**
 * Shared handle between the motion manager (writer) and the costs (readers).
 * The MPC solve is synchronous after preSolverRun, but a mutex keeps the
 * handle safe regardless of threading assumptions.
 */
class StylizedReferenceBuffer {
 public:
  StylizedReferenceBuffer() : snapshot_(std::make_shared<const StylizedReferenceSnapshot>()) {}

  void update(StylizedReferenceSnapshot snapshot) {
    auto ptr = std::make_shared<const StylizedReferenceSnapshot>(std::move(snapshot));
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_ = std::move(ptr);
  }

  std::shared_ptr<const StylizedReferenceSnapshot> get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
  }

  // ---- cost-facing queries -------------------------------------------------

  /** Whether the given frame has at least one active cartesian axis. */
  bool frameActive(const std::string& frameName) const;

  /**
   * [pRefWorld(3), sqrtWeights(3)] for a frame at an absolute time. Inactive
   * axes get weight 0 and reference 0.
   */
  vector_t cartesianParameters(const std::string& frameName, scalar_t time) const;

  /** Whether the placement cost for foot (0=L,1=R) is inside its window. */
  bool footstepWindowActive(int foot, scalar_t time) const;

  /** [xyRefWorld(2), sqrtWeight(1)] for the footstep targeted at `time`. */
  vector_t footstepParameters(int foot, scalar_t time) const;

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<const StylizedReferenceSnapshot> snapshot_;
};

}  // namespace ocs2::humanoid
