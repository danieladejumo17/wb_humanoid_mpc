/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#pragma once

#include <array>
#include <string>
#include <vector>

#include <ocs2_core/Types.h>

#include <humanoid_common_mpc/common/Types.h>
#include <humanoid_common_mpc/gait/ModeSequenceTemplate.h>

namespace ocs2::humanoid {

struct StylizedCurveChannel {
  std::string name;  // "joint/<name>" or "base/{px,py,pz,yaw,pitch,roll}"
  scalar_t weight{0.0};
  std::vector<scalar_t> values;  // sampled on the command time grid
};

struct StylizedCartesianCurve {
  std::string frameName;
  std::array<bool, 3> activeXyz{{false, false, false}};
  std::array<scalar_t, 3> weightsXyz{{0.0, 0.0, 0.0}};
  std::vector<scalar_t> x;  // gait frame
  std::vector<scalar_t> y;  // gait frame
  std::vector<scalar_t> z;  // absolute above ground
};

struct StylizedFootstep {
  int foot{0};  // 0 = left, 1 = right
  scalar_t touchdownTime{0.0};
  scalar_t x{0.0};
  scalar_t y{0.0};
};

/**
 * ROS-free image of a StylizedGaitCommand message: one gait cycle described by
 * a shared uniform time grid, contact mode template, sampled reference curves
 * and footsteps. Provides periodic sampling helpers used to tile the cycle
 * over the MPC horizon.
 */
struct StylizedGaitCommandData {
  size_t commandId{0};
  bool enable{false};
  scalar_t cycleDuration{0.0};
  std::vector<scalar_t> timeGrid;

  // Template-style: switchingTimes include 0 and cycleDuration.
  std::vector<scalar_t> modeSwitchingTimes;
  std::vector<size_t> modeSequence;

  std::vector<StylizedCurveChannel> jointCurves;
  std::vector<StylizedCurveChannel> baseCurves;
  std::vector<StylizedCartesianCurve> cartesianCurves;
  std::vector<StylizedFootstep> footsteps;

  scalar_t footstepWeight{0.0};
  std::vector<scalar_t> swingApexHeights;

  /** Returns an empty string when valid, else a human-readable reason. */
  std::string validate() const;

  ModeSequenceTemplate toModeSequenceTemplate() const { return ModeSequenceTemplate(modeSwitchingTimes, modeSequence); }

  /**
   * Linear interpolation of a sampled curve at cycle phase tau, with periodic
   * wraparound (tau is wrapped into [0, cycleDuration)).
   */
  scalar_t samplePeriodic(const std::vector<scalar_t>& values, scalar_t tau) const;

  /**
   * Central-difference derivative of a sampled curve at phase tau, treating
   * the curve as periodic.
   */
  scalar_t sampleDerivativePeriodic(const std::vector<scalar_t>& values, scalar_t tau) const;

  /** Wraps an arbitrary phase into [0, cycleDuration). */
  scalar_t wrapPhase(scalar_t tau) const;

  /**
   * Per-cycle planar displacement (gait frame): average over both feet of
   * (last touchdown position - initial position). Feet should agree; the
   * average tolerates small user misclosure.
   */
  vector2_t cycleDisplacement() const;

  /** Whether the contact timing (duration + template) equals another command's. */
  bool sameTiming(const StylizedGaitCommandData& other, scalar_t tol = 1e-6) const;
};

}  // namespace ocs2::humanoid
