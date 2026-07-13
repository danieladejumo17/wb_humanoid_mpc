/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#include "humanoid_stylized_mpc/StylizedReferenceBuffer.h"

#include <algorithm>
#include <cmath>

#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>

#include "humanoid_stylized_mpc/StylizedReferenceBuilder.h"

namespace ocs2::humanoid {

namespace {

bool footInContactAtMode(size_t mode, int foot) {
  const auto flags = modeNumber2StanceLeg(mode);
  return flags[static_cast<size_t>(foot)];
}

}  // namespace

scalar_t StylizedReferenceSnapshot::phaseAt(scalar_t time) const {
  if (time <= anchorTime) {
    return 0.0;
  }
  return command.wrapPhase(time - anchorTime);
}

vector3_t StylizedReferenceSnapshot::cycleAnchorAt(scalar_t time) const {
  if (time <= anchorTime) {
    return anchorPose;
  }
  const auto cycles = static_cast<size_t>(std::floor((time - anchorTime) / command.cycleDuration));
  return stylized_reference::anchorAfterCycles(command, anchorPose, cycles);
}

const StylizedCartesianCurve* StylizedReferenceSnapshot::findCurve(const std::string& frameName) const {
  for (const auto& c : command.cartesianCurves) {
    if (c.frameName == frameName) {
      return &c;
    }
  }
  return nullptr;
}

std::vector<FootstepWindow> computeFootstepWindows(const StylizedGaitCommandData& command,
                                                   scalar_t windowFraction,
                                                   scalar_t maxWindow) {
  std::vector<FootstepWindow> windows;
  const auto& times = command.modeSwitchingTimes;
  const auto& modes = command.modeSequence;
  const scalar_t T = command.cycleDuration;

  for (const auto& step : command.footsteps) {
    if (step.touchdownTime <= 1e-9) {
      continue;  // initial stance positions have no swing before them
    }
    // Swing start: walk backwards (periodically) from the touchdown until the
    // foot is in contact again.
    scalar_t swingStart = step.touchdownTime;
    for (int guard = 0; guard < 2 * static_cast<int>(modes.size()); ++guard) {
      // Find the segment ending at swingStart (wrapping around the cycle).
      scalar_t target = swingStart;
      if (target <= 1e-9) target += T;
      size_t segment = modes.size();
      for (size_t i = 0; i < modes.size(); ++i) {
        if (std::abs(times[i + 1] - target) < 1e-6) {
          segment = i;
          break;
        }
      }
      if (segment == modes.size() || footInContactAtMode(modes[segment], step.foot)) {
        break;
      }
      swingStart = times[segment];
    }
    scalar_t swingDuration = step.touchdownTime - swingStart;
    if (swingDuration < 0.0) swingDuration += T;
    if (swingDuration <= 1e-6) swingDuration = T;  // degenerate: whole cycle

    const scalar_t window = std::min(windowFraction * swingDuration, maxWindow);
    FootstepWindow w;
    w.foot = step.foot;
    w.touchdownPhase = step.touchdownTime;
    w.windowStart = step.touchdownTime - window;  // may be negative; wrap at query
    w.positionGaitFrame = vector2_t(step.x, step.y);
    windows.push_back(w);
  }
  return windows;
}

bool StylizedReferenceBuffer::frameActive(const std::string& frameName) const {
  const auto snapshot = get();
  if (!snapshot->active) {
    return false;
  }
  const auto* curve = snapshot->findCurve(frameName);
  if (curve == nullptr) {
    return false;
  }
  return curve->activeXyz[0] || curve->activeXyz[1] || curve->activeXyz[2];
}

vector_t StylizedReferenceBuffer::cartesianParameters(const std::string& frameName, scalar_t time) const {
  const auto snapshot = get();
  vector_t params = vector_t::Zero(6);
  if (!snapshot->active) {
    return params;
  }
  const auto* curve = snapshot->findCurve(frameName);
  if (curve == nullptr) {
    return params;
  }

  const scalar_t tau = snapshot->phaseAt(time);
  const vector3_t anchor = snapshot->cycleAnchorAt(time);
  const auto& cmd = snapshot->command;

  // Gait-frame reference; z absolute.
  const scalar_t xg = cmd.samplePeriodic(curve->x, tau);
  const scalar_t yg = cmd.samplePeriodic(curve->y, tau);
  const scalar_t z = cmd.samplePeriodic(curve->z, tau);
  const vector2_t xyWorld =
      vector2_t(anchor.x(), anchor.y()) + stylized_reference::rotateToWorld(vector2_t(xg, yg), anchor.z());

  params(0) = xyWorld.x();
  params(1) = xyWorld.y();
  params(2) = z;
  for (size_t i = 0; i < 3; ++i) {
    params(3 + static_cast<Eigen::Index>(i)) = curve->activeXyz[i] ? std::sqrt(std::max(curve->weightsXyz[i], 0.0)) : 0.0;
  }
  return params;
}

bool StylizedReferenceBuffer::footstepWindowActive(int foot, scalar_t time) const {
  const auto snapshot = get();
  if (!snapshot->active || snapshot->command.footstepWeight <= 0.0 || time < snapshot->anchorTime) {
    return false;
  }
  const scalar_t tau = snapshot->phaseAt(time);
  const scalar_t T = snapshot->command.cycleDuration;
  for (const auto& w : snapshot->footstepWindows) {
    if (w.foot != foot) continue;
    // Window may wrap the cycle start (windowStart < 0).
    if (tau >= w.windowStart && tau <= w.touchdownPhase) return true;
    if (w.windowStart < 0.0 && tau - T >= w.windowStart) return true;
  }
  return false;
}

vector_t StylizedReferenceBuffer::footstepParameters(int foot, scalar_t time) const {
  const auto snapshot = get();
  vector_t params = vector_t::Zero(3);
  if (!snapshot->active) {
    return params;
  }
  const scalar_t tau = snapshot->phaseAt(time);
  const scalar_t T = snapshot->command.cycleDuration;
  vector3_t anchor = snapshot->cycleAnchorAt(time);

  // Pick the window containing tau (or the nearest upcoming touchdown).
  const FootstepWindow* best = nullptr;
  scalar_t bestDelta = 2.0 * T;
  bool bestWraps = false;
  for (const auto& w : snapshot->footstepWindows) {
    if (w.foot != foot) continue;
    scalar_t delta = w.touchdownPhase - tau;
    bool wraps = false;
    if (delta < -1e-9) {
      delta += T;  // touchdown happens next cycle
      wraps = true;
    }
    if (delta < bestDelta) {
      bestDelta = delta;
      best = &w;
      bestWraps = wraps;
    }
  }
  if (best == nullptr) {
    return params;
  }
  if (bestWraps) {
    anchor = stylized_reference::anchorAfterCycles(snapshot->command, anchor, 1);
  }
  const vector2_t xyWorld = vector2_t(anchor.x(), anchor.y()) +
                            stylized_reference::rotateToWorld(best->positionGaitFrame, anchor.z());
  params(0) = xyWorld.x();
  params(1) = xyWorld.y();
  params(2) = std::sqrt(std::max(snapshot->command.footstepWeight, 0.0));
  return params;
}

}  // namespace ocs2::humanoid
