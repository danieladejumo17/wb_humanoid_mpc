/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#include "humanoid_stylized_mpc/StylizedReference.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>

namespace ocs2::humanoid {

std::string StylizedGaitCommandData::validate() const {
  std::ostringstream err;
  if (cycleDuration <= 0.0) {
    err << "cycle_duration must be positive; ";
  }
  if (timeGrid.size() < 2) {
    err << "time_grid needs at least 2 samples; ";
  } else {
    if (std::abs(timeGrid.front()) > 1e-9 || std::abs(timeGrid.back() - cycleDuration) > 1e-6) {
      err << "time_grid must span [0, cycle_duration]; ";
    }
    if (!std::is_sorted(timeGrid.begin(), timeGrid.end())) {
      err << "time_grid must be increasing; ";
    }
  }
  if (modeSequence.empty() || modeSwitchingTimes.size() != modeSequence.size() + 1) {
    err << "mode schedule needs len(event_times) == len(mode_sequence)+1 > 1; ";
  } else {
    if (std::abs(modeSwitchingTimes.front()) > 1e-9 || std::abs(modeSwitchingTimes.back() - cycleDuration) > 1e-6) {
      err << "mode schedule must span [0, cycle_duration]; ";
    }
    if (!std::is_sorted(modeSwitchingTimes.begin(), modeSwitchingTimes.end())) {
      err << "mode schedule event times must be increasing; ";
    }
    for (size_t m : modeSequence) {
      if (m > ModeNumber::STANCE) {
        err << "invalid mode number " << m << "; ";
        break;
      }
    }
  }
  for (const auto& curveVec : {jointCurves, baseCurves}) {
    for (const auto& c : curveVec) {
      if (c.values.size() != timeGrid.size()) {
        err << "curve '" << c.name << "' length != time_grid length; ";
      }
    }
  }
  for (const auto& c : cartesianCurves) {
    if (c.x.size() != timeGrid.size() || c.y.size() != timeGrid.size() || c.z.size() != timeGrid.size()) {
      err << "cartesian curve '" << c.frameName << "' length != time_grid length; ";
    }
  }
  if (enable) {
    bool hasLeftInit = false;
    bool hasRightInit = false;
    for (const auto& s : footsteps) {
      if (s.touchdownTime <= 1e-9) {
        (s.foot == 0 ? hasLeftInit : hasRightInit) = true;
      }
      if (s.touchdownTime < 0.0 || s.touchdownTime > cycleDuration + 1e-6) {
        err << "footstep touchdown_time outside [0, cycle_duration]; ";
      }
    }
    if (!footsteps.empty() && (!hasLeftInit || !hasRightInit)) {
      err << "footsteps must include initial (t=0) positions for both feet; ";
    }
  }
  return err.str();
}

scalar_t StylizedGaitCommandData::wrapPhase(scalar_t tau) const {
  scalar_t wrapped = std::fmod(tau, cycleDuration);
  if (wrapped < 0.0) {
    wrapped += cycleDuration;
  }
  return wrapped;
}

scalar_t StylizedGaitCommandData::samplePeriodic(const std::vector<scalar_t>& values, scalar_t tau) const {
  const scalar_t t = wrapPhase(tau);
  const auto it = std::upper_bound(timeGrid.begin(), timeGrid.end(), t);
  if (it == timeGrid.begin()) {
    return values.front();
  }
  if (it == timeGrid.end()) {
    return values.back();
  }
  const size_t i = static_cast<size_t>(it - timeGrid.begin()) - 1;
  const scalar_t h = timeGrid[i + 1] - timeGrid[i];
  const scalar_t alpha = (h > 1e-12) ? (t - timeGrid[i]) / h : 0.0;
  return (1.0 - alpha) * values[i] + alpha * values[i + 1];
}

scalar_t StylizedGaitCommandData::sampleDerivativePeriodic(const std::vector<scalar_t>& values, scalar_t tau) const {
  const scalar_t dt = (timeGrid.size() > 1) ? (timeGrid[1] - timeGrid[0]) : cycleDuration;
  const scalar_t vPlus = samplePeriodic(values, tau + dt);
  const scalar_t vMinus = samplePeriodic(values, tau - dt);
  return (vPlus - vMinus) / (2.0 * dt);
}

vector2_t StylizedGaitCommandData::cycleDisplacement() const {
  vector2_t displacement = vector2_t::Zero();
  int feetWithData = 0;
  for (int foot = 0; foot < 2; ++foot) {
    const StylizedFootstep* initial = nullptr;
    const StylizedFootstep* last = nullptr;
    for (const auto& s : footsteps) {
      if (s.foot != foot) continue;
      if (s.touchdownTime <= 1e-9) {
        initial = &s;
      }
      if (last == nullptr || s.touchdownTime >= last->touchdownTime) {
        last = &s;
      }
    }
    if (initial != nullptr && last != nullptr) {
      displacement += vector2_t(last->x - initial->x, last->y - initial->y);
      ++feetWithData;
    }
  }
  if (feetWithData > 0) {
    displacement /= static_cast<scalar_t>(feetWithData);
  }
  return displacement;
}

bool StylizedGaitCommandData::sameTiming(const StylizedGaitCommandData& other, scalar_t tol) const {
  if (std::abs(cycleDuration - other.cycleDuration) > tol) return false;
  if (modeSequence != other.modeSequence) return false;
  if (modeSwitchingTimes.size() != other.modeSwitchingTimes.size()) return false;
  for (size_t i = 0; i < modeSwitchingTimes.size(); ++i) {
    if (std::abs(modeSwitchingTimes[i] - other.modeSwitchingTimes[i]) > tol) return false;
  }
  return true;
}

}  // namespace ocs2::humanoid
