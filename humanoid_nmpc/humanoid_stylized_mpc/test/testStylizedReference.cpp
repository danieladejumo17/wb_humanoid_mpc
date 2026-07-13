/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#include <gtest/gtest.h>

#include "humanoid_stylized_mpc/StylizedReference.h"

#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>

using namespace ocs2;
using namespace ocs2::humanoid;

namespace {

// Default walk cycle (gait.info 'walk'): LF, DS, RF, DS over T = 1.4 s.
StylizedGaitCommandData makeWalkCommand() {
  StylizedGaitCommandData cmd;
  cmd.commandId = 1;
  cmd.enable = true;
  cmd.cycleDuration = 1.4;
  const size_t n = 71;
  cmd.timeGrid.resize(n);
  for (size_t i = 0; i < n; ++i) {
    cmd.timeGrid[i] = 1.4 * static_cast<scalar_t>(i) / static_cast<scalar_t>(n - 1);
  }
  cmd.modeSwitchingTimes = {0.0, 0.6, 0.7, 1.3, 1.4};
  cmd.modeSequence = {ModeNumber::LF, ModeNumber::STANCE, ModeNumber::RF, ModeNumber::STANCE};

  // Footsteps: staggered periodic forward walk, step length 0.25 m.
  cmd.footsteps = {
      {0, 0.0, 0.0, 0.10},    // initial left
      {1, 0.0, -0.25, -0.10}, // initial right
      {1, 0.6, 0.25, -0.10},  // right touchdown
      {0, 1.3, 0.5, 0.10},    // left touchdown
  };
  return cmd;
}

}  // namespace

TEST(StylizedReference, validWalkCommandPasses) {
  EXPECT_EQ(makeWalkCommand().validate(), "");
}

TEST(StylizedReference, invalidModeScheduleRejected) {
  auto cmd = makeWalkCommand();
  cmd.modeSwitchingTimes = {0.0, 0.6, 0.7};  // wrong length
  EXPECT_NE(cmd.validate(), "");

  cmd = makeWalkCommand();
  cmd.modeSwitchingTimes.back() = 1.0;  // does not span cycle
  EXPECT_NE(cmd.validate(), "");

  cmd = makeWalkCommand();
  cmd.modeSequence = {ModeNumber::LF, ModeNumber::STANCE, ModeNumber::RF, 7};
  EXPECT_NE(cmd.validate(), "");
}

TEST(StylizedReference, curveLengthMismatchRejected) {
  auto cmd = makeWalkCommand();
  StylizedCurveChannel c;
  c.name = "joint/waist_yaw_joint";
  c.values.assign(10, 0.0);  // wrong length
  cmd.jointCurves.push_back(c);
  EXPECT_NE(cmd.validate(), "");
}

TEST(StylizedReference, missingInitialFootstepRejected) {
  auto cmd = makeWalkCommand();
  cmd.footsteps.erase(cmd.footsteps.begin());  // drop initial left
  EXPECT_NE(cmd.validate(), "");
}

TEST(StylizedReference, periodicSamplingWrapsAndInterpolates) {
  auto cmd = makeWalkCommand();
  std::vector<scalar_t> values(cmd.timeGrid.size());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = std::sin(2.0 * M_PI * cmd.timeGrid[i] / cmd.cycleDuration);
  }

  // Hits grid values exactly.
  EXPECT_NEAR(cmd.samplePeriodic(values, cmd.timeGrid[10]), values[10], 1e-12);
  // Linear interpolation between grid points.
  const scalar_t mid = 0.5 * (cmd.timeGrid[10] + cmd.timeGrid[11]);
  EXPECT_NEAR(cmd.samplePeriodic(values, mid), 0.5 * (values[10] + values[11]), 1e-12);
  // Wraparound: tau = T + x samples at x; negative wraps too.
  EXPECT_NEAR(cmd.samplePeriodic(values, cmd.cycleDuration + 0.2), cmd.samplePeriodic(values, 0.2), 1e-12);
  EXPECT_NEAR(cmd.samplePeriodic(values, -0.2), cmd.samplePeriodic(values, cmd.cycleDuration - 0.2), 1e-12);
}

TEST(StylizedReference, derivativeOfSineIsCosine) {
  auto cmd = makeWalkCommand();
  const scalar_t omega = 2.0 * M_PI / cmd.cycleDuration;
  std::vector<scalar_t> values(cmd.timeGrid.size());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = std::sin(omega * cmd.timeGrid[i]);
  }
  for (scalar_t tau : {0.1, 0.35, 0.7, 1.2}) {
    EXPECT_NEAR(cmd.sampleDerivativePeriodic(values, tau), omega * std::cos(omega * tau), 0.05 * omega);
  }
}

TEST(StylizedReference, cycleDisplacementAveragesFeet) {
  auto cmd = makeWalkCommand();
  const vector2_t d = cmd.cycleDisplacement();
  EXPECT_NEAR(d.x(), 0.5, 1e-12);
  EXPECT_NEAR(d.y(), 0.0, 1e-12);
}

TEST(StylizedReference, sameTimingComparison) {
  auto a = makeWalkCommand();
  auto b = makeWalkCommand();
  b.commandId = 99;
  StylizedCurveChannel c;
  c.name = "joint/waist_yaw_joint";
  c.values.assign(b.timeGrid.size(), 0.3);
  b.jointCurves.push_back(c);
  EXPECT_TRUE(a.sameTiming(b));  // curves differ, timing equal

  b.modeSwitchingTimes[1] = 0.5;
  EXPECT_FALSE(a.sameTiming(b));

  b = makeWalkCommand();
  b.cycleDuration = 2.8;
  EXPECT_FALSE(a.sameTiming(b));
}

TEST(StylizedReference, toModeSequenceTemplate) {
  auto cmd = makeWalkCommand();
  const auto tpl = cmd.toModeSequenceTemplate();
  EXPECT_EQ(tpl.switchingTimes.size(), 5u);
  EXPECT_EQ(tpl.modeSequence.size(), 4u);
  EXPECT_EQ(tpl.modeSequence[0], static_cast<size_t>(ModeNumber::LF));
  EXPECT_DOUBLE_EQ(tpl.switchingTimes.back(), 1.4);
}
