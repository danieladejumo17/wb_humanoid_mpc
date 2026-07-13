/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#include <gtest/gtest.h>

#include "humanoid_stylized_mpc/StylizedReferenceBuffer.h"

#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>

using namespace ocs2;
using namespace ocs2::humanoid;

namespace {

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
  cmd.footsteps = {
      {0, 0.0, 0.0, 0.10},
      {1, 0.0, -0.25, -0.10},
      {1, 0.6, 0.25, -0.10},  // right touchdown after swing [0.0, 0.6]
      {0, 1.3, 0.5, 0.10},    // left touchdown after swing [0.7, 1.3]
  };
  cmd.footstepWeight = 100.0;
  return cmd;
}

StylizedReferenceSnapshot makeSnapshot(scalar_t anchorTime = 10.0, vector3_t anchorPose = vector3_t(0.0, 0.0, 0.0)) {
  StylizedReferenceSnapshot s;
  s.active = true;
  s.anchorTime = anchorTime;
  s.anchorPose = anchorPose;
  s.command = makeWalkCommand();
  s.footstepWindows = computeFootstepWindows(s.command);
  return s;
}

}  // namespace

TEST(FootstepWindows, computedFromSwingPhases) {
  const auto windows = computeFootstepWindows(makeWalkCommand());
  ASSERT_EQ(windows.size(), 2u);

  // Right touchdown at 0.6, swing [0, 0.6] (RF swings during LF mode) => window
  // min(0.8 * 0.6, 0.5) = 0.48 before touchdown.
  const auto& right = windows[0].foot == 1 ? windows[0] : windows[1];
  EXPECT_EQ(right.foot, 1);
  EXPECT_NEAR(right.touchdownPhase, 0.6, 1e-9);
  EXPECT_NEAR(right.windowStart, 0.6 - 0.48, 1e-9);

  // Left touchdown at 1.3, swing [0.7, 1.3].
  const auto& left = windows[0].foot == 0 ? windows[0] : windows[1];
  EXPECT_EQ(left.foot, 0);
  EXPECT_NEAR(left.touchdownPhase, 1.3, 1e-9);
  EXPECT_NEAR(left.windowStart, 1.3 - 0.48, 1e-9);
}

TEST(StylizedReferenceBuffer, footstepWindowActivation) {
  StylizedReferenceBuffer buffer;
  buffer.update(makeSnapshot(10.0));

  // Right foot: window [10.12, 10.6].
  EXPECT_FALSE(buffer.footstepWindowActive(1, 10.05));
  EXPECT_TRUE(buffer.footstepWindowActive(1, 10.5));
  EXPECT_FALSE(buffer.footstepWindowActive(1, 10.7));
  // Left foot: window [10.82, 11.3]; also repeats next cycle [12.22, 12.7].
  EXPECT_TRUE(buffer.footstepWindowActive(0, 11.2));
  EXPECT_TRUE(buffer.footstepWindowActive(0, 11.2 + 1.4));
  // Before the anchor nothing is active.
  EXPECT_FALSE(buffer.footstepWindowActive(0, 9.5));
}

TEST(StylizedReferenceBuffer, footstepTargetAdvancesWithCycles) {
  StylizedReferenceBuffer buffer;
  buffer.update(makeSnapshot(10.0));

  // First cycle: right footstep at gait-frame (0.25, -0.10).
  const vector_t p0 = buffer.footstepParameters(1, 10.5);
  EXPECT_NEAR(p0(0), 0.25, 1e-9);
  EXPECT_NEAR(p0(1), -0.10, 1e-9);
  EXPECT_NEAR(p0(2), std::sqrt(100.0), 1e-9);

  // Second cycle: displaced by 0.5 m.
  const vector_t p1 = buffer.footstepParameters(1, 10.5 + 1.4);
  EXPECT_NEAR(p1(0), 0.75, 1e-9);
  EXPECT_NEAR(p1(1), -0.10, 1e-9);
}

TEST(StylizedReferenceBuffer, cartesianParametersTransformToWorld) {
  auto snapshot = makeSnapshot(10.0, vector3_t(1.0, 2.0, M_PI_2));

  StylizedCartesianCurve curve;
  curve.frameName = "left_rubber_hand";
  curve.activeXyz = {true, false, true};
  curve.weightsXyz = {4.0, 0.0, 9.0};
  const size_t n = snapshot.command.timeGrid.size();
  curve.x.assign(n, 0.3);
  curve.y.assign(n, 0.0);
  curve.z.assign(n, 1.1);
  snapshot.command.cartesianCurves.push_back(curve);

  StylizedReferenceBuffer buffer;
  buffer.update(std::move(snapshot));

  EXPECT_TRUE(buffer.frameActive("left_rubber_hand"));
  EXPECT_FALSE(buffer.frameActive("right_rubber_hand"));

  const vector_t p = buffer.cartesianParameters("left_rubber_hand", 10.1);
  // Gait frame rotated 90 deg: (0.3, 0) -> world (+1.0, +2.3).
  EXPECT_NEAR(p(0), 1.0, 1e-9);
  EXPECT_NEAR(p(1), 2.3, 1e-9);
  EXPECT_NEAR(p(2), 1.1, 1e-9);  // z absolute
  EXPECT_NEAR(p(3), 2.0, 1e-9);  // sqrt(4)
  EXPECT_NEAR(p(4), 0.0, 1e-9);  // inactive axis
  EXPECT_NEAR(p(5), 3.0, 1e-9);  // sqrt(9)
}

TEST(StylizedReferenceBuffer, inactiveSnapshotDisablesEverything) {
  StylizedReferenceBuffer buffer;
  auto snapshot = makeSnapshot();
  snapshot.active = false;
  buffer.update(std::move(snapshot));

  EXPECT_FALSE(buffer.frameActive("left_rubber_hand"));
  EXPECT_FALSE(buffer.footstepWindowActive(0, 11.2));
  EXPECT_TRUE(buffer.cartesianParameters("left_rubber_hand", 11.0).isZero());
}
