/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#include <gtest/gtest.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <humanoid_common_mpc/common/ModelSettings.h>
#include <humanoid_wb_mpc/common/WBAccelMpcRobotModel.h>

#include "humanoid_stylized_mpc/StylizedReferenceBuilder.h"

using namespace ocs2;
using namespace ocs2::humanoid;
namespace sr = ocs2::humanoid::stylized_reference;

namespace {

class BuilderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    const std::string taskFile = ament_index_cpp::get_package_share_directory("g1_wb_mpc") + "/config/mpc/task.info";
    const std::string urdfFile = ament_index_cpp::get_package_share_directory("g1_description") + "/urdf/g1_29dof.urdf";
    modelSettings_ = new ModelSettings(taskFile, urdfFile, "stylized_test", false);
    model_ = new WBAccelMpcRobotModel<scalar_t>(*modelSettings_);
    defaultJointState_ = new vector_t(vector_t::Zero(model_->getJointDim()));
  }

  static void TearDownTestSuite() {
    delete defaultJointState_;
    delete model_;
    delete modelSettings_;
  }

  static StylizedGaitCommandData makeWalkCommand() {
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
        {1, 0.6, 0.25, -0.10},
        {0, 1.3, 0.5, 0.10},
    };
    return cmd;
  }

  static ModelSettings* modelSettings_;
  static WBAccelMpcRobotModel<scalar_t>* model_;
  static vector_t* defaultJointState_;
  static constexpr scalar_t kBaseHeight = 0.78;
};

ModelSettings* BuilderTest::modelSettings_ = nullptr;
WBAccelMpcRobotModel<scalar_t>* BuilderTest::model_ = nullptr;
vector_t* BuilderTest::defaultJointState_ = nullptr;

}  // namespace

TEST_F(BuilderTest, anchorAdvancesByDisplacementPerCycle) {
  const auto cmd = makeWalkCommand();
  const vector3_t anchor0(1.0, 2.0, 0.0);

  const vector3_t anchor3 = sr::anchorAfterCycles(cmd, anchor0, 3);
  EXPECT_NEAR(anchor3.x(), 1.0 + 3 * 0.5, 1e-12);
  EXPECT_NEAR(anchor3.y(), 2.0, 1e-12);
  EXPECT_NEAR(anchor3.z(), 0.0, 1e-12);

  // With a rotated gait frame the displacement rotates too.
  const vector3_t anchorRot(0.0, 0.0, M_PI_2);
  const vector3_t after = sr::anchorAfterCycles(cmd, anchorRot, 1);
  EXPECT_NEAR(after.x(), 0.0, 1e-12);
  EXPECT_NEAR(after.y(), 0.5, 1e-12);
}

TEST_F(BuilderTest, defaultStateProgressesLinearly) {
  const auto cmd = makeWalkCommand();
  const vector3_t anchor(0.0, 0.0, 0.0);

  const vector_t s0 = sr::buildStateAtPhase(*model_, *defaultJointState_, kBaseHeight, cmd, 0.0, anchor);
  const vector_t sHalf = sr::buildStateAtPhase(*model_, *defaultJointState_, kBaseHeight, cmd, 0.7, anchor);

  const vector6_t pose0 = model_->getBasePose(s0);
  const vector6_t poseHalf = model_->getBasePose(sHalf);
  EXPECT_NEAR(pose0(0), 0.0, 1e-12);
  EXPECT_NEAR(poseHalf(0), 0.25, 1e-12);  // half of the 0.5 m cycle displacement
  EXPECT_NEAR(pose0(2), kBaseHeight, 1e-12);
  EXPECT_NEAR(poseHalf(2), kBaseHeight, 1e-12);

  // Default forward velocity = displacement / T.
  const vector6_t vel = model_->getBaseComVelocity(sHalf);
  EXPECT_NEAR(vel(0), 0.5 / 1.4, 1e-9);
}

TEST_F(BuilderTest, jointCurveOverridesDefault) {
  auto cmd = makeWalkCommand();
  StylizedCurveChannel c;
  c.name = "joint/waist_yaw_joint";
  c.values.resize(cmd.timeGrid.size());
  for (size_t i = 0; i < c.values.size(); ++i) {
    c.values[i] = 0.3 * std::sin(2.0 * M_PI * cmd.timeGrid[i] / cmd.cycleDuration);
  }
  cmd.jointCurves.push_back(c);

  const vector3_t anchor(0.0, 0.0, 0.0);
  const scalar_t tau = 0.35;  // quarter cycle: sin = 1
  const vector_t s = sr::buildStateAtPhase(*model_, *defaultJointState_, kBaseHeight, cmd, tau, anchor);
  const vector_t joints = model_->getJointAngles(s);
  const size_t waistIndex = model_->getJointIndex("waist_yaw_joint");
  EXPECT_NEAR(joints(waistIndex), 0.3, 1e-3);

  // Other joints keep the default.
  const size_t kneeIndex = model_->getJointIndex("left_knee_joint");
  EXPECT_NEAR(joints(kneeIndex), (*defaultJointState_)(kneeIndex), 1e-12);
}

TEST_F(BuilderTest, baseCurveOverridesHeightAndVelocity) {
  auto cmd = makeWalkCommand();
  StylizedCurveChannel c;
  c.name = "base/pz";
  const scalar_t omega = 2.0 * M_PI / cmd.cycleDuration;
  c.values.resize(cmd.timeGrid.size());
  for (size_t i = 0; i < c.values.size(); ++i) {
    c.values[i] = 0.72 + 0.03 * std::sin(omega * cmd.timeGrid[i]);
  }
  cmd.baseCurves.push_back(c);

  const vector3_t anchor(0.0, 0.0, 0.0);
  const vector_t s = sr::buildStateAtPhase(*model_, *defaultJointState_, kBaseHeight, cmd, 0.35, anchor);
  EXPECT_NEAR(model_->getBasePose(s)(2), 0.75, 1e-3);
  // dz/dt at the sine peak ~ 0.
  EXPECT_NEAR(model_->getBaseComVelocity(s)(2), 0.0, 0.02);
}

TEST_F(BuilderTest, tiledTrajectoriesAreContinuousAcrossCycles) {
  const auto cmd = makeWalkCommand();
  const vector3_t anchor(0.0, 0.0, 0.0);
  const scalar_t anchorTime = 10.0;

  const auto target = sr::buildTargetTrajectories(*model_, *defaultJointState_, kBaseHeight, cmd, anchorTime, anchor,
                                                  10.0, 10.0 + 3 * 1.4, 0.02);

  ASSERT_GT(target.timeTrajectory.size(), 10u);
  EXPECT_DOUBLE_EQ(target.timeTrajectory.front(), 10.0);
  EXPECT_NEAR(target.timeTrajectory.back(), 10.0 + 3 * 1.4, 1e-9);

  // Base x must be monotonically non-decreasing and continuous (no jumps
  // bigger than displacement/T * dt * margin).
  scalar_t prevX = -1e9;
  for (size_t i = 0; i < target.stateTrajectory.size(); ++i) {
    const scalar_t x = model_->getBasePose(target.stateTrajectory[i])(0);
    EXPECT_GE(x, prevX - 1e-9);
    if (i > 0) {
      EXPECT_LT(x - prevX, 0.5 / 1.4 * 0.02 * 3.0);
    }
    prevX = x;
  }
  // Total forward progress after 3 cycles = 1.5 m.
  EXPECT_NEAR(model_->getBasePose(target.stateTrajectory.back())(0), 1.5, 1e-6);
}

TEST_F(BuilderTest, preRollHoldsCycleStartPose) {
  const auto cmd = makeWalkCommand();
  const vector3_t anchor(0.5, -0.2, 0.0);
  // Horizon starts 0.5 s before the anchor.
  const auto target = sr::buildTargetTrajectories(*model_, *defaultJointState_, kBaseHeight, cmd, 1.0, anchor, 0.5, 2.0, 0.02);
  const vector6_t poseStart = model_->getBasePose(target.stateTrajectory.front());
  EXPECT_NEAR(poseStart(0), 0.5, 1e-9);
  EXPECT_NEAR(poseStart(1), -0.2, 1e-9);
}

TEST_F(BuilderTest, stanceTargetHoldsCurrentPlanarPose) {
  vector_t initState = vector_t::Zero(model_->getStateDim());
  vector6_t pose;
  pose << 1.5, 0.3, 0.65, 0.4, 0.1, -0.05;
  model_->setBasePose(initState, pose);

  const auto target = sr::buildStanceTargetTrajectories(*model_, *defaultJointState_, kBaseHeight, 0.0, 1.0, initState);
  ASSERT_EQ(target.stateTrajectory.size(), 2u);
  const vector6_t targetPose = model_->getBasePose(target.stateTrajectory[0]);
  EXPECT_NEAR(targetPose(0), 1.5, 1e-12);
  EXPECT_NEAR(targetPose(1), 0.3, 1e-12);
  EXPECT_NEAR(targetPose(2), kBaseHeight, 1e-12);  // default height, not current
  EXPECT_NEAR(targetPose(3), 0.4, 1e-12);          // keep yaw
  EXPECT_NEAR(targetPose(4), 0.0, 1e-12);          // level pitch/roll
}
