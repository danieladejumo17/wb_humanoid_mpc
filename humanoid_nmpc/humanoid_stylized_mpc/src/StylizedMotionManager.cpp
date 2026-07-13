/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#include "humanoid_stylized_mpc/StylizedMotionManager.h"

#include <cmath>

#include <ocs2_core/misc/LoadData.h>

#include "humanoid_stylized_mpc/StylizedReferenceBuilder.h"

namespace ocs2::humanoid {

namespace sr = stylized_reference;

StylizedMotionManager::StylizedMotionManager(std::shared_ptr<SwitchedModelReferenceManager> referenceManagerPtr,
                                             const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                             const std::string& referenceFile)
    : switchedModelReferenceManagerPtr_(std::move(referenceManagerPtr)),
      gaitSchedulePtr_(switchedModelReferenceManagerPtr_->getGaitSchedule()),
      mpcRobotModelPtr_(&mpcRobotModel) {
  defaultJointState_.resize(mpcRobotModelPtr_->getJointDim());
  loadData::loadEigenMatrix(referenceFile, "defaultJointState", defaultJointState_);
  loadData::loadCppDataType(referenceFile, "defaultBaseHeight", defaultBaseHeight_);
}

void StylizedMotionManager::setCommand(const StylizedGaitCommandData& command) {
  std::lock_guard<std::mutex> lock(commandMutex_);
  stagedCommand_ = command;
}

void StylizedMotionManager::publishStatus(bool accepted, const std::string& message, size_t commandId) {
  if (statusCallback_) {
    StylizedGaitStatusData status;
    status.lastCommandId = commandId;
    status.accepted = accepted;
    status.message = message;
    status.active = active_;
    status.anchorTime = anchorTime_;
    status.cycleDuration = active_ ? command_.cycleDuration : 0.0;
    statusCallback_(status);
  }
}

vector3_t StylizedMotionManager::anchorAfterCycles(size_t numCycles) const {
  return sr::anchorAfterCycles(command_, anchorPose_, numCycles);
}

void StylizedMotionManager::advanceAnchor(scalar_t initTime) {
  while (active_ && initTime - anchorTime_ >= command_.cycleDuration) {
    anchorPose_ = anchorAfterCycles(1);
    anchorTime_ += command_.cycleDuration;
    publishStatus(true, "cycle", command_.commandId);
  }
}

void StylizedMotionManager::adoptPendingCommand(scalar_t initTime, scalar_t finalTime) {
  std::optional<StylizedGaitCommandData> staged;
  {
    std::lock_guard<std::mutex> lock(commandMutex_);
    staged.swap(stagedCommand_);
  }
  if (!staged) {
    return;
  }

  const std::string error = staged->validate();
  if (!error.empty()) {
    publishStatus(false, error, staged->commandId);
    return;
  }

  if (!staged->enable) {
    // Return to standing: replace the schedule with all-stance.
    active_ = false;
    anchorPoseInitialized_ = false;
    pendingTimingCommand_.reset();
    ModeSequenceTemplate stance({0.0, 1.0}, {ModeNumber::STANCE});
    gaitSchedulePtr_->insertModeSequenceTemplate(stance, initTime, finalTime);
    command_ = *staged;
    publishStatus(true, "disabled", staged->commandId);
    return;
  }

  if (!active_) {
    // First activation: give the solver preview time before the cycle starts.
    const scalar_t spliceTime = initTime + kStartDelayFraction * (finalTime - initTime);
    gaitSchedulePtr_->insertModeSequenceTemplate(staged->toModeSequenceTemplate(), spliceTime,
                                                 finalTime + staged->cycleDuration);
    command_ = *staged;
    footstepWindows_ = computeFootstepWindows(command_);
    anchorTime_ = spliceTime;
    active_ = true;
    anchorPoseInitialized_ = false;  // anchored to the base pose once the cycle starts
    publishStatus(true, "activated", staged->commandId);
    return;
  }

  if (staged->sameTiming(command_)) {
    // Hot-swap curves/footsteps, keep the phase anchor.
    command_ = *staged;
    footstepWindows_ = computeFootstepWindows(command_);
    pendingTimingCommand_.reset();
    publishStatus(true, "updated", staged->commandId);
  } else {
    // Timing changed: defer to the next cycle boundary to avoid cutting a
    // swing mid-flight.
    const scalar_t elapsed = std::max(0.0, initTime - anchorTime_);
    const auto cyclesAhead = static_cast<size_t>(std::floor(elapsed / command_.cycleDuration)) + 1;
    pendingTimingCommand_ = *staged;
    pendingActivationTime_ = anchorTime_ + static_cast<scalar_t>(cyclesAhead) * command_.cycleDuration;
    gaitSchedulePtr_->insertModeSequenceTemplate(staged->toModeSequenceTemplate(), pendingActivationTime_,
                                                 pendingActivationTime_ + staged->cycleDuration);
    publishStatus(true, "retiming scheduled", staged->commandId);
  }
}

void StylizedMotionManager::preSolverRun(scalar_t initTime,
                                         scalar_t finalTime,
                                         const vector_t& initState,
                                         const ReferenceManagerInterface& /*referenceManager*/) {
  adoptPendingCommand(initTime, finalTime);

  // Activate a deferred timing change once its cycle boundary passes.
  if (pendingTimingCommand_ && initTime >= pendingActivationTime_ - 1e-9) {
    advanceAnchor(std::min(initTime, pendingActivationTime_));
    anchorPose_ = anchorAfterCycles(0);  // anchor pose at the boundary is already advanced
    command_ = *pendingTimingCommand_;
    footstepWindows_ = computeFootstepWindows(command_);
    anchorTime_ = pendingActivationTime_;
    pendingTimingCommand_.reset();
    publishStatus(true, "retimed", command_.commandId);
  }

  if (active_ && !anchorPoseInitialized_ && initTime >= anchorTime_ - 1e-9) {
    // The first cycle starts now: anchor the gait frame at the current base.
    const vector6_t basePose = mpcRobotModelPtr_->getBasePose(initState);
    anchorPose_ = vector3_t(basePose(0), basePose(1), basePose(3));
    anchorPoseInitialized_ = true;
  }

  if (active_ && anchorPoseInitialized_) {
    advanceAnchor(initTime);
  }

  TargetTrajectories target;
  if (active_) {
    vector3_t anchorPose = anchorPose_;
    if (!anchorPoseInitialized_) {
      // Cycle has not started yet: pre-roll references anchored at the
      // predicted start = current base pose.
      const vector6_t basePose = mpcRobotModelPtr_->getBasePose(initState);
      anchorPose = vector3_t(basePose(0), basePose(1), basePose(3));
    }
    target = sr::buildTargetTrajectories(*mpcRobotModelPtr_, defaultJointState_, defaultBaseHeight_, command_, anchorTime_,
                                         anchorPose, initTime, finalTime, kReferenceDt);
  } else {
    target = sr::buildStanceTargetTrajectories(*mpcRobotModelPtr_, defaultJointState_, defaultBaseHeight_, initTime,
                                               finalTime, initState);
  }
  switchedModelReferenceManagerPtr_->setTargetTrajectories(std::move(target));

  // Publish the per-solve snapshot for the cartesian/footstep costs.
  StylizedReferenceSnapshot snapshot;
  snapshot.active = active_ && anchorPoseInitialized_;
  snapshot.anchorTime = anchorTime_;
  snapshot.anchorPose = anchorPose_;
  snapshot.command = command_;
  snapshot.footstepWindows = footstepWindows_;
  referenceBufferPtr_->update(std::move(snapshot));
}

TargetTrajectories StylizedMotionManager::buildTargetTrajectories(scalar_t initTime,
                                                                  scalar_t finalTime,
                                                                  const vector_t& /*initState*/) const {
  return sr::buildTargetTrajectories(*mpcRobotModelPtr_, defaultJointState_, defaultBaseHeight_, command_, anchorTime_,
                                     anchorPose_, initTime, finalTime, kReferenceDt);
}

TargetTrajectories StylizedMotionManager::buildStanceTargetTrajectories(scalar_t initTime,
                                                                        scalar_t finalTime,
                                                                        const vector_t& initState) const {
  return sr::buildStanceTargetTrajectories(*mpcRobotModelPtr_, defaultJointState_, defaultBaseHeight_, initTime,
                                           finalTime, initState);
}

}  // namespace ocs2::humanoid
