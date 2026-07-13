/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>

#include <humanoid_common_mpc/common/MpcRobotModelBase.h>
#include <humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h>

#include "humanoid_stylized_mpc/StylizedReference.h"
#include "humanoid_stylized_mpc/StylizedReferenceBuffer.h"

namespace ocs2::humanoid {

struct StylizedGaitStatusData {
  size_t lastCommandId{0};
  bool accepted{false};
  std::string message;
  bool active{false};
  scalar_t anchorTime{0.0};
  scalar_t cycleDuration{0.0};
};

/**
 * Solver-synchronized module that drives the MPC from a user-authored gait
 * cycle (StylizedGaitCommandData) instead of velocity teleop:
 *
 *  - splices the cycle's contact template into the GaitSchedule so that the
 *    template phase 0 is anchored at a known solver time (anchorTime),
 *  - every preSolverRun tiles the cycle's curves over the horizon into
 *    TargetTrajectories: the gait-frame anchor advances by the footstep
 *    cycle displacement each cycle, active channels override the defaults,
 *  - defers contact-timing changes to the next cycle boundary (curves and
 *    footsteps hot-swap immediately),
 *  - with no active command (or enable=false) holds a standing target.
 *
 * Thread safety: setCommand() may be called from a ROS callback thread; the
 * command is staged under a mutex and adopted at the start of preSolverRun.
 */
class StylizedMotionManager : public SolverSynchronizedModule {
 public:
  using StatusCallback = std::function<void(const StylizedGaitStatusData&)>;

  StylizedMotionManager(std::shared_ptr<SwitchedModelReferenceManager> referenceManagerPtr,
                        const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                        const std::string& referenceFile);

  StylizedMotionManager(const StylizedMotionManager&) = delete;

  /** Stage a new command (thread-safe); adopted on the next preSolverRun. */
  void setCommand(const StylizedGaitCommandData& command);

  void setStatusCallback(StatusCallback callback) { statusCallback_ = std::move(callback); }

  /** Shared reference buffer read by the stylized costs (cartesian/footstep). */
  std::shared_ptr<const StylizedReferenceBuffer> getReferenceBuffer() const { return referenceBufferPtr_; }

  void preSolverRun(scalar_t initTime,
                    scalar_t finalTime,
                    const vector_t& initState,
                    const ReferenceManagerInterface& referenceManager) override;

  void postSolverRun(const PrimalSolution& primalSolution) override {}

  // Exposed for testing and for the cost buffers (phase 3).
  bool isActive() const { return active_; }
  scalar_t getAnchorTime() const { return anchorTime_; }
  const vector3_t& getAnchorPose2D() const { return anchorPose_; }  // (x, y, yaw)
  const StylizedGaitCommandData& getCurrentCommand() const { return command_; }

  /**
   * Builds the tiled reference trajectories for [initTime, finalTime] given
   * the current command and anchor. Public and deterministic for unit tests.
   */
  TargetTrajectories buildTargetTrajectories(scalar_t initTime, scalar_t finalTime, const vector_t& initState) const;

  /** Standing-still target used when no stylized command is active. */
  TargetTrajectories buildStanceTargetTrajectories(scalar_t initTime, scalar_t finalTime, const vector_t& initState) const;

  /** Composes the gait-frame anchor k cycles ahead of the current one. */
  vector3_t anchorAfterCycles(size_t numCycles) const;

  static constexpr scalar_t kStartDelayFraction = 0.7;  // of the horizon, before the first cycle starts
  static constexpr scalar_t kReferenceDt = 0.02;

 private:
  void adoptPendingCommand(scalar_t initTime, scalar_t finalTime);
  void advanceAnchor(scalar_t initTime);
  void publishStatus(bool accepted, const std::string& message, size_t commandId);

  std::shared_ptr<SwitchedModelReferenceManager> switchedModelReferenceManagerPtr_;
  std::shared_ptr<GaitSchedule> gaitSchedulePtr_;
  const MpcRobotModelBase<scalar_t>* mpcRobotModelPtr_;

  vector_t defaultJointState_;
  scalar_t defaultBaseHeight_{0.8};

  // Staged command from the ROS thread.
  std::mutex commandMutex_;
  std::optional<StylizedGaitCommandData> stagedCommand_;

  // Active command state (solver thread only).
  StylizedGaitCommandData command_;
  bool active_{false};
  scalar_t anchorTime_{0.0};
  bool anchorPoseInitialized_{false};
  vector3_t anchorPose_{vector3_t::Zero()};  // (x, y, yaw) of the gait frame

  // A command whose timing change waits for the next cycle boundary.
  std::optional<StylizedGaitCommandData> pendingTimingCommand_;
  scalar_t pendingActivationTime_{0.0};

  std::vector<FootstepWindow> footstepWindows_;
  std::shared_ptr<StylizedReferenceBuffer> referenceBufferPtr_{std::make_shared<StylizedReferenceBuffer>()};

  StatusCallback statusCallback_;
};

}  // namespace ocs2::humanoid
