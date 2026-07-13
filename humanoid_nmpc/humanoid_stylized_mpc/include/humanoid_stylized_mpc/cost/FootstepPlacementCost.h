/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#pragma once

#include <memory>
#include <string>

#include <ocs2_core/cost/StateInputGaussNewtonCostAd.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include <humanoid_common_mpc/common/ModelSettings.h>
#include <humanoid_common_mpc/common/MpcRobotModelBase.h>

#include "humanoid_stylized_mpc/StylizedReferenceBuffer.h"

namespace ocs2::humanoid {

/**
 * Pulls the swing foot's contact frame toward the planned footstep XY inside
 * a window before touchdown. Footsteps are otherwise free variables of the
 * whole-body OCP; this cost is the styling hook that pins them to the plan.
 */
class FootstepPlacementCost final : public StateInputCostGaussNewtonAd {
 public:
  FootstepPlacementCost(std::shared_ptr<const StylizedReferenceBuffer> buffer,
                        const PinocchioInterface& pinocchioInterface,
                        const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel,
                        size_t contactIndex,  // 0 = left, 1 = right (contactNames6DoF order)
                        const ModelSettings& modelSettings);

  ~FootstepPlacementCost() override = default;
  FootstepPlacementCost* clone() const override { return new FootstepPlacementCost(*this); }

  vector_t getParameters(scalar_t time, const TargetTrajectories& targetTrajectories, const PreComputation& preComputation) const override;

  bool isActive(scalar_t time) const override {
    return bufferPtr_->footstepWindowActive(static_cast<int>(contactIndex_), time);
  }

 private:
  FootstepPlacementCost(const FootstepPlacementCost& other);

  ad_vector_t costVectorFunction(ad_scalar_t time,
                                 const ad_vector_t& state,
                                 const ad_vector_t& input,
                                 const ad_vector_t& parameters) override;

  std::shared_ptr<const StylizedReferenceBuffer> bufferPtr_;
  size_t contactIndex_;
  std::size_t frameID_;  // pinocchio::FrameIndex
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  std::unique_ptr<MpcRobotModelBase<ad_scalar_t>> mpcRobotModelPtr_;
};

}  // namespace ocs2::humanoid
