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
 * Position tracking of one robot frame against the stylized cartesian
 * reference curves. References AND weights arrive via runtime parameters
 * (from the StylizedReferenceBuffer), so an inactive channel simply has
 * weight zero and the CppAd model never needs regeneration.
 */
class StylizedTaskSpaceCost final : public StateInputCostGaussNewtonAd {
 public:
  StylizedTaskSpaceCost(std::shared_ptr<const StylizedReferenceBuffer> buffer,
                        const PinocchioInterface& pinocchioInterface,
                        const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel,
                        std::string frameName,
                        const ModelSettings& modelSettings);

  ~StylizedTaskSpaceCost() override = default;
  StylizedTaskSpaceCost* clone() const override { return new StylizedTaskSpaceCost(*this); }

  vector_t getParameters(scalar_t time, const TargetTrajectories& targetTrajectories, const PreComputation& preComputation) const override;

  bool isActive(scalar_t time) const override { return bufferPtr_->frameActive(frameName_); }

 private:
  StylizedTaskSpaceCost(const StylizedTaskSpaceCost& other);

  ad_vector_t costVectorFunction(ad_scalar_t time,
                                 const ad_vector_t& state,
                                 const ad_vector_t& input,
                                 const ad_vector_t& parameters) override;

  std::shared_ptr<const StylizedReferenceBuffer> bufferPtr_;
  std::string frameName_;
  std::size_t frameID_;  // pinocchio::FrameIndex
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  std::unique_ptr<MpcRobotModelBase<ad_scalar_t>> mpcRobotModelPtr_;
};

}  // namespace ocs2::humanoid
