/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_stylized_mpc/cost/FootstepPlacementCost.h"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace ocs2::humanoid {

FootstepPlacementCost::FootstepPlacementCost(std::shared_ptr<const StylizedReferenceBuffer> buffer,
                                             const PinocchioInterface& pinocchioInterface,
                                             const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel,
                                             size_t contactIndex,
                                             const ModelSettings& modelSettings)
    : StateInputCostGaussNewtonAd(),
      bufferPtr_(std::move(buffer)),
      contactIndex_(contactIndex),
      frameID_(pinocchioInterface.getModel().getFrameId(modelSettings.contactNames6DoF[contactIndex])),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      mpcRobotModelPtr_(mpcRobotModel.clone()) {
  initialize(mpcRobotModelPtr_->getStateDim(), mpcRobotModelPtr_->getInputDim(), 3,
             "footstep_placement_cost_" + modelSettings.contactNames6DoF[contactIndex], modelSettings.modelFolderCppAd,
             modelSettings.recompileLibrariesCppAd);
}

FootstepPlacementCost::FootstepPlacementCost(const FootstepPlacementCost& other)
    : StateInputCostGaussNewtonAd(other),
      bufferPtr_(other.bufferPtr_),
      contactIndex_(other.contactIndex_),
      frameID_(other.frameID_),
      pinocchioInterfaceCppAd_(other.pinocchioInterfaceCppAd_),
      mpcRobotModelPtr_(other.mpcRobotModelPtr_->clone()) {}

ad_vector_t FootstepPlacementCost::costVectorFunction(ad_scalar_t /*time*/,
                                                      const ad_vector_t& state,
                                                      const ad_vector_t& /*input*/,
                                                      const ad_vector_t& parameters) {
  const auto& model = pinocchioInterfaceCppAd_.getModel();
  auto& data = pinocchioInterfaceCppAd_.getData();

  const ad_vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  pinocchio::forwardKinematics(model, data, q);
  const auto& framePlacement = pinocchio::updateFramePlacement(model, data, frameID_);

  const ad_vector_t xy = framePlacement.translation().head(2);
  const ad_vector_t reference = parameters.head(2);
  const ad_scalar_t sqrtWeight = parameters(2);

  return (xy - reference) * sqrtWeight;
}

vector_t FootstepPlacementCost::getParameters(scalar_t time,
                                              const TargetTrajectories& /*targetTrajectories*/,
                                              const PreComputation& /*preComputation*/) const {
  return bufferPtr_->footstepParameters(static_cast<int>(contactIndex_), time);
}

}  // namespace ocs2::humanoid
