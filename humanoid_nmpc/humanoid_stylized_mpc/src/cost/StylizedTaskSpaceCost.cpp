/******************************************************************************
Copyright (c) 2026. All rights reserved. BSD-3-Clause license, see the
wb_humanoid_mpc LICENSE file.
******************************************************************************/

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_stylized_mpc/cost/StylizedTaskSpaceCost.h"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace ocs2::humanoid {

StylizedTaskSpaceCost::StylizedTaskSpaceCost(std::shared_ptr<const StylizedReferenceBuffer> buffer,
                                             const PinocchioInterface& pinocchioInterface,
                                             const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel,
                                             std::string frameName,
                                             const ModelSettings& modelSettings)
    : StateInputCostGaussNewtonAd(),
      bufferPtr_(std::move(buffer)),
      frameName_(std::move(frameName)),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      mpcRobotModelPtr_(mpcRobotModel.clone()) {
  if (!pinocchioInterface.getModel().existFrame(frameName_)) {
    throw std::runtime_error("[StylizedTaskSpaceCost] frame '" + frameName_ + "' does not exist in the pinocchio model");
  }
  frameID_ = pinocchioInterface.getModel().getFrameId(frameName_);
  initialize(mpcRobotModelPtr_->getStateDim(), mpcRobotModelPtr_->getInputDim(), 6, "stylized_pos_cost_" + frameName_,
             modelSettings.modelFolderCppAd, modelSettings.recompileLibrariesCppAd);
}

StylizedTaskSpaceCost::StylizedTaskSpaceCost(const StylizedTaskSpaceCost& other)
    : StateInputCostGaussNewtonAd(other),
      bufferPtr_(other.bufferPtr_),
      frameName_(other.frameName_),
      frameID_(other.frameID_),
      pinocchioInterfaceCppAd_(other.pinocchioInterfaceCppAd_),
      mpcRobotModelPtr_(other.mpcRobotModelPtr_->clone()) {}

ad_vector_t StylizedTaskSpaceCost::costVectorFunction(ad_scalar_t /*time*/,
                                                      const ad_vector_t& state,
                                                      const ad_vector_t& /*input*/,
                                                      const ad_vector_t& parameters) {
  const auto& model = pinocchioInterfaceCppAd_.getModel();
  auto& data = pinocchioInterfaceCppAd_.getData();

  const ad_vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  pinocchio::forwardKinematics(model, data, q);
  const auto& framePlacement = pinocchio::updateFramePlacement(model, data, frameID_);

  const ad_vector_t position = framePlacement.translation();
  const ad_vector_t reference = parameters.head(3);
  const ad_vector_t sqrtWeights = parameters.segment(3, 3);

  return (position - reference).cwiseProduct(sqrtWeights);
}

vector_t StylizedTaskSpaceCost::getParameters(scalar_t time,
                                              const TargetTrajectories& /*targetTrajectories*/,
                                              const PreComputation& /*preComputation*/) const {
  return bufferPtr_->cartesianParameters(frameName_, time);
}

}  // namespace ocs2::humanoid
