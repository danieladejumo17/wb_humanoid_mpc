/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#pragma once

#include <string>
#include <vector>

#include "robot_core/Types.h"
#include "robot_model/RobotHWInterfaceBase.h"
#include "robot_model/RobotState.h"

#include "tcp_bridge.pb.h"

namespace robot::tcp_interface {

struct TcpSimConfig {
  std::string host = "localhost";
  int port = 9000;
  std::string scenePath;
  double dt{0.0005};
  std::shared_ptr<model::RobotState> initStatePtr;
  // TODO: Add render and verbose configs
};

class TcpHWInterface : public robot::model::RobotHWInterfaceBase {
 public:
  TcpHWInterface(const TcpSimConfig& config, const std::string& urdfPath);
  ~TcpHWInterface();

  TcpHWInterface(const TcpHWInterface&) = delete;
  TcpHWInterface& operator=(const TcpHWInterface&) = delete;

  void initSim();   // After initialization, a valid state must be available for the controller to use.
  void startSim();  // Start the simulation loop.
  void reset();

  // Requests the latest state from the Python sim over TCP, then calls
  // updateInterfaceStateFromRobot() so the controller can read it.
  void requestState();

  // Calls applyJointAction() to capture the controller's output, then
  // sends it to the Python sim over TCP.
  void sendAction();

 private:
  void connectToServer();
  void disconnect();

  void sendMessage(const robot_bridge::TcpMessage& msg);
  robot_bridge::TcpMessage receiveMessage();
  void receiveAck();

  void sendRawBytes(const void* data, size_t len);
  void recvRawBytes(void* data, size_t len);

  void packSimConfig(robot_bridge::SimConfig& configMsg);
  void packRobotState(robot_bridge::RobotStateMsg& stateMsg, const model::RobotState& state);
  void packJointAction(robot_bridge::JointActionMsg& actionMsg);
  void unpackRobotState(const robot_bridge::RobotStateMsg& stateMsg);

  TcpSimConfig config_;
  int socket_fd_ = -1;

  model::RobotState robotStateInternal_;
  model::RobotJointAction robotJointActionInternal_;

  std::vector<std::string> jointNames_;
  std::vector<joint_index_t> jointIndices_;
};

}  // namespace robot::tcp_interface
