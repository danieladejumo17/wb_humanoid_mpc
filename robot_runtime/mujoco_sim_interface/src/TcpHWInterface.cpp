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

#include "mujoco_sim_interface/TcpHWInterface.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace robot::tcp_interface {

TcpHWInterface::TcpHWInterface(const TcpSimConfig& config, const std::string& urdfPath)
    : RobotHWInterfaceBase(urdfPath),
      config_(config),
      robotStateInternal_(model::RobotState(this->getRobotDescription())),
      robotJointActionInternal_(model::RobotJointAction(this->getRobotDescription())) {
  jointNames_ = getRobotDescription().getJointNames();
  jointIndices_ = getRobotDescription().getJointIndices();
}

TcpHWInterface::~TcpHWInterface() {
  disconnect();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void TcpHWInterface::connectToServer() {
  struct addrinfo hints {};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  struct addrinfo* result = nullptr;
  std::string portStr = std::to_string(config_.port);
  int rc = getaddrinfo(config_.host.c_str(), portStr.c_str(), &hints, &result);
  if (rc != 0 || result == nullptr) {
    throw std::runtime_error("TcpHWInterface: Failed to resolve host: " + config_.host + " - " + gai_strerror(rc));
  }

  socket_fd_ = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
  if (socket_fd_ < 0) {
    freeaddrinfo(result);
    throw std::runtime_error("TcpHWInterface: Failed to create socket: " + std::string(std::strerror(errno)));
  }

  int flag = 1;
  setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

  if (connect(socket_fd_, result->ai_addr, result->ai_addrlen) < 0) {
    close(socket_fd_);
    socket_fd_ = -1;
    freeaddrinfo(result);
    throw std::runtime_error("TcpHWInterface: Failed to connect to " + config_.host + ":" + std::to_string(config_.port) + " - " +
                             std::strerror(errno));
  }

  freeaddrinfo(result);
  std::cerr << "TcpHWInterface: Connected to " << config_.host << ":" << config_.port << std::endl;
}

void TcpHWInterface::disconnect() {
  if (socket_fd_ >= 0) {
    close(socket_fd_);
    socket_fd_ = -1;
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void TcpHWInterface::sendRawBytes(const void* data, size_t len) {
  const auto* ptr = static_cast<const uint8_t*>(data);
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = send(socket_fd_, ptr + sent, len - sent, MSG_NOSIGNAL);
    if (n <= 0) {
      throw std::runtime_error("TcpHWInterface: send failed: " + std::string(std::strerror(errno)));
    }
    sent += static_cast<size_t>(n);
  }
}

void TcpHWInterface::recvRawBytes(void* data, size_t len) {
  auto* ptr = static_cast<uint8_t*>(data);
  size_t received = 0;
  while (received < len) {
    ssize_t n = recv(socket_fd_, ptr + received, len - received, MSG_WAITALL);
    if (n <= 0) {
      throw std::runtime_error("TcpHWInterface: recv failed: " + std::string(std::strerror(errno)));
    }
    received += static_cast<size_t>(n);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void TcpHWInterface::sendMessage(const robot_bridge::TcpMessage& msg) {
  std::string serialized = msg.SerializeAsString();
  uint32_t length = static_cast<uint32_t>(serialized.size());
  sendRawBytes(&length, sizeof(length));
  sendRawBytes(serialized.data(), serialized.size());
}

robot_bridge::TcpMessage TcpHWInterface::receiveMessage() {
  uint32_t length = 0;
  recvRawBytes(&length, sizeof(length));

  std::string buffer(length, '\0');
  recvRawBytes(buffer.data(), length);

  robot_bridge::TcpMessage msg;
  if (!msg.ParseFromString(buffer)) {
    throw std::runtime_error("TcpHWInterface: Failed to parse incoming TcpMessage");
  }
  return msg;
}

void TcpHWInterface::receiveAck() {
  robot_bridge::TcpMessage response = receiveMessage();
  if (response.type() != robot_bridge::ACK) {
    throw std::runtime_error("TcpHWInterface: Expected ACK, got message type " + std::to_string(response.type()));
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void TcpHWInterface::packSimConfig(robot_bridge::SimConfig& configMsg) {
  configMsg.set_scene_path(config_.scenePath);
  configMsg.set_dt(config_.dt);

  for (const auto& name : jointNames_) {
    configMsg.add_joint_names(name);
  }

  if (config_.initStatePtr) {
    packRobotState(*configMsg.mutable_init_state(), *config_.initStatePtr);
  }
}

void TcpHWInterface::packRobotState(robot_bridge::RobotStateMsg& stateMsg, const model::RobotState& state) {
  stateMsg.set_time(state.getTime());

  vector3_t pos = state.getRootPositionInWorldFrame();
  stateMsg.add_root_position(pos[0]);
  stateMsg.add_root_position(pos[1]);
  stateMsg.add_root_position(pos[2]);

  quaternion_t quat = state.getRootRotationLocalToWorldFrame();
  stateMsg.add_root_orientation(quat.w());
  stateMsg.add_root_orientation(quat.x());
  stateMsg.add_root_orientation(quat.y());
  stateMsg.add_root_orientation(quat.z());

  vector3_t linVel = state.getRootLinearVelocityInLocalFrame();
  stateMsg.add_root_linear_vel(linVel[0]);
  stateMsg.add_root_linear_vel(linVel[1]);
  stateMsg.add_root_linear_vel(linVel[2]);

  vector3_t angVel = state.getRootAngularVelocityInLocalFrame();
  stateMsg.add_root_angular_vel(angVel[0]);
  stateMsg.add_root_angular_vel(angVel[1]);
  stateMsg.add_root_angular_vel(angVel[2]);

  for (auto idx : jointIndices_) {
    stateMsg.add_joint_positions(state.getJointPosition(idx));
    stateMsg.add_joint_velocities(state.getJointVelocity(idx));
  }

  std::vector<bool> contacts = state.getContactFlags();
  for (bool c : contacts) {
    stateMsg.add_contact_flags(c);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void TcpHWInterface::packJointAction(robot_bridge::JointActionMsg& actionMsg) {
  for (auto idx : jointIndices_) {
    const auto& action = robotJointActionInternal_.at(idx).value();
    actionMsg.add_q_des(action.q_des);
    actionMsg.add_qd_des(action.qd_des);
    actionMsg.add_kp(action.kp);
    actionMsg.add_kd(action.kd);
    actionMsg.add_feed_forward_effort(action.feed_forward_effort);
  }
}

void TcpHWInterface::unpackRobotState(const robot_bridge::RobotStateMsg& stateMsg) {
  robotStateInternal_.setTime(stateMsg.time());

  if (stateMsg.root_position_size() == 3) {
    robotStateInternal_.setRootPositionInWorldFrame(
        vector3_t(stateMsg.root_position(0), stateMsg.root_position(1), stateMsg.root_position(2)));
  }

  if (stateMsg.root_orientation_size() == 4) {
    robotStateInternal_.setRootRotationLocalToWorldFrame(quaternion_t(stateMsg.root_orientation(0), stateMsg.root_orientation(1),
                                                                      stateMsg.root_orientation(2), stateMsg.root_orientation(3)));
  }

  if (stateMsg.root_linear_vel_size() == 3) {
    robotStateInternal_.setRootLinearVelocityInLocalFrame(
        vector3_t(stateMsg.root_linear_vel(0), stateMsg.root_linear_vel(1), stateMsg.root_linear_vel(2)));
  }

  if (stateMsg.root_angular_vel_size() == 3) {
    robotStateInternal_.setRootAngularVelocityInLocalFrame(
        vector3_t(stateMsg.root_angular_vel(0), stateMsg.root_angular_vel(1), stateMsg.root_angular_vel(2)));
  }

  size_t nJoints = std::min(static_cast<size_t>(stateMsg.joint_positions_size()), jointIndices_.size());
  for (size_t i = 0; i < nJoints; ++i) {
    robotStateInternal_.setJointPosition(jointIndices_[i], stateMsg.joint_positions(static_cast<int>(i)));
    robotStateInternal_.setJointVelocity(jointIndices_[i], stateMsg.joint_velocities(static_cast<int>(i)));
  }

  for (int i = 0; i < stateMsg.contact_flags_size(); ++i) {
    robotStateInternal_.setContactFlag(static_cast<size_t>(i), stateMsg.contact_flags(i));
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void TcpHWInterface::initSim() {
  connectToServer();

  robot_bridge::TcpMessage configMessage;
  configMessage.set_type(robot_bridge::INIT_CONFIG);
  packSimConfig(*configMessage.mutable_sim_config());

  sendMessage(configMessage);
  receiveAck();

  robot_bridge::TcpMessage initMessage;
  initMessage.set_type(robot_bridge::INIT_SIM);  // Sim should get a valid state (one physics step) ready after init sim
  sendMessage(initMessage);
  receiveAck();

  std::cerr << "TcpHWInterface: Simulation initialized." << std::endl;
}

void TcpHWInterface::startSim() {
  robot_bridge::TcpMessage startMessage;
  startMessage.set_type(robot_bridge::START_SIM);
  sendMessage(startMessage);
  receiveAck();

  std::cerr << "TcpHWInterface: Simulation started." << std::endl;
}

void TcpHWInterface::reset() {
  robot_bridge::TcpMessage resetMessage;
  resetMessage.set_type(robot_bridge::RESET);
  sendMessage(resetMessage);
  receiveAck();

  std::cerr << "TcpHWInterface: Simulation reset." << std::endl;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void TcpHWInterface::requestState() {
  robot_bridge::TcpMessage request;
  request.set_type(robot_bridge::GET_STATE);
  sendMessage(request);

  robot_bridge::TcpMessage response = receiveMessage();
  if (response.type() != robot_bridge::STATE_RESPONSE) {
    throw std::runtime_error("TcpHWInterface: Expected STATE_RESPONSE, got message type " + std::to_string(response.type()));
  }

  unpackRobotState(response.robot_state());
  threadSafeRobotState_.set(robotStateInternal_);
  updateInterfaceStateFromRobot();
}

void TcpHWInterface::sendAction() {
  applyJointAction();
  threadSafeRobotJointAction_.copy_value(robotJointActionInternal_);

  robot_bridge::TcpMessage actionMessage;
  actionMessage.set_type(robot_bridge::SEND_ACTION);
  packJointAction(*actionMessage.mutable_joint_action());

  sendMessage(actionMessage);
  receiveAck();
}

}  // namespace robot::tcp_interface
