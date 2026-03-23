Here's a top-down roadmap for exploring this repository with the goal of adding a new simulator.

---

## Layer 1: Understand the Abstraction Boundary

Start here — this is the key interface you'll need to implement.

**`robot_runtime/robot_model/include/robot_model/RobotHWInterfaceBase.h`** — The base class that decouples the MPC from any specific simulator. It exposes:

| Method | Direction | Purpose |
|--------|-----------|---------|
| `updateInterfaceStateFromRobot()` | Sim → MPC | Copy sim state into thread-safe `RobotState` |
| `applyJointAction()` | MPC → Sim | Copy joint torques/targets from `RobotJointAction` into the sim |
| `getRobotState()` | Read | Current robot state (joint positions, velocities, base pose) |
| `getRobotJointAction()` | Read/Write | Current control action (q_des, kp, kd, ff torque) |

Also read **`RobotState`** and **`RobotJointAction`** (in `robot_runtime/robot_model/`) to understand the data structures being exchanged.

---

## Layer 2: Study the Existing MuJoCo Implementation

This is your reference for what a simulator integration looks like.

| File | What to look for |
|------|-----------------|
| `robot_runtime/mujoco_sim_interface/include/mujoco_sim_interface/MujocoSimInterface.h` | How it extends `RobotHWInterfaceBase`, what extra state it holds (`mjModel*`, `mjData*`) |
| `robot_runtime/mujoco_sim_interface/src/MujocoSimInterface.cpp` | `simulationStep()` — the core loop: apply torques → `mj_step()` → read state back. Also `simulationLoop()` for the threading model |
| `robot_runtime/mujoco_sim_interface/CMakeLists.txt` | How it links against MuJoCo, what the build target looks like |

---

## Layer 3: Understand the MPC ↔ Sim Wiring

There are **two** simulation modes in the repo. Understanding both clarifies your options:

### Mode A: In-process (MuJoCo path)

MPC and simulator run in **one process**, no ROS topics between them.

| File | Role |
|------|------|
| `humanoid_nmpc/humanoid_centroidal_mpc_ros2/exe/CentroidalMpcRobotSim.cpp` | Main loop: `updateInterfaceStateFromRobot()` → `computeJointControlAction()` → `applyJointAction()` at 500 Hz |
| `humanoid_nmpc/humanoid_wb_mpc_ros2/exe/WBMpcRobotSim.cpp` | Same pattern for whole-body MPC |

The control loop is:
```
Sim thread: mj_step() → write RobotState
MRT thread: read RobotState → MPC policy → write RobotJointAction
Sim thread: read RobotJointAction → apply torques → mj_step()
```

### Mode B: Distributed (Dummy sim path)

MPC runs as a **separate ROS2 node**, communicating via topics (`mpc_observation`, `mpc_policy`).

| File | Role |
|------|------|
| `humanoid_nmpc/humanoid_centroidal_mpc_ros2/exe/CentroidalMpcSqpNode.cpp` | Standalone MPC solver node |
| `humanoid_nmpc/humanoid_centroidal_mpc_ros2/exe/CentroidalMpcDummySimNode.cpp` | Dummy loop that forward-simulates using the MPC model (no physics engine) |

---

## Layer 4: Look at Launch Files and Config

Understand how a user selects which simulator to run.

| File | What it shows |
|------|---------------|
| `robot_models/unitree_g1/g1_centroidal_mpc/launch/mujoco_sim.launch.py` | Launches `mpc_sim` (in-process MuJoCo + MPC) |
| `robot_models/unitree_g1/g1_centroidal_mpc/launch/dummy_sim.launch.py` | Launches `mpc_node` + `dummy_sim_node` separately |
| `humanoid_nmpc/humanoid_common_mpc_ros2/humanoid_common_mpc_ros2/mpc_launch_config.py` | Central launch config — defines `mpc_sim`, `mpc_node`, `dummy_sim_node` executables and their parameters |

Config files that are **simulator-agnostic** (reusable with your new sim):
- `g1_centroidal_mpc/config/mpc/task.info` — MPC costs, constraints, solver params
- `humanoid_common_mpc/config/command/gait.info` — gait schedules
- `g1_centroidal_mpc/config/command/reference.info` — reference trajectories

Config that is **MuJoCo-specific**:
- `g1_description/urdf/g1_29dof.xml` — MuJoCo scene XML

---

## Layer 5: Messages and Topics

If you go the distributed route (Mode B), understand the ROS2 interface:

| Topic | Message | Direction |
|-------|---------|-----------|
| `{robot}/mpc_observation` | `ocs2_ros2_msgs/MpcObservation` | Sim → MPC |
| `{robot}/mpc_policy` | `ocs2_ros2_msgs/MpcFlattenedController` | MPC → Sim |
| `humanoid/walking_velocity_command` | `humanoid_mpc_msgs/WalkingVelocityCommand` | User → MPC |

---

## Recommended Exploration Order

1. **`RobotHWInterfaceBase.h`** → understand the contract
2. **`MujocoSimInterface.h/.cpp`** → see a working implementation
3. **`CentroidalMpcRobotSim.cpp`** → see how the sim is wired to MPC
4. **`mpc_launch_config.py`** → see how launch files select simulators
5. **A launch file** (e.g., `mujoco_sim.launch.py`) → see the end-user entry point
6. **`CMakeLists.txt`** in both `mujoco_sim_interface/` and `humanoid_centroidal_mpc_ros2/` → understand build dependencies

To add a new simulator, you'd create a new package (like `my_sim_interface/`) that implements `RobotHWInterfaceBase`, write a new `*RobotSim.cpp` executable (or refactor the existing ones to accept the interface via injection), add a new launch file, and conditionally build it in CMake — mirroring exactly what `mujoco_sim_interface` does today.

