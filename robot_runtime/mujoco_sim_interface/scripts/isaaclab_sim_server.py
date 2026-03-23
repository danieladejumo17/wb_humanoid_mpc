#!/usr/bin/env python3
"""
Isaac Lab 3.0 (Newton / MuJoCo-Warp) simulation server that communicates
with the C++ TcpHWInterface client over length-prefixed Protobuf/TCP.

Threads:
  - Main thread:  IsaacLab sim loop + rendering (render at ~10 Hz)
  - TCP thread:   accepts one client, dispatches protobuf messages

Convention differences vs MuJoCo server (and how they are handled):
  - Quaternion: IsaacLab uses xyzw, Protobuf uses wxyz  -> converted on read/write
  - Root velocities: IsaacLab world-frame, Protobuf local/body-frame
      -> Use root_link_lin_vel_b / root_link_ang_vel_b for state snapshot
      -> Convert local->world when writing init state
"""

import argparse
import socket
import struct
import sys
import threading
import time

# ---------------------------------------------------------------------------
# AppLauncher must be created before any IsaacLab imports
# ---------------------------------------------------------------------------

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IsaacLab TCP simulation server (Newton/MuJoCo backend)")
parser.add_argument("--host", default="0.0.0.0", help="TCP bind address")
parser.add_argument("--port", type=int, default=9000, help="TCP port")
parser.add_argument("--usd_path", default="/wb_humanoid_mpc_ws/g1_description/g1_29_dof/g1_29dof_rev_1_0.usd",
                    help="Path to robot USD file")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# IsaacLab imports (must come after AppLauncher)
# ---------------------------------------------------------------------------

import logging  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # noqa: E402
from isaaclab_visualizers.newton import NewtonVisualizerCfg  # noqa: E402

import tcp_bridge_pb2 as pb  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_torch(warp_array):
    if isinstance(warp_array, torch.Tensor):
        return warp_array
    return wp.to_torch(warp_array)


def quat_rotate_xyzw(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q in (x, y, z, w) convention."""
    x, y, z, w = q_xyzw
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def wxyz_to_xyzw(q_wxyz):
    """Convert quaternion from (w,x,y,z) to (x,y,z,w)."""
    return [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]


def xyzw_to_wxyz(q_xyzw):
    """Convert quaternion from (x,y,z,w) to (w,x,y,z)."""
    return [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]


# ---------------------------------------------------------------------------
# G1 Robot Configuration (stiffness=0, damping=10 for raw torque + passive damping)
# ---------------------------------------------------------------------------

_EFFORT_LIMITS = {
    "hip_pitch": 88.0, "hip_roll": 139.0, "hip_yaw": 88.0, "knee": 139.0,
    "ankle": 50.0, "waist_yaw": 88.0, "waist_roll_pitch": 50.0,
    "shoulder": 25.0, "elbow": 25.0, "wrist_strong": 25.0, "wrist_weak": 5.0,
}

_VELOCITY_LIMITS = {
    "hip_pitch": 32.0, "hip_roll": 20.0, "hip_yaw": 32.0, "knee": 20.0,
    "ankle": 37.0, "waist_yaw": 32.0, "waist_roll_pitch": 37.0,
    "shoulder": 37.0, "elbow": 37.0, "wrist_strong": 37.0, "wrist_weak": 22.0,
}

_STIFFNESS = 0.0
_JOINT_DAMPING = 10.0


def _make_articulation_cfg(usd_path: str) -> ArticulationCfg:
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, fix_root_link=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
        actuators={
            "hip_pitch": ImplicitActuatorCfg(
                joint_names_expr=["left_hip_pitch_joint", "right_hip_pitch_joint"],
                effort_limit_sim=_EFFORT_LIMITS["hip_pitch"],
                velocity_limit_sim=_VELOCITY_LIMITS["hip_pitch"],
                stiffness=_STIFFNESS, damping=_JOINT_DAMPING,
            ),
            "hip_roll": ImplicitActuatorCfg(
                joint_names_expr=["left_hip_roll_joint", "right_hip_roll_joint"],
                effort_limit_sim=_EFFORT_LIMITS["hip_roll"],
                velocity_limit_sim=_VELOCITY_LIMITS["hip_roll"],
                stiffness=_STIFFNESS, damping=_JOINT_DAMPING,
            ),
            "hip_yaw": ImplicitActuatorCfg(
                joint_names_expr=["left_hip_yaw_joint", "right_hip_yaw_joint"],
                effort_limit_sim=_EFFORT_LIMITS["hip_yaw"],
                velocity_limit_sim=_VELOCITY_LIMITS["hip_yaw"],
                stiffness=_STIFFNESS, damping=_JOINT_DAMPING,
            ),
            "knee": ImplicitActuatorCfg(
                joint_names_expr=["left_knee_joint", "right_knee_joint"],
                effort_limit_sim=_EFFORT_LIMITS["knee"],
                velocity_limit_sim=_VELOCITY_LIMITS["knee"],
                stiffness=_STIFFNESS, damping=_JOINT_DAMPING,
            ),
            "ankle": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                    "left_ankle_roll_joint", "right_ankle_roll_joint",
                ],
                effort_limit_sim=_EFFORT_LIMITS["ankle"],
                velocity_limit_sim=_VELOCITY_LIMITS["ankle"],
                stiffness=_STIFFNESS, damping=_JOINT_DAMPING,
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                effort_limit_sim=_EFFORT_LIMITS["waist_yaw"],
                velocity_limit_sim=_VELOCITY_LIMITS["waist_yaw"],
                stiffness=_STIFFNESS, damping=_JOINT_DAMPING,
            ),
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
                    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
                ],
                effort_limit_sim=_EFFORT_LIMITS["shoulder"],
                velocity_limit_sim=_VELOCITY_LIMITS["shoulder"],
                stiffness=_STIFFNESS, damping=_JOINT_DAMPING,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["left_elbow_joint", "right_elbow_joint"],
                effort_limit_sim=_EFFORT_LIMITS["elbow"],
                velocity_limit_sim=_VELOCITY_LIMITS["elbow"],
                stiffness=_STIFFNESS, damping=_JOINT_DAMPING,
            ),
            "wrist_strong": ImplicitActuatorCfg(
                joint_names_expr=["left_wrist_roll_joint", "right_wrist_roll_joint"],
                effort_limit_sim=_EFFORT_LIMITS["wrist_strong"],
                velocity_limit_sim=_VELOCITY_LIMITS["wrist_strong"],
                stiffness=_STIFFNESS, damping=_JOINT_DAMPING,
            ),
            "wrist_weak": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
                    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
                ],
                effort_limit_sim=_EFFORT_LIMITS["wrist_weak"],
                velocity_limit_sim=_VELOCITY_LIMITS["wrist_weak"],
                stiffness=_STIFFNESS, damping=_JOINT_DAMPING,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Newton solver config (same as g1_test_sim.py)
# ---------------------------------------------------------------------------

_SOLVER_CFG = MJWarpSolverCfg(
    solver="newton",
    integrator="implicitfast",
    njmax=2000,
    nconmax=1000,
    impratio=100.0,
    cone="elliptic",
    update_data_interval=2,
    iterations=20, # TODO
    ls_iterations=100,
    ccd_iterations=80, # TODO
    ls_parallel=True,
    use_mujoco_contacts=False, # TODO
)

_NEWTON_CFG = NewtonCfg(solver_cfg=_SOLVER_CFG, num_substeps=1)


# ===========================================================================
# IsaacLab Simulation Server
# ===========================================================================


class IsaacLabSimServer:
    def __init__(self, host: str, port: int, usd_path: str):
        self.host = host
        self.port = port
        self.usd_path = usd_path

        self.data_lock = threading.Lock()

        self.config_ready = threading.Event()
        self.scene_created = threading.Event()
        self.init_sim_ready = threading.Event()
        self.init_step_done = threading.Event()
        self.sim_running = threading.Event()
        self.terminate = threading.Event()
        self.reset_requested = False

        self.joint_names: list[str] = [] # joint names in the order of Robot Description from C++ layer
        self.joint_indices: list[int] = [] # index - TCP Message idx, value - this IsaacLab joint idx

        self.latest_q_des: np.ndarray = np.array([])
        self.latest_qd_des: np.ndarray = np.array([])
        self.latest_kp: np.ndarray = np.array([])
        self.latest_kd: np.ndarray = np.array([])
        self.latest_ff: np.ndarray = np.array([])

        self.latest_state: dict = {}
        self.dt: float = 0.0005
        self.sim_time: float = 0.0

        # Init state backup for reset (saved after INIT_CONFIG + scene creation)
        self.init_root_pose: torch.Tensor | None = None
        self.init_root_vel: torch.Tensor | None = None
        self.init_joint_pos: torch.Tensor | None = None
        self.init_joint_vel: torch.Tensor | None = None

        self.sim = None
        self.scene = None
        self.robot = None
        self.device = None

    # =========================================================================
    # TCP framing helpers (identical to MuJoCo server)
    # =========================================================================

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed by peer")
            buf.extend(chunk)
        return bytes(buf)

    def _recv_message(self, conn: socket.socket) -> pb.TcpMessage:
        length_bytes = self._recv_exact(conn, 4)
        length = struct.unpack("<I", length_bytes)[0]
        payload = self._recv_exact(conn, length)
        msg = pb.TcpMessage()
        msg.ParseFromString(payload)
        return msg

    @staticmethod
    def _send_message(conn: socket.socket, msg: pb.TcpMessage):
        payload = msg.SerializeToString()
        header = struct.pack("<I", len(payload))
        conn.sendall(header + payload)

    @staticmethod
    def _send_ack(conn: socket.socket):
        ack = pb.TcpMessage()
        ack.type = pb.ACK
        IsaacLabSimServer._send_message(conn, ack)

    # =========================================================================
    # Message handlers
    # =========================================================================

    def _handle_init_config(self, conn: socket.socket, sim_config: pb.SimConfig):
        logger.info("[init_config] dt = %s", sim_config.dt)
        self.dt = sim_config.dt

        self.joint_names = list(sim_config.joint_names)
        logger.info("[init_config] Received %d joint names", len(self.joint_names))

        if sim_config.HasField("init_state"):
            self._pending_init_state = sim_config.init_state
        else:
            self._pending_init_state = None

        self.config_ready.set()
        self.scene_created.wait()
        self._send_ack(conn)
        logger.info("[init_config] Done.")

    def _handle_init_sim(self, conn: socket.socket):
        self.init_sim_ready.set()
        self.init_step_done.wait()
        self._send_ack(conn)
        logger.info("[init_sim] Acknowledged.")

    def _handle_start_sim(self, conn: socket.socket):
        self.sim_running.set()
        self._send_ack(conn)
        logger.info("[start_sim] Simulation loop started.")

    def _handle_get_state(self, conn: socket.socket):
        response = pb.TcpMessage()
        response.type = pb.STATE_RESPONSE
        state_msg = response.robot_state

        with self.data_lock:
            s = self.latest_state

        state_msg.time = s.get("time", 0.0)
        state_msg.root_position.extend(s.get("root_position", [0, 0, 0]))
        state_msg.root_orientation.extend(s.get("root_orientation", [1, 0, 0, 0]))
        state_msg.root_linear_vel.extend(s.get("root_linear_vel", [0, 0, 0]))
        state_msg.root_angular_vel.extend(s.get("root_angular_vel", [0, 0, 0]))
        state_msg.joint_positions.extend(s.get("joint_positions", []))
        state_msg.joint_velocities.extend(s.get("joint_velocities", []))
        state_msg.contact_flags.extend(s.get("contact_flags", [True, True]))

        self._send_message(conn, response)

    def _handle_send_action(self, conn: socket.socket, action: pb.JointActionMsg):
        with self.data_lock:
            n = len(self.joint_names)
            if len(action.q_des) == n:
                self.latest_q_des[:] = action.q_des
            if len(action.qd_des) == n:
                self.latest_qd_des[:] = action.qd_des
            if len(action.kp) == n:
                self.latest_kp[:] = action.kp
            if len(action.kd) == n:
                self.latest_kd[:] = action.kd
            if len(action.feed_forward_effort) == n:
                self.latest_ff[:] = action.feed_forward_effort
        self._send_ack(conn)

    def _handle_reset(self, conn: socket.socket):
        with self.data_lock:
            self.reset_requested = True
        self._send_ack(conn)
        logger.info("[reset] Reset requested.")

    # =========================================================================
    # Scene creation (called on main thread after INIT_CONFIG)
    # =========================================================================

    def _create_scene(self):
        render_interval = max(1, int(1.0 / (10.0 * self.dt)))
        logger.info("[scene] Creating with dt=%.6f  render_interval=%d", self.dt, render_interval)

        sim_cfg = sim_utils.SimulationCfg(
            dt=self.dt,
            render_interval=render_interval,
            physics=_NEWTON_CFG,
            device=args_cli.device,
            visualizer_cfgs=NewtonVisualizerCfg(
                update_frequency=render_interval,
                camera_position=(3.5, 0.0, 3.2),
                camera_target=(0.0, 0.0, 0.5),
            ),
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
        self.device = self.sim.device

        art_cfg = _make_articulation_cfg(self.usd_path)

        class _SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
            dome_light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
            )
            robot = art_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

        scene_cfg = _SceneCfg(num_envs=1, env_spacing=2.5)
        self.scene = InteractiveScene(scene_cfg)
        self.robot = self.scene["robot"]

        self.sim.reset()

        self._build_joint_mapping()
        self._apply_init_state()

        n_joints = len(self.joint_names)
        self.latest_q_des = np.zeros(n_joints)
        self.latest_qd_des = np.zeros(n_joints)
        self.latest_kp = np.zeros(n_joints)
        self.latest_kd = np.zeros(n_joints)
        self.latest_ff = np.zeros(n_joints)

        logger.info("[scene] Scene ready. joints=%d  device=%s", self.robot.num_joints, self.device)

    def _build_joint_mapping(self):
        """Map INIT_CONFIG joint names to IsaacLab joint indices."""
        isaac_joint_names = self.robot.joint_names
        name_to_idx = {name: i for i, name in enumerate(isaac_joint_names)}

        self.joint_indices = []
        for name in self.joint_names:
            if name in name_to_idx:
                self.joint_indices.append(name_to_idx[name])
            else:
                logger.warning("[joint_map] Joint '%s' not found in IsaacLab model", name)
                self.joint_indices.append(0)

        self.joint_idx_tensor = torch.tensor(self.joint_indices, dtype=torch.long, device=self.device)
        self.joint_idx_np = np.array(self.joint_indices, dtype=np.intp)

        logger.info("[joint_map] Mapped %d joints. IsaacLab has %d total joints.",
                    len(self.joint_indices), len(isaac_joint_names))

    def _apply_init_state(self):
        """Write the pending init state from INIT_CONFIG into the sim and save for resets.

        Any field not provided in INIT_CONFIG falls back to the robot's default state.
        """
        init_state = getattr(self, "_pending_init_state", None)

        default_root = _to_torch(self.robot.data.default_root_state).clone()
        default_root[:, :3] += self.scene.env_origins

        root_pose = default_root[:, :7].clone()
        root_vel = default_root[:, 7:].clone()
        jpos = _to_torch(self.robot.data.default_joint_pos).clone()
        jvel = _to_torch(self.robot.data.default_joint_vel).clone()

        if init_state is not None:
            if len(init_state.root_position) == 3:
                root_pose[0, :3] = torch.tensor(list(init_state.root_position), device=self.device)

            if len(init_state.root_orientation) == 4:
                q_xyzw = wxyz_to_xyzw(list(init_state.root_orientation))
                root_pose[0, 3:7] = torch.tensor(q_xyzw, device=self.device)

            q_xyzw_np = root_pose[0, 3:7].cpu().numpy()

            if len(init_state.root_linear_vel) == 3:
                v_local = np.array(list(init_state.root_linear_vel))
                v_world = quat_rotate_xyzw(q_xyzw_np, v_local)
                root_vel[0, :3] = torch.tensor(v_world, dtype=torch.float32, device=self.device)

            if len(init_state.root_angular_vel) == 3:
                w_local = np.array(list(init_state.root_angular_vel))
                w_world = quat_rotate_xyzw(q_xyzw_np, w_local)
                root_vel[0, 3:6] = torch.tensor(w_world, dtype=torch.float32, device=self.device)

            if len(init_state.joint_positions) > 0:
                for i, idx in enumerate(self.joint_indices):
                    if i < len(init_state.joint_positions):
                        jpos[0, idx] = init_state.joint_positions[i]

            if len(init_state.joint_velocities) > 0:
                for i, idx in enumerate(self.joint_indices):
                    if i < len(init_state.joint_velocities):
                        jvel[0, idx] = init_state.joint_velocities[i]

        self.robot.write_root_link_pose_to_sim_index(root_pose=root_pose)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=root_vel)
        self.robot.write_joint_position_to_sim_index(position=jpos)
        self.robot.write_joint_velocity_to_sim_index(velocity=jvel)

        self.init_root_pose = root_pose.clone()
        self.init_root_vel = root_vel.clone()
        self.init_joint_pos = jpos.clone()
        self.init_joint_vel = jvel.clone()

        logger.info("[init_state] Applied and saved initial state.")

    # =========================================================================
    # State snapshot
    # =========================================================================

    def _snapshot_state(self):
        """Read IsaacLab state and store as a thread-safe dict.

        Conventions:
          - Quaternion: convert xyzw (IsaacLab) -> wxyz (Protobuf)
          - Velocities: use body-frame accessors (root_link_*_vel_b)
        """
        d = self.robot.data
        idx = self.joint_idx_np

        pos = _to_torch(d.root_link_pos_w)[0].cpu().numpy()
        quat_xyzw = _to_torch(d.root_link_quat_w)[0].cpu().numpy()
        lin_vel_b = _to_torch(d.root_link_lin_vel_b)[0].cpu().numpy()
        ang_vel_b = _to_torch(d.root_link_ang_vel_b)[0].cpu().numpy()
        all_jpos = _to_torch(d.joint_pos)[0].cpu().numpy()
        all_jvel = _to_torch(d.joint_vel)[0].cpu().numpy()

        snapshot = {
            "time": self.sim_time,
            "root_position": pos.tolist(),
            "root_orientation": [float(quat_xyzw[3]), float(quat_xyzw[0]),
                                 float(quat_xyzw[1]), float(quat_xyzw[2])],
            "root_linear_vel": lin_vel_b.tolist(),
            "root_angular_vel": ang_vel_b.tolist(),
            "joint_positions": all_jpos[idx].tolist(),
            "joint_velocities": all_jvel[idx].tolist(),
            "contact_flags": [True, True],
        }

        with self.data_lock:
            self.latest_state = snapshot

    # =========================================================================
    # Reset
    # =========================================================================

    def _reset_robot(self):
        """Reset robot to the INIT_CONFIG state (matching MuJoCo server's qpos_init/qvel_init reset)."""
        self.robot.write_root_link_pose_to_sim_index(root_pose=self.init_root_pose.clone())
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=self.init_root_vel.clone())
        self.robot.write_joint_position_to_sim_index(position=self.init_joint_pos.clone())
        self.robot.write_joint_velocity_to_sim_index(velocity=self.init_joint_vel.clone())
        zero_efforts = torch.zeros_like(self.init_joint_pos)
        self.robot.set_joint_effort_target_index(target=zero_efforts)
        self.scene.reset()

    # =========================================================================
    # Simulation step
    # =========================================================================

    def _simulation_step(self):
        with self.data_lock:
            q_des = self.latest_q_des.copy()
            qd_des = self.latest_qd_des.copy()
            kp = self.latest_kp.copy()
            kd = self.latest_kd.copy()
            ff = self.latest_ff.copy()
            do_reset = self.reset_requested
            self.reset_requested = False

        if do_reset:
            self._reset_robot()
            self._do_sim_step()
            self._snapshot_state()
            time.sleep(1.0)
            return

        idx = self.joint_idx_tensor
        q = _to_torch(self.robot.data.joint_pos)[0, idx]
        qd = _to_torch(self.robot.data.joint_vel)[0, idx]

        q_des_t = torch.as_tensor(q_des, dtype=torch.float32, device=self.device)
        qd_des_t = torch.as_tensor(qd_des, dtype=torch.float32, device=self.device)
        kp_t = torch.as_tensor(kp, dtype=torch.float32, device=self.device)
        kd_t = torch.as_tensor(kd, dtype=torch.float32, device=self.device)
        ff_t = torch.as_tensor(ff, dtype=torch.float32, device=self.device)

        tau = kp_t * (q_des_t - q) + kd_t * (qd_des_t - qd) + ff_t

        torques = torch.zeros(1, self.robot.num_joints, device=self.device)
        torques[0, idx] = tau

        self.robot.set_joint_effort_target_index(target=torques)
        self._do_sim_step()
        self._snapshot_state()

        root_z = _to_torch(self.robot.data.root_link_pos_w)[0, 2].item()
        if root_z < 0.2:
            logger.info("[sim] Auto-reset triggered (z=%.3f < 0.2)", root_z)
            self._reset_robot()
            self._do_sim_step()
            self._snapshot_state()
            time.sleep(1.0)

    def _do_sim_step(self):
        """Execute one IsaacLab physics step."""
        self.scene.write_data_to_sim()
        self.sim.step()
        self.sim_time += self.dt
        self.scene.update(self.dt)

    # =========================================================================
    # TCP server thread
    # =========================================================================

    def _tcp_loop(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(1)
        logger.info("[tcp] Listening on %s:%d", self.host, self.port)

        conn, addr = server_sock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        logger.info("[tcp] Client connected from %s", addr)

        try:
            while not self.terminate.is_set():
                msg = self._recv_message(conn)
                self._dispatch(conn, msg)
        except ConnectionError as e:
            logger.info("[tcp] Connection lost: %s", e)
        finally:
            conn.close()
            server_sock.close()
            self.terminate.set()

    def _dispatch(self, conn: socket.socket, msg: pb.TcpMessage):
        if msg.type == pb.INIT_CONFIG:
            self._handle_init_config(conn, msg.sim_config)
        elif msg.type == pb.INIT_SIM:
            self._handle_init_sim(conn)
        elif msg.type == pb.START_SIM:
            self._handle_start_sim(conn)
        elif msg.type == pb.GET_STATE:
            self._handle_get_state(conn)
        elif msg.type == pb.SEND_ACTION:
            self._handle_send_action(conn, msg.joint_action)
        elif msg.type == pb.RESET:
            self._handle_reset(conn)
        else:
            logger.warning("[tcp] Unknown message type: %s", msg.type)

    # =========================================================================
    # Main entry point
    # =========================================================================

    def run(self):
        """Main thread: start TCP, wait for config, create scene, run sim loop."""
        tcp_thread = threading.Thread(target=self._tcp_loop, daemon=True)
        tcp_thread.start()

        logger.info("[main] Waiting for INIT_CONFIG...")
        self.config_ready.wait()

        logger.info("[main] Creating IsaacLab scene...")
        self._create_scene()
        self.scene_created.set()

        logger.info("[main] Waiting for INIT_SIM...")
        self.init_sim_ready.wait()

        logger.info("[main] Taking initial physics step...")
        self._do_sim_step()
        self._snapshot_state()
        self.init_step_done.set()

        logger.info("[main] Waiting for START_SIM...")
        self.sim_running.wait()

        logger.info("[main] Entering simulation loop at dt=%.6f", self.dt)
        next_wakeup = time.monotonic()
        wall_start = time.monotonic()
        sim_start = self.sim_time
        step_count = 0
        log_interval = 5.0
        next_log = wall_start + log_interval
        step_dur_sum = 0.0
        step_dur_max = 0.0

        while simulation_app.is_running() and not self.terminate.is_set():
            t0 = time.monotonic()
            self._simulation_step()
            step_dur = time.monotonic() - t0
            step_dur_sum += step_dur
            step_dur_max = max(step_dur_max, step_dur)
            step_count += 1

            next_wakeup += self.dt
            sleep_time = next_wakeup - time.monotonic()
            # if sleep_time > 0:
            #     time.sleep(sleep_time)
            if sleep_time > 2e-3:
                time.sleep(sleep_time - 1e-3)
            while time.monotonic() < next_wakeup:
                pass

            now = time.monotonic()
            if now >= next_log:
                wall_elapsed = now - wall_start
                sim_elapsed = self.sim_time - sim_start
                ratio = sim_elapsed / wall_elapsed if wall_elapsed > 0 else 0.0
                drift = sim_elapsed - wall_elapsed
                avg_step = (step_dur_sum / step_count) * 1e3
                max_step = step_dur_max * 1e3
                logger.info(
                    "[perf] wall=%.1fs  sim=%.1fs  ratio=%.4f  drift=%+.3fs  "
                    "step_avg=%.2fms  step_max=%.2fms  steps=%d",
                    wall_elapsed, sim_elapsed, ratio, drift,
                    avg_step, max_step, step_count,
                )
                step_dur_sum = 0.0
                step_dur_max = 0.0
                step_count = 0
                wall_start = now
                sim_start = self.sim_time
                next_log = now + log_interval

        logger.info("[main] Shutting down.")
        self.terminate.set()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    server = IsaacLabSimServer(args_cli.host, args_cli.port, args_cli.usd_path)
    server.run()
    simulation_app.close()
