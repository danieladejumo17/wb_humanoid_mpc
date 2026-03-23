#!/usr/bin/env python3
"""
Multi-threaded MuJoCo simulation server that communicates with the C++
TcpHWInterface client over length-prefixed Protobuf/TCP.

Threads:
  - Main thread:  MuJoCo passive viewer at ~60 Hz
  - TCP thread:   accepts one client, dispatches protobuf messages
  - Sim thread:   physics loop at dt received from INIT_CONFIG
"""

import argparse
import socket
import struct
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

import tcp_bridge_pb2 as pb


def quat_conjugate(q_wxyz: np.ndarray) -> np.ndarray:
    """Return the conjugate of a (w, x, y, z) quaternion."""
    return np.array([q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]])


def quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (w, x, y, z)."""
    w, x, y, z = q_wxyz
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


class MujocoSimServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        self.model = None
        self.data = None

        # Locks
        self.mj_lock = threading.Lock()
        self.data_lock = threading.Lock()

        # Lifecycle events / flags
        self.config_ready = threading.Event()
        self.sim_running = threading.Event()
        self.terminate = threading.Event()
        self.reset_requested = False

        # Joint mapping (populated on INIT_CONFIG)
        self.joint_names: list[str] = [] # joint names in the order of Robot Description from C++ layer
        self.qpos_indices: np.ndarray = np.array([], dtype=int) # index - TCP Message idx, value - this MuJoCo qpos idx
        self.qvel_indices: np.ndarray = np.array([], dtype=int) # index - TCP Message idx, value - this MuJoCo qvel idx
        self.actuator_indices: np.ndarray = np.array([], dtype=int) # index - TCP Message idx, value - this MuJoCo actuator idx

        # Init state backup for reset
        self.qpos_init: np.ndarray | None = None
        self.qvel_init: np.ndarray | None = None

        # Shared action buffer (written by TCP thread, read by sim thread)
        self.latest_q_des: np.ndarray = np.array([]) # index - TCP Message idx, value - this MuJoCo qpos idx
        self.latest_qd_des: np.ndarray = np.array([])
        self.latest_kp: np.ndarray = np.array([])
        self.latest_kd: np.ndarray = np.array([])
        self.latest_ff: np.ndarray = np.array([])

        # Shared state snapshot (written by sim thread, read by TCP thread)
        self.latest_state: dict = {}

        self.dt: float = 0.0005

    # =========================================================================
    # TCP framing helpers
    # =========================================================================

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes from the socket."""
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
        MujocoSimServer._send_message(conn, ack)

    # =========================================================================
    # Message handlers
    # =========================================================================

    def _handle_init_config(self, conn: socket.socket, sim_config: pb.SimConfig):
        print(f"[init_config] Loading model from: {sim_config.scene_path}")
        print(f"[init_config] dt = {sim_config.dt}")

        self.model = mujoco.MjModel.from_xml_path(sim_config.scene_path)
        self.data = mujoco.MjData(self.model)

        self.dt = sim_config.dt
        self.model.opt.timestep = self.dt

        # Build joint index map from the agreed joint_names ordering
        self.joint_names = list(sim_config.joint_names)
        n_joints = len(self.joint_names)
        self.qpos_indices = np.zeros(n_joints, dtype=int)
        self.qvel_indices = np.zeros(n_joints, dtype=int)
        self.actuator_indices = np.zeros(n_joints, dtype=int)

        for i, name in enumerate(self.joint_names):
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id < 0:
                print(f"[init_config] WARNING: joint '{name}' not found in MuJoCo model")
                continue
            self.qpos_indices[i] = self.model.jnt_qposadr[jnt_id]
            self.qvel_indices[i] = self.model.jnt_dofadr[jnt_id]

            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id < 0:
                print(f"[init_config] WARNING: actuator '{name}' not found in MuJoCo model")
            self.actuator_indices[i] = max(act_id, 0)

        print(f"[init_config] Mapped {n_joints} joints")

        # Default joint damping (matching C++ lines 93-99)
        for i in range(6, self.model.nv):
            self.model.dof_damping[i] = 10.0

        # Apply initial state if provided
        if sim_config.HasField("init_state"):
            self._apply_init_state(sim_config.init_state)

        # Save init state for resets
        self.qpos_init = self.data.qpos.copy()
        self.qvel_init = self.data.qvel.copy()

        # Initialize action buffers to zero
        self.latest_q_des = np.zeros(n_joints)
        self.latest_qd_des = np.zeros(n_joints)
        self.latest_kp = np.zeros(n_joints)
        self.latest_kd = np.zeros(n_joints)
        self.latest_ff = np.zeros(n_joints)

        self.config_ready.set()
        self._send_ack(conn)
        print("[init_config] Done.")

    def _apply_init_state(self, init_state: pb.RobotStateMsg):
        """Write the init state into mjData, converting velocity frames."""
        if len(init_state.root_position) == 3:
            self.data.qpos[0:3] = init_state.root_position

        if len(init_state.root_orientation) == 4:
            self.data.qpos[3:7] = init_state.root_orientation  # w, x, y, z

        if len(init_state.root_linear_vel) == 3:
            # Proto sends local-frame linear vel; MuJoCo qvel[0:3] is world-frame
            quat = np.array(init_state.root_orientation) if len(init_state.root_orientation) == 4 else np.array([1, 0, 0, 0])
            v_local = np.array(init_state.root_linear_vel)
            self.data.qvel[0:3] = quat_rotate(quat, v_local)

        if len(init_state.root_angular_vel) == 3:
            # Angular vel is already in local frame for MuJoCo
            self.data.qvel[3:6] = init_state.root_angular_vel

        if len(init_state.joint_positions) > 0:
            for i, pos in enumerate(init_state.joint_positions):
                if i < len(self.qpos_indices):
                    self.data.qpos[self.qpos_indices[i]] = pos

        if len(init_state.joint_velocities) > 0:
            for i, vel in enumerate(init_state.joint_velocities):
                if i < len(self.qvel_indices):
                    self.data.qvel[self.qvel_indices[i]] = vel

    def _handle_init_sim(self, conn: socket.socket):
        """Take one physics step so derived quantities are valid."""
        with self.mj_lock:
            mujoco.mj_step(self.model, self.data)
        self._snapshot_state()
        self._send_ack(conn)
        print("[init_sim] One step taken. State ready.")

    def _handle_start_sim(self, conn: socket.socket):
        self.sim_running.set()
        self._send_ack(conn)
        print("[start_sim] Simulation loop started.")

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
        print("[reset] Reset requested.")

    # =========================================================================
    # State snapshot
    # =========================================================================

    def _snapshot_state(self):
        """Read current mjData and store a thread-safe state dict."""
        quat = self.data.qpos[3:7].copy()  # w, x, y, z
        v_world = self.data.qvel[0:3].copy()
        v_local = quat_rotate(quat_conjugate(quat), v_world)
        ang_vel_local = self.data.qvel[3:6].copy()

        joint_pos = self.data.qpos[self.qpos_indices].copy() if len(self.qpos_indices) > 0 else []
        joint_vel = self.data.qvel[self.qvel_indices].copy() if len(self.qvel_indices) > 0 else []

        snapshot = {
            "time": self.data.time,
            "root_position": self.data.qpos[0:3].tolist(),
            "root_orientation": quat.tolist(),
            "root_linear_vel": v_local.tolist(),
            "root_angular_vel": ang_vel_local.tolist(),
            "joint_positions": joint_pos.tolist() if isinstance(joint_pos, np.ndarray) else joint_pos,
            "joint_velocities": joint_vel.tolist() if isinstance(joint_vel, np.ndarray) else joint_vel,
            "contact_flags": [True, True],
        }

        with self.data_lock:
            self.latest_state = snapshot

    # =========================================================================
    # Simulation thread
    # =========================================================================

    def _simulation_step(self):
        """One iteration of the physics loop."""
        with self.data_lock:
            q_des = self.latest_q_des.copy()
            qd_des = self.latest_qd_des.copy()
            kp = self.latest_kp.copy()
            kd = self.latest_kd.copy()
            ff = self.latest_ff.copy()
            do_reset = self.reset_requested
            self.reset_requested = False

        if do_reset and self.qpos_init is not None:
            with self.mj_lock:
                self.data.qpos[:] = self.qpos_init
                self.data.qvel[:] = self.qvel_init
                self.data.ctrl[:] = 0.0
                mujoco.mj_step(self.model, self.data)
            self._snapshot_state()
            time.sleep(1.0)
            return

        # Compute PD torques and apply to ctrl
        for i in range(len(self.joint_names)):
            q = self.data.qpos[self.qpos_indices[i]]
            qd = self.data.qvel[self.qvel_indices[i]]
            torque = kp[i] * (q_des[i] - q) + kd[i] * (qd_des[i] - qd) + ff[i]
            self.data.ctrl[self.actuator_indices[i]] = torque

        with self.mj_lock:
            mujoco.mj_step(self.model, self.data)

        self._snapshot_state()

        # Auto-reset if robot has fallen
        if self.data.qpos[2] < 0.2:
            print("[sim] Auto-reset triggered (z < 0.2)")
            with self.mj_lock:
                self.data.qpos[:] = self.qpos_init
                self.data.qvel[:] = self.qvel_init
                self.data.ctrl[:] = 0.0
                mujoco.mj_step(self.model, self.data)
            self._snapshot_state()
            time.sleep(1.0)

    def _sim_loop(self):
        """Sim thread entry point. Blocks until START_SIM, then runs at dt."""
        self.config_ready.wait()
        self.sim_running.wait()

        print(f"[sim_loop] Running at dt = {self.dt}")
        next_wakeup = time.monotonic()

        while not self.terminate.is_set():
            self._simulation_step()
            next_wakeup += self.dt
            sleep_time = next_wakeup - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

    # =========================================================================
    # TCP server thread
    # =========================================================================

    def _tcp_loop(self):
        """TCP thread entry point. Binds, accepts one client, dispatches."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(1)
        print(f"[tcp] Listening on {self.host}:{self.port}")

        conn, addr = server_sock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[tcp] Client connected from {addr}")

        try:
            while not self.terminate.is_set():
                msg = self._recv_message(conn)
                self._dispatch(conn, msg)
        except ConnectionError as e:
            print(f"[tcp] Connection lost: {e}")
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
            print(f"[tcp] Unknown message type: {msg.type}")

    # =========================================================================
    # Main thread (viewer)
    # =========================================================================

    def run(self):
        """Start all threads. Main thread runs the MuJoCo viewer."""
        tcp_thread = threading.Thread(target=self._tcp_loop, daemon=True)
        tcp_thread.start()

        sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        sim_thread.start()

        print("[main] Waiting for INIT_CONFIG...")
        self.config_ready.wait()

        print("[main] Launching viewer...")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and not self.terminate.is_set():
                with self.mj_lock:
                    viewer.sync()
                time.sleep(1.0 / 60.0)

        print("[main] Viewer closed. Shutting down.")
        self.terminate.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo TCP simulation server")
    parser.add_argument("--host", default="0.0.0.0", help="TCP bind address")
    parser.add_argument("--port", type=int, default=9000, help="TCP port")
    args = parser.parse_args()

    server = MujocoSimServer(args.host, args.port)
    server.run()
