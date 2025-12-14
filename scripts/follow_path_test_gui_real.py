#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nav2 FollowPath Test GUI Client (map統一版)
機能:
  - 規定のパス生成とNav2への送信 (ActionClient)
  - start_pose TFは発行しない（TF不安定対策）
  - Nav2へ送る参照経路は map 座標系
  - RVizで描画する経路・軌跡も map 座標系
  - 自己位置は map->base を listen し、
    経路追従開始時の start_pose(origin) を使って数値変換し、
    実質 start_pose 基準の位置姿勢をCSV保存する
  - ロボットの軌跡、指令値、オドメトリのCSV記録
  - RViz上の可視化 (Marker)
  - TkinterによるGUI操作
"""

# --- Standard Library Imports ---
import math
import threading
import time
import datetime
import csv
import os
import copy

# --- Third Party Imports ---
import numpy as np
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from tkinter import messagebox, ttk

# --- ROS 2 Imports ---
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    qos_profile_sensor_data,
    ReliabilityPolicy
)

# --- ROS 2 Messages ---
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Twist, Quaternion
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import BatteryState, Imu
from nav2_msgs.action import FollowPath
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool

# --- TF2 Imports ---
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


# ==========================================
# Helper Functions
# ==========================================

def yaw_to_quat(z_yaw_rad: float) -> tuple:
    """Yaw角(rad)からクォータニオン(x, y, z, w)を生成する"""
    half = z_yaw_rad * 0.5
    qz = math.sin(half)
    qw = math.cos(half)
    return (0.0, 0.0, qz, qw)


def make_path(frame_id: str) -> tuple:
    """
    実験用の規定パスを生成する関数
    Returns:
        (path_A, path_B, path_C): 生成された3種類のPathメッセージ
    """
    paths = []
    theta_list = [np.pi/4, np.pi/2, 3*np.pi/4]
    l_segment = 3.0

    for theta in theta_list:
        x1 = np.linspace(0, 1, 100)
        y1 = np.zeros_like(x1)

        x2 = np.linspace(1.0, 1.0 + l_segment * math.cos(theta), 300)
        y2 = np.linspace(0.0, l_segment * math.sin(theta), 300)

        x3 = np.linspace(
            1.0 + l_segment * math.cos(theta),
            4.0 + l_segment * math.cos(theta), 300)
        y3 = np.ones_like(x3) * l_segment * math.sin(theta)

        xs = np.concatenate([x1, x2, x3])
        ys = np.concatenate([y1, y2, y3])

        dx = np.gradient(xs)
        dy = np.gradient(ys)
        yaws = np.unwrap(np.arctan2(dy, dx))

        path = Path()
        path.header.frame_id = frame_id

        for x, y, yaw in zip(xs, ys, yaws):
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            _, _, qz, qw = yaw_to_quat(yaw)
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path.poses.append(ps)

        paths.append(path)

    return (paths[0], paths[1], paths[2])


def make_iso_path(frame_id: str) -> tuple:
    """
    ISO用の規定パス
    Returns:
        (path_A, path_B, path_C)
    """
    paths = []
    Lu = 0.985

    x_A = np.linspace(0, 5*Lu, 500)
    y_A = np.zeros_like(x_A)

    x_B1 = np.linspace(0, 5*Lu, 500)
    y_B1 = np.zeros_like(x_B1)
    y_B2 = np.linspace(0, -5*Lu, 500)
    x_B2 = np.ones_like(y_B2) * 5 * Lu
    x_B3 = np.linspace(5*Lu, 0, 500)
    y_B3 = np.ones_like(x_B3) * -5 * Lu
    y_B4 = np.linspace(-5*Lu, 0, 500)
    x_B4 = np.zeros_like(y_B4)
    x_B = np.concatenate([x_B1, x_B2, x_B3, x_B4])
    y_B = np.concatenate([y_B1, y_B2, y_B3, y_B4])

    x_C1 = np.linspace(0, 5*Lu, 500)
    y_C1 = np.zeros_like(x_C1)
    theta_list = np.linspace(0, np.pi/2, 500)
    x_C2 = 5*Lu*(1+np.sin(theta_list))
    y_C2 = 5*Lu*(np.cos(theta_list)-1)
    x_C = np.concatenate([x_C1, x_C2])
    y_C = np.concatenate([y_C1, y_C2])

    x_list = [x_A, x_B, x_C]
    y_list = [y_A, y_B, y_C]

    for xs, ys in zip(x_list, y_list):
        dx = np.gradient(xs)
        dy = np.gradient(ys)
        yaws = np.unwrap(np.arctan2(dy, dx))

        path = Path()
        path.header.frame_id = frame_id

        for x, y, yaw in zip(xs, ys, yaws):
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            _, _, qz, qw = yaw_to_quat(yaw)
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path.poses.append(ps)

        paths.append(path)

    return (paths[0], paths[1], paths[2])


# ==========================================
# Main ROS 2 Node Class
# ==========================================

class FollowPathClient(Node):
    """
    Nav2のFollowPathアクションを呼び出し、実験データを記録・可視化するノード
    - 経路・可視化は map 統一
    - start_pose TFは発行しない
    - ログは start_pose(origin) で数値変換して保存
    """
    def __init__(self):
        super().__init__('follow_path_gui_client')

        # --- Parameters ---
        self.record_frequency = self.declare_parameter('record_frequency', 30).value
        self.data_dir = self.declare_parameter('data_dir', '/tmp').value
        self.map_frame_id = self.declare_parameter('map_frame_id', 'map').value
        self.base_frame_id = self.declare_parameter('base_frame_id', 'base_footprint').value
        self.experiment_name = self.declare_parameter('experiment_name', 'dwpp').value  # dwpp or nelson

        # --- Internal State Variables ---
        self._reentrant_group = ReentrantCallbackGroup()
        self._current_goal_handle = None
        self._recording = False
        self._active_traj = None
        self.path_name = None
        self._traj_lock = threading.Lock()
        self._record_lock = threading.Lock()
        self._controller_id = None
        self._path_publish_ready_at = time.time() + 2.0

        # start_pose(origin) を map 基準で保持（TFは出さない）
        self.start_origin_pose_map = None          # geometry_msgs/Point 相当 (translation)
        self.start_origin_quat_map = None          # geometry_msgs/Quaternion 相当 (rotation)

        # パス可視化用（常にmap）
        self._path_publish_paths = make_path(self.map_frame_id)
        # self._path_publish_paths = make_iso_path(self.map_frame_id)

        # 現在姿勢（map）
        self.current_pose_map = None
        self.current_quat_map = None

        # 現在姿勢（start_pose基準, ログ用）
        self.current_pose_start = None
        self.current_quat_start = None

        # データ記録用バッファ
        self._reset_record_buffer()

        # 軌跡描画用の点列バッファ（mapで描く）
        self._traj_points = {'PP': [], 'APP': [], 'RPP': [], 'DWPP': []}
        self._traj_colors = {
            'PP':   (1.0, 0.0, 0.0),
            'APP':  (0.0, 0.7, 0.2),
            'RPP':  (0.0, 0.4, 1.0),
            'DWPP': (0.8, 0.2, 0.8)
        }

        # 受信データキャッシュ
        self.current_odom = None
        self.current_cmd_vel_nav = None
        self.current_cmd_vel = None
        self.current_velocity_violation = False
        self.battery_voltage = float('nan')
        self.battery_current = float('nan')
        self.battery_percent = float('nan')
        self.imu_angular_vel_x = float('nan')
        self.imu_angular_vel_y = float('nan')
        self.imu_angular_vel_z = float('nan')
        self.imu_linear_acc_x = float('nan')
        self.imu_linear_acc_y = float('nan')
        self.imu_linear_acc_z = float('nan')

        # --- QoS Settings ---
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # --- TF Components (listenのみ) ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Action Client ---
        self._client = ActionClient(self, FollowPath, '/follow_path')

        # --- Publishers ---
        self._initpose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        self._label_pub = self.create_publisher(MarkerArray, '/viz/path_labels', latched_qos)
        self._path_markers_pub = self.create_publisher(MarkerArray, '/viz/path_markers', latched_qos)
        self._traj_pub = self.create_publisher(MarkerArray, '/viz/robot_trajs', 10)

        # --- Subscribers ---
        self._odom_sub = self.create_subscription(Odometry, '/odom', self._on_odom, qos_profile_sensor_data)
        self._cmd_vel_nav_sub = self.create_subscription(Twist, '/cmd_vel_nav', self._cmd_vel_nav_callback, 1)
        self._cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_callback, 1)
        self._velocity_violation_sub = self.create_subscription(
            Bool, '/constraints_violation_flag', self._velocity_violation_callback, 1)
        self.battery_state_sub = self.create_subscription(
            BatteryState, '/whill/states/battery_state', self._battery_state_callback, 1)
        self.imu_sub = self.create_subscription(Imu, '/ouster/imu', self._imu_callback, qos)

        # --- Background Threads & Timers ---
        threading.Thread(target=self._auto_publish_initial_pose, daemon=True).start()

        self._path_publish_timer = self.create_timer(
            0.2, self._periodic_path_publish, callback_group=self._reentrant_group
        )

        # ロボット姿勢取得（mapでlisten）
        self._get_robot_pose_timer = self.create_timer(
            1.0 / 60.0,  # 60 Hz
            self._get_robot_pose_loop,
            callback_group=self._reentrant_group
        )

        # データ記録ループ
        self.record_timer = self.create_timer(
            1.0 / float(self.record_frequency),
            self._recording_loop,
            callback_group=self._reentrant_group
        )

        # 軌跡描画ループ（mapで描く）
        self._traj_draw_timer = self.create_timer(
            0.1,  # 10 Hz
            self._trajectory_draw_loop,
            callback_group=self._reentrant_group
        )

    # =========================================================================
    # Callbacks (Subscribers)
    # =========================================================================

    def _get_robot_pose_loop(self):
        """ロボットの現在位置を map で取得し、start_pose基準も数値変換で更新"""
        pose_map, quat_map = self._get_robot_pose(from_frame=self.map_frame_id)
        if pose_map is None:
            return

        self.current_pose_map = pose_map
        self.current_quat_map = quat_map

        # start origin が未設定なら、start基準はまだ作れない
        if self.start_origin_pose_map is None or self.start_origin_quat_map is None:
            return

        t_rel, r_rel = self._transform_map_to_start(pose_map, quat_map)

        self.current_pose_start = Point(
            x=float(t_rel[0]),
            y=float(t_rel[1]),
            z=float(t_rel[2])
        )
        q = r_rel.as_quat()
        self.current_quat_start = Quaternion(
            x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])
        )

    def _imu_callback(self, msg: Imu):
        self.imu_angular_vel_x = msg.angular_velocity.x
        self.imu_angular_vel_y = msg.angular_velocity.y
        self.imu_angular_vel_z = msg.angular_velocity.z
        self.imu_linear_acc_x = msg.linear_acceleration.x
        self.imu_linear_acc_y = msg.linear_acceleration.y
        self.imu_linear_acc_z = msg.linear_acceleration.z

    def _battery_state_callback(self, msg: BatteryState):
        self.battery_voltage = msg.voltage
        self.battery_current = msg.current

    def _on_odom(self, msg: Odometry):
        self.current_odom = msg

    def _cmd_vel_nav_callback(self, msg: Twist):
        self.current_cmd_vel_nav = msg

    def _cmd_vel_callback(self, msg: Twist):
        self.current_cmd_vel = msg

    def _velocity_violation_callback(self, msg: Bool):
        self.current_velocity_violation = msg.data

    # =========================================================================
    # Data Recording & CSV Logic
    # =========================================================================

    def _reset_record_buffer(self):
        """記録用バッファを初期化"""
        self.record_state_dict = {
            "sec": [], "nsec": [],
            # start_pose基準の自己位置（数値変換）
            "x": [], "y": [], "yaw": [],
            "v_cmd": [], "w_cmd": [],
            "battery_v": [], "battery_i": [], "battery_percent": [],
            "v_real": [], "w_real": [],
            "imu_ax": [], "imu_ay": [], "imu_az": [],
            "imu_vx": [], "imu_vy": [], "imu_vz": [],
            "v_nav": [], "w_nav": [],
            "velocity_violation": []
        }

    def _recording_loop(self):
        """
        データ記録用Timerコールバック。
        _recording True の間、start_pose基準(数値変換)の状態を記録。
        FalseになったらCSV保存。
        """
        if (self.current_cmd_vel_nav is None or
            self.current_cmd_vel is None or
            self.current_odom is None):
            return

        # start_pose基準がまだ作れていないなら記録しない
        if self.current_pose_start is None or self.current_quat_start is None:
            return

        data_snapshot = None

        with self._record_lock:
            if self._recording:
                # yaw
                r = R.from_quat([
                    self.current_quat_start.x, self.current_quat_start.y,
                    self.current_quat_start.z, self.current_quat_start.w
                ])
                yaw = r.as_euler('xyz')[2]

                now_ns = self.get_clock().now().nanoseconds
                sec = now_ns * 1e-9
                nsec = now_ns % int(1e9)

                d = self.record_state_dict
                d["sec"].append(sec)
                d["nsec"].append(nsec)
                d["x"].append(self.current_pose_start.x)
                d["y"].append(self.current_pose_start.y)
                d["yaw"].append(yaw)

                d["v_real"].append(self.current_odom.twist.twist.linear.x)
                d["w_real"].append(self.current_odom.twist.twist.angular.z)

                d["v_cmd"].append(self.current_cmd_vel_nav.linear.x)
                d["w_cmd"].append(self.current_cmd_vel_nav.angular.z)

                d["v_nav"].append(self.current_cmd_vel.linear.x)
                d["w_nav"].append(self.current_cmd_vel.angular.z)

                d["velocity_violation"].append(self.current_velocity_violation)

                d["battery_v"].append(self.battery_voltage)
                d["battery_i"].append(self.battery_current)
                d["battery_percent"].append(self.battery_percent)

                d["imu_ax"].append(self.imu_linear_acc_x)
                d["imu_ay"].append(self.imu_linear_acc_y)
                d["imu_az"].append(self.imu_linear_acc_z)
                d["imu_vx"].append(self.imu_angular_vel_x)
                d["imu_vy"].append(self.imu_angular_vel_y)
                d["imu_vz"].append(self.imu_angular_vel_z)
            else:
                if len(self.record_state_dict["x"]) > 0:
                    data_snapshot = {k: v.copy() for k, v in self.record_state_dict.items()}
                    self._reset_record_buffer()

        if data_snapshot is not None:
            self._save_to_csv(data_snapshot, self.path_name, self._controller_id)

    def _save_to_csv(self, data_snapshot: dict, traj_name: str, controller_id: str):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{self.data_dir}/{self.experiment_name}"
        traj_label = traj_name or "None"
        controller_label = controller_id or "None"
        filename = f"{dir_name}/{traj_label}_{controller_label}_{timestamp}.csv"

        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = list(data_snapshot.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            data_len = len(data_snapshot["x"])
            for i in range(data_len):
                row = {field: data_snapshot[field][i] for field in fieldnames}
                writer.writerow(row)

        self.get_logger().info(f"Saved trajectory data to '{filename}'")

    # =========================================================================
    # Action Client Logic (FollowPath)
    # =========================================================================

    def send_path(self, path_msg: Path, path_name: str, controller_id: str, goal_checker_id: str):
        """指定されたパス(map座標)をNav2に送信する。送信時点のロボット姿勢を start_origin として保存する。"""
        self.path_name = path_name

        if not self._client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Action server `/follow_path` not available.")
            return

        # 1) 経路追従開始時点の start_origin(map基準) を更新
        self._update_start_origin_from_current_map_pose()
        if self.start_origin_pose_map is None:
            self.get_logger().error("Failed to set start origin (map->base TF unavailable).")
            return

        # 2) pathは map 座標系前提
        if path_msg.header.frame_id != self.map_frame_id:
            self.get_logger().warn(
                f"Path frame_id is '{path_msg.header.frame_id}', but expected '{self.map_frame_id}'. "
                "For safety, forcing it to map without transforming points."
            )
            path_to_send = copy.deepcopy(path_msg)
            path_to_send.header.frame_id = self.map_frame_id
            for ps in path_to_send.poses:
                ps.header.frame_id = self.map_frame_id
        else:
            path_to_send = path_msg  # そのまま送る

        goal = FollowPath.Goal()
        goal.path = path_to_send
        goal.controller_id = controller_id
        goal.goal_checker_id = goal_checker_id

        with self._traj_lock:
            self._active_traj = controller_id
            self._traj_points[controller_id] = []
            self._recording = True
            self._controller_id = controller_id

        send_future = self._client.send_goal_async(goal, feedback_callback=self._feedback_cb)
        send_future.add_done_callback(lambda f: self._goal_response_cb(f, controller_id))

    def _goal_response_cb(self, future, controller_id):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected.')
            with self._traj_lock:
                self._recording = False
            return

        self.get_logger().info(f'Goal accepted. Controller: {controller_id}')
        self._current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        with self._traj_lock:
            self._recording = False

        try:
            result = future.result()
            self.get_logger().info(f'Result: status={result.status}')
        except Exception as e:
            self.get_logger().error(f'Result callback failed: {e}')

    def cancel_current_goal(self):
        if self._current_goal_handle is None:
            self.get_logger().info('No active goal to cancel.')
            return

        self.get_logger().info('Canceling goal...')
        self._current_goal_handle.cancel_goal_async()
        with self._traj_lock:
            self._recording = False

    def _feedback_cb(self, feedback_msg):
        pass

    # =========================================================================
    # Visualization (map統一)
    # =========================================================================

    def _trajectory_draw_loop(self):
        """ロボット軌跡を map 座標系で描画"""
        with self._traj_lock:
            active_traj = self._active_traj

        if active_traj is None:
            return
        if self.current_pose_map is None:
            return

        with self._traj_lock:
            self._draw_robot_trajectory_map(self.current_pose_map, active_traj)

    def _draw_robot_trajectory_map(self, current_pos_map, traj_name):
        """ロボットの軌跡をMarkerArrayとしてPublish (map)"""
        if traj_name not in self._traj_points:
            return

        pts = self._traj_points[traj_name]
        current_point = Point(x=current_pos_map.x, y=current_pos_map.y, z=current_pos_map.z)

        if not pts or self._distance_2d(pts[-1], current_point) > 0.05:
            pts.append(current_point)
            if len(pts) > 5000:
                self._traj_points[traj_name] = pts[-2000:]

        marr = MarkerArray()
        now = self.get_clock().now().to_msg()
        mid = 0

        for name, points in self._traj_points.items():
            if len(points) < 2:
                continue

            r, g, b = self._traj_colors.get(name, (0.5, 0.5, 0.5))

            m = Marker()
            m.header.frame_id = self.map_frame_id
            m.header.stamp = now
            m.ns = 'robot_trajectory'
            m.id = mid
            mid += 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.035
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.95
            m.points = copy.deepcopy(points)
            marr.markers.append(m)

        self._traj_pub.publish(marr)

    def publish_paths_and_labels(self, path_A: Path, path_B: Path, path_C: Path):
        """規定パスとラベルをRVizに表示（map）"""
        markers = MarkerArray()
        path_configs = [
            (path_A, (1.0, 0.0, 0.0), 'Path A'),
            (path_B, (0.0, 1.0, 0.0), 'Path B'),
            (path_C, (0.0, 0.0, 1.0), 'Path C')
        ]
        now = self.get_clock().now().to_msg()

        for mid, (path, color, _) in enumerate(path_configs):
            if not path.poses:
                continue

            m = Marker()
            m.header.frame_id = self.map_frame_id
            m.header.stamp.sec = 0
            m.header.stamp.nanosec = 0
            m.ns = 'path_visualization'
            m.id = mid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.05
            m.color.r = color[0]; m.color.g = color[1]; m.color.b = color[2]; m.color.a = 1.0

            for ps in path.poses:
                m.points.append(ps.pose.position)
            markers.markers.append(m)

        self._path_markers_pub.publish(markers)

        labels = MarkerArray()
        for mid, (path, _, name) in enumerate(path_configs, start=100):
            m = Marker()
            m.header.frame_id = self.map_frame_id
            m.header.stamp = now
            m.ns = 'path_labels'
            m.id = mid
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            if path.poses:
                m.pose.position.x = path.poses[-1].pose.position.x + 0.1
                m.pose.position.y = path.poses[-1].pose.position.y + 0.1
                m.pose.position.z = 0.3
            m.scale.z = 0.25
            m.color.r = 1.0; m.color.g = 1.0; m.color.b = 1.0; m.color.a = 1.0
            m.text = name
            labels.markers.append(m)

        self._label_pub.publish(labels)

    def clear_trajectory(self):
        with self._traj_lock:
            for k in self._traj_points:
                self._traj_points[k] = []

        ma = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        ma.markers.append(m)
        self._traj_pub.publish(ma)
        self.get_logger().info("Cleared all robot trajectories.")

    # =========================================================================
    # Transform Helpers & Utilities
    # =========================================================================

    def _update_start_origin_from_current_map_pose(self):
        """現在のロボット位置姿勢(map->base)を start_origin として保存"""
        pose_map, quat_map = self._get_robot_pose(from_frame=self.map_frame_id)
        if pose_map is None:
            return
        self.start_origin_pose_map = pose_map
        self.start_origin_quat_map = quat_map

    def _transform_map_to_start(self, pose_map, quat_map):
        """
        map 座標系の pose を start_pose基準に変換（TFは使わず数値計算）
        """
        t0 = np.array([self.start_origin_pose_map.x, self.start_origin_pose_map.y, self.start_origin_pose_map.z])
        r0 = R.from_quat([self.start_origin_quat_map.x, self.start_origin_quat_map.y,
                          self.start_origin_quat_map.z, self.start_origin_quat_map.w])

        t = np.array([pose_map.x, pose_map.y, pose_map.z])
        r = R.from_quat([quat_map.x, quat_map.y, quat_map.z, quat_map.w])

        t_rel = r0.inv().apply(t - t0)
        r_rel = r0.inv() * r

        return t_rel, r_rel

    def _get_robot_pose(self, from_frame: str):
        """指定フレームから見たロボット(base_frame_id)の位置姿勢を取得"""
        try:
            tf = self.tf_buffer.lookup_transform(
                from_frame,
                self.base_frame_id,
                rclpy.time.Time()
            )
            return tf.transform.translation, tf.transform.rotation
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return None, None

    def _distance_2d(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    # =========================================================================
    # Periodic Tasks
    # =========================================================================

    def _auto_publish_initial_pose(self):
        """起動直後に初期位置(0,0,0)をPublish（RViz初期化用）"""
        time.sleep(1.0)
        for _ in range(3):
            self.publish_initial_pose(0.0, 0.0, 0.0)
            time.sleep(0.5)

    def _periodic_path_publish(self):
        """定期的にパスをRVizにPublish（再接続時対策）"""
        if time.time() < self._path_publish_ready_at:
            return

        try:
            path_A, path_B, path_C = self._path_publish_paths
            self.publish_paths_and_labels(path_A, path_B, path_C)
        except Exception as exc:
            self.get_logger().warn(f"Path publish failed: {exc}")

    def publish_initial_pose(self, x: float, y: float, yaw_rad: float):
        """/initialpose トピックを発行"""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame_id
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        _, _, qz, qw = yaw_to_quat(yaw_rad)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance = [0.0] * 36
        self._initpose_pub.publish(msg)


# ==========================================
# GUI Class (Tkinter)
# ==========================================

class AppGUI:
    """Tkinterベースの操作盤"""

    FONT_L = ("Arial", 20)
    FONT_L_BOLD = ("Arial", 20, "bold")

    def __init__(self, node: FollowPathClient, paths_dict: dict):
        self.node = node
        self.paths_dict = paths_dict
        self.root = tk.Tk()
        self._setup_window()
        self._create_widgets()

    def _setup_window(self):
        self.root.title("FollowPath GUI (Nav2)")
        self.root.geometry("800x450")
        self.root.option_add("*Font", self.FONT_L)

    def _create_widgets(self):
        frm_ctrl = tk.Frame(self.root)
        frm_ctrl.pack(pady=15)

        tk.Label(frm_ctrl, text="Controller:").pack(side=tk.LEFT, padx=5)

        self.controller_var = tk.StringVar(value="PP")
        controllers = ["PP", "APP", "RPP", "DWPP"]
        cb = ttk.Combobox(frm_ctrl, textvariable=self.controller_var,
                          values=controllers, state="readonly", width=10,
                          font=self.FONT_L_BOLD)
        cb.pack(side=tk.LEFT, padx=5)

        frm_btns = tk.Frame(self.root)
        frm_btns.pack(pady=10)

        tk.Button(frm_btns, text="Update Start Origin", font=self.FONT_L_BOLD,
                  command=self._on_update_start_origin).pack(side=tk.LEFT, padx=10)
        tk.Button(frm_btns, text="Clear Trajectory", font=self.FONT_L_BOLD,
                  command=self._on_clear_traj).pack(side=tk.LEFT, padx=10)

        tk.Label(self.root, text="Select Path to Start").pack(pady=(20, 5))
        frm_paths = tk.Frame(self.root)
        frm_paths.pack()

        keys = list(self.paths_dict.keys())
        for i, name in enumerate(keys):
            tk.Button(frm_paths, text=name, width=20, font=self.FONT_L_BOLD,
                      command=lambda n=name: self._on_send(n)).grid(
                          row=i // 2, column=i % 2, padx=10, pady=10)

        tk.Button(self.root, text="STOP / CANCEL", font=self.FONT_L_BOLD,
                  fg="red", command=self._on_cancel).pack(pady=20)

    def _on_clear_traj(self):
        self.node.clear_trajectory()

    def _on_update_start_origin(self):
        self.node._update_start_origin_from_current_map_pose()
        if self.node.start_origin_pose_map is None:
            messagebox.showerror("Error", "Failed to update start origin (TF unavailable).")
        else:
            messagebox.showinfo("Info", "Start origin updated to current robot pose (map).")

    def _on_send(self, path_name: str):
        controller_id = self.controller_var.get()
        goal_checker = "general_goal_checker"

        self.node.get_logger().info(f"UI: Send '{path_name}' with '{controller_id}'")
        try:
            path_msg = self.paths_dict[path_name]
            self.node.send_path(path_msg, path_name, controller_id, goal_checker)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_cancel(self):
        self.node.cancel_current_goal()

    def run(self):
        self.root.mainloop()


# ==========================================
# Main Entry Point
# ==========================================

def main():
    rclpy.init()

    # パス生成（map基準）
    frame_id = "map"
    path_A, path_B, path_C = make_path(frame_id)
    # path_A, path_B, path_C = make_iso_path(frame_id)

    paths_registry = {
        "Path A (45 deg)": path_A,
        "Path B (90 deg)": path_B,
        "Path C (135 deg)": path_C,
    }

    node = FollowPathClient()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        gui = AppGUI(node, paths_registry)
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
