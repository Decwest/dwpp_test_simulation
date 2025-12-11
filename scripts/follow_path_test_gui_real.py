#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nav2 FollowPath Test GUI Client
機能:
  - 規定のパス生成とNav2への送信 (ActionClient)
  - 現在のロボット位置を基準としたパス座標変換
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
import copy  # オブジェクトのディープコピー用

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
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    qos_profile_sensor_data
)

# --- ROS 2 Messages ---
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, TransformStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import BatteryState, Imu
from nav2_msgs.action import FollowPath
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool

# --- TF2 Imports ---
from tf2_ros import TransformException, TransformBroadcaster
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
    # 生成するカーブの角度パターン
    theta_list = [np.pi/4, np.pi/2, 3*np.pi/4]
    l_segment = 3.0  # 直線区間の長さパラメータ

    for theta in theta_list:
        # 1. 直進 (0 -> 1m)
        x1 = np.linspace(0, 1, 100)
        y1 = np.zeros_like(x1)
        
        # 2. 斜め直線 (角度thetaで長さl)
        # Note: 実際には直線補間だが、ここでは簡易的に生成
        x2 = np.linspace(1.0, 1.0 + l_segment * math.cos(theta), 100)
        y2 = np.linspace(0.0, l_segment * math.sin(theta), 100)
        
        # 3. 終端直進 (さらに3m進む)
        x3 = np.linspace(
            1.0 + l_segment * math.cos(theta), 
            4.0 + l_segment * math.cos(theta), 
            100
        )
        y3 = np.ones_like(x3) * l_segment * math.sin(theta)

        # 結合
        xs = np.concatenate([x1, x2, x3])
        ys = np.concatenate([y1, y2, y3])
        
        # 方位角(Yaw)の計算
        dx = np.gradient(xs)
        dy = np.gradient(ys)
        yaws = np.unwrap(np.arctan2(dy, dx))

        # Pathメッセージの作成
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

    # 展開して返す
    return (paths[0], paths[1], paths[2])


# ==========================================
# Main ROS 2 Node Class
# ==========================================

class FollowPathClient(Node):
    """
    Nav2のFollowPathアクションを呼び出し、実験データを記録・可視化するノード
    """
    def __init__(self, path_frame_id: str = "start_pose"):
        super().__init__('follow_path_gui_client')
        
        # --- Parameters ---
        self.path_frame_id = path_frame_id
        self.record_frequency = self.declare_parameter('record_frequency', 30).value
        self.data_dir = self.declare_parameter('data_dir', '/tmp').value
        self.map_frame_id = self.declare_parameter('map_frame_id', 'map').value
        self.base_frame_id = self.declare_parameter('base_frame_id', 'base_footprint').value
        self.experiment_name = self.declare_parameter('experiment_name', 'dwpp').value  # dwpp or nelson

        # --- Internal State Variables ---
        self._current_goal_handle = None
        self.path_frame_origin_pose = None
        self.path_frame_origin_orientation = None
        self._recording = False
        self._active_traj = None  # 現在実行中の制御手法名(PP, APP等)
        self._traj_lock = threading.Lock()

        # データ記録用バッファ
        self._reset_record_buffer()

        # 軌跡描画用の点列バッファ
        self._traj_points = {'PP': [], 'APP': [], 'RPP': [], 'DWPP': []}
        self._traj_colors = {
            'PP':   (1.0, 0.0, 0.0),   # red
            'APP':  (0.0, 0.7, 0.2),   # green
            'RPP':  (0.0, 0.4, 1.0),   # blue
            'DWPP': (0.8, 0.2, 0.8)    # purple
        }

        # 受信データキャッシュ
        self.current_odom = None
        self.current_cmd_vel_nav = None  # Control Server出力
        self.current_cmd_vel = None      # Nav2最終出力
        self.current_velocity_violation = False
        self.battery_voltage = float('nan')
        self.battery_current = float('nan')
        self.imu_angular_vel_x = float('nan')
        self.imu_angular_vel_y = float('nan')
        self.imu_angular_vel_z = float('nan')
        self.imu_linear_acc_x = float('nan')
        self.imu_linear_acc_y = float('nan')
        self.imu_linear_acc_z = float('nan')

        # --- QoS Settings ---
        # RVizで「後からSubscribeしても見える」ようにするための設定
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # --- TF Components ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- Action Client ---
        self._client = ActionClient(self, FollowPath, '/follow_path')

        # --- Publishers ---
        self._initpose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        self._label_pub = self.create_publisher(MarkerArray, '/viz/path_labels', latched_qos)
        self._path_markers_pub = self.create_publisher(MarkerArray, '/viz/path_markers', latched_qos)
        self._traj_pub = self.create_publisher(MarkerArray, '/viz/robot_trajs', 10)

        # --- Subscribers ---
        self._odom_sub = self.create_subscription(
            Odometry, '/odom', self._on_odom, qos_profile_sensor_data)
        self._cmd_vel_nav_sub = self.create_subscription(
            Twist, '/cmd_vel_nav', self._cmd_vel_nav_callback, 1)
        self._cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_callback, 1)
        self._velocity_violation_sub = self.create_subscription(
            Bool, '/constraints_violation_flag', self._velocity_violation_callback, 1)
        self.battery_state_sub = self.create_subscription(
            BatteryState, '/whill/states/battery_state', self._battery_state_callback, 1)
        self.imu_sub = self.create_subscription(
            Imu, '/ouster/imu', self._imu_callback, 1)

        # --- Background Threads & Timers ---
        # 1. 起動時の初期位置合わせ & パス可視化 (別スレッドで実行)
        threading.Thread(target=self._auto_publish_initial_pose, daemon=True).start()
        threading.Thread(target=self._periodic_path_publish, daemon=True).start()

        # 2. データ記録ループ (Timer)
        self.record_rate = self.create_rate(self.record_frequency)
        threading.Thread(target=self._recording_loop, daemon=True).start()

        # 3. Path FrameのTF配信ループ (Timer)
        self.path_frame_rate = self.create_rate(100)  # 100 Hz
        threading.Thread(target=self._broadcast_path_frame_loop, daemon=True).start()

    # =========================================================================
    # Callbacks (Subscribers)
    # =========================================================================

    def _imu_callback(self, msg: Imu):
        # header = msg.header.frame_id
        # print(header)
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
            "sec": [], "nsec": [], "x": [], "y": [], "yaw": [], 
            "v_cmd": [], "w_cmd": [], 
            "battery_v": [], "battery_i": [],
            "v_real": [], "w_real": [], 
            "imu_ax": [], "imu_ay": [], "imu_az": [],
            "imu_vx": [], "imu_vy": [], "imu_vz": [],
            "v_nav": [], "w_nav": [], 
            "velocity_violation": []
        }

    def _recording_loop(self):
        """
        データ記録用スレッド
        _recordingフラグがTrueの間、状態をリストに追加し、FalseになったらCSV保存する
        """
        while rclpy.ok():
            # 必要なデータが揃うまでスキップ
            if (self.current_cmd_vel_nav is None or 
                self.current_cmd_vel is None or 
                self.current_odom is None):
                self.record_rate.sleep()
                continue
            
            if self._recording:
                # --- データ取得 ---
                # 現在位置 (path_frame基準)
                current_pose, current_orientation = self._get_robot_pose(from_frame=self.path_frame_id)
                if current_pose is None:
                    continue

                # RViz軌跡更新
                self._draw_robot_trajectory(current_pose, self.path_frame_id)
                
                # 姿勢(Yaw)計算
                r = R.from_quat([
                    current_orientation.x, current_orientation.y, 
                    current_orientation.z, current_orientation.w
                ])
                yaw = r.as_euler('xyz')[2]
                
                # 時間
                sec = self.get_clock().now().nanoseconds * 1e-9
                nsec = self.get_clock().now().nanoseconds % 1e9

                # --- バッファへ追加 ---
                d = self.record_state_dict
                d["sec"].append(sec)
                d["nsec"].append(nsec)
                d["x"].append(current_pose.x)
                d["y"].append(current_pose.y)
                d["yaw"].append(yaw)
                # 実測値
                d["v_real"].append(self.current_odom.twist.twist.linear.x)
                d["w_real"].append(self.current_odom.twist.twist.angular.z)
                # 指令値 (Control Server)
                d["v_cmd"].append(self.current_cmd_vel_nav.linear.x)
                d["w_cmd"].append(self.current_cmd_vel_nav.angular.z)
                # 指令値 (Nav2 Final)
                d["v_nav"].append(self.current_cmd_vel.linear.x)
                d["w_nav"].append(self.current_cmd_vel.angular.z)
                # フラグ
                d["velocity_violation"].append(self.current_velocity_violation)
                # バッテリーデータ
                d["battery_v"].append(self.battery_voltage)
                d["battery_i"].append(self.battery_current)
                # IMUデータ
                d["imu_ax"].append(self.imu_linear_acc_x)
                d["imu_ay"].append(self.imu_linear_acc_y)
                d["imu_az"].append(self.imu_linear_acc_z)
                d["imu_vx"].append(self.imu_angular_vel_x)
                d["imu_vy"].append(self.imu_angular_vel_y)
                d["imu_vz"].append(self.imu_angular_vel_z)
                
            else:
                # 記録停止中かつバッファにデータがある場合 -> CSV保存
                if len(self.record_state_dict["x"]) > 0:
                    self._save_to_csv()
                    self._reset_record_buffer()
                
            self.record_rate.sleep()

    def _save_to_csv(self):
        """記録したバッファをCSVファイルに書き出す"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{self.data_dir}/{self.experiment_name}"
        filename = f"{dir_name}/{self._active_traj}_{timestamp}.csv"
        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = list(self.record_state_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            data_len = len(self.record_state_dict["x"])
            for i in range(data_len):
                row = {field: self.record_state_dict[field][i] for field in fieldnames}
                writer.writerow(row)
                
        self.get_logger().info(f"Saved trajectory data to '{filename}'")

    # =========================================================================
    # Action Client Logic (FollowPath)
    # =========================================================================

    def send_path(self, path_msg: Path, controller_id: str, goal_checker_id: str):
        """指定されたパスをNav2に送信する"""
        if not self._client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Action server `/follow_path` not available.")
            return

        # 1. パスの基準座標系を更新（現在のロボット位置を原点とする）
        self._update_path_frame_origin()
        time.sleep(0.5)  # TF反映待ち
        
        # 2. 座標変換 (path_frame -> map_frame)
        # 注意: 元のpath_msgを変更しないようにdeepcopyする
        transformed_path = copy.deepcopy(path_msg)
        
        try:
            # path_frame -> map_frame の変換を取得
            tf_stamped = self.tf_buffer.lookup_transform(
                self.map_frame_id,
                path_msg.header.frame_id,
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().error(f"Transform error: {ex}")
            return

        # パス内の全点を変換
        t_vec = tf_stamped.transform.translation
        r_quat = tf_stamped.transform.rotation
        r_tf = R.from_quat([r_quat.x, r_quat.y, r_quat.z, r_quat.w])
        offset = np.array([t_vec.x, t_vec.y, t_vec.z])

        # Header Frame ID を Map に変更
        transformed_path.header.frame_id = self.map_frame_id

        for pose_stamped in transformed_path.poses:
            # 位置の変換
            p = pose_stamped.pose.position
            p_vec = np.array([p.x, p.y, p.z])
            p_new = r_tf.apply(p_vec) + offset
            
            pose_stamped.pose.position.x = p_new[0]
            pose_stamped.pose.position.y = p_new[1]
            pose_stamped.pose.position.z = p_new[2]

            # 向きの変換
            q = pose_stamped.pose.orientation
            r_pose = R.from_quat([q.x, q.y, q.z, q.w])
            r_new = r_tf * r_pose  # 回転の合成
            q_new = r_new.as_quat()

            pose_stamped.pose.orientation.x = q_new[0]
            pose_stamped.pose.orientation.y = q_new[1]
            pose_stamped.pose.orientation.z = q_new[2]
            pose_stamped.pose.orientation.w = q_new[3]

        # 3. ゴールの作成と送信
        goal = FollowPath.Goal()
        goal.path = transformed_path
        goal.controller_id = controller_id
        goal.goal_checker_id = goal_checker_id

        # 記録開始
        with self._traj_lock:
            self._active_traj = controller_id
            self._traj_points[controller_id] = [] # 軌跡リセット
            self._recording = True

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
        """ゴール到達（または失敗・キャンセル）時の処理"""
        with self._traj_lock:
            self._recording = False
        
        try:
            result = future.result()
            self.get_logger().info(f'Result: status={result.status}')
        except Exception as e:
            self.get_logger().error(f'Result callback failed: {e}')

    def cancel_current_goal(self):
        """実行中のゴールをキャンセル"""
        if self._current_goal_handle is None:
            self.get_logger().info('No active goal to cancel.')
            return
        
        self.get_logger().info('Canceling goal...')
        self._current_goal_handle.cancel_goal_async()
        with self._traj_lock:
            self._recording = False

    def _feedback_cb(self, feedback_msg):
        # ログが多すぎる場合はコメントアウト推奨
        # self.get_logger().debug(f'Feedback received...')
        pass

    # =========================================================================
    # Visualization & TF Logic
    # =========================================================================

    def _draw_robot_trajectory(self, current_pos, frame_id):
        """ロボットの軌跡をMarkerArrayとしてPublish"""
        if self._active_traj not in self._traj_points:
            return
        
        pts = self._traj_points[self._active_traj]
        
        # 点の間引き (前回から5cm以上移動していたら追加)
        if not pts or self._distance_2d(pts[-1], current_pos) > 0.05:
            pts.append(current_pos)
            # メモリ節約のため上限を設定
            if len(pts) > 5000:
                self._traj_points[self._active_traj] = pts[-2000:]

        # マーカー作成
        marr = MarkerArray()
        now = self.get_clock().now().to_msg()
        mid = 0
        
        for name, points in self._traj_points.items():
            if len(points) < 2:
                continue
                
            r, g, b = self._traj_colors.get(name, (0.5, 0.5, 0.5))
            
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = now
            m.ns = 'robot_trajectory'
            m.id = mid
            mid += 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.035
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.95
            
            # pointsのコピーを渡す
            m.points = copy.deepcopy(points)
            marr.markers.append(m)

        self._traj_pub.publish(marr)

    def publish_paths_and_labels(self, path_A: Path, path_B: Path, path_C: Path):
        """規定パスとラベルをRVizに表示"""
        markers = MarkerArray()
        path_configs = [
            (path_A, (1.0, 0.0, 0.0), 'Path A'),
            (path_B, (0.0, 1.0, 0.0), 'Path B'),
            (path_C, (0.0, 0.0, 1.0), 'Path C')
        ]
        now = self.get_clock().now().to_msg()

        # Lines
        for mid, (path, color, _) in enumerate(path_configs):
            if not path.poses: continue
            
            m = Marker()
            m.header.frame_id = path.header.frame_id
            m.header.stamp = now
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

        # Labels
        labels = MarkerArray()
        for mid, (path, _, name) in enumerate(path_configs, start=100):
            m = Marker()
            m.header.frame_id = path.header.frame_id
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
        """描画済み軌跡の消去"""
        with self._traj_lock:
            for k in self._traj_points:
                self._traj_points[k] = []

        # DELETEALL マーカーを送信
        ma = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        ma.markers.append(m)
        self._traj_pub.publish(ma)
        self.get_logger().info("Cleared all robot trajectories.")

    # =========================================================================
    # Transform Helpers & Utilities
    # =========================================================================

    def _update_path_frame_origin(self):
        """現在のロボット位置を取得し、Path Frameの原点として更新する"""
        current_pose, current_orientation = self._get_robot_pose(from_frame=self.map_frame_id)
        if current_pose is not None:
            self.path_frame_origin_pose = current_pose
            self.path_frame_origin_orientation = current_orientation
            # self.get_logger().info("Path frame origin updated.")

    def _broadcast_path_frame_loop(self):
        """Path FrameをTFツリーに配信し続ける"""
        while rclpy.ok():
            if (self.path_frame_origin_pose is not None and 
                self.path_frame_origin_orientation is not None):
                
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = self.map_frame_id
                t.child_frame_id = self.path_frame_id
                t.transform.translation = self.path_frame_origin_pose
                t.transform.rotation = self.path_frame_origin_orientation
                self.tf_broadcaster.sendTransform(t)
            
            self.path_frame_rate.sleep()

    def _get_robot_pose(self, from_frame: str):
        """指定フレームから見たロボット(base_link)の位置姿勢を取得"""
        try:
            # 最新のTransformを取得
            tf = self.tf_buffer.lookup_transform(
                from_frame,
                self.base_frame_id,
                rclpy.time.Time()
            )
            return tf.transform.translation, tf.transform.rotation
        except TransformException as ex:
            # 頻繁に出るとログが汚れるため Warn レベルで抑制しても良い
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return None, None

    def _distance_2d(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    # =========================================================================
    # Periodic Tasks
    # =========================================================================

    def _auto_publish_initial_pose(self):
        """起動直後に初期位置(0,0,0)をPublishする（RViz初期化用）"""
        time.sleep(1.0) # システム安定待ち
        for _ in range(3):
            self.publish_initial_pose(0.0, 0.0, 0.0)
            self._update_path_frame_origin()
            time.sleep(0.5)

    def _periodic_path_publish(self):
        """定期的にパスをRVizにPublishする（再接続時対策）"""
        time.sleep(2.0)
        # Note: path生成は軽いのでここで都度呼んでも良いし、キャッシュしても良い
        path_A, path_B, path_C = make_path(self.path_frame_id)
        while rclpy.ok():
            try:
                self.publish_paths_and_labels(path_A, path_B, path_C)
                time.sleep(0.5)
            except Exception:
                time.sleep(1.0)

    def publish_initial_pose(self, x: float, y: float, yaw_rad: float):
        """/initialpose トピックを発行（Nav2のリセット等に使用）"""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame_id
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        _, _, qz, qw = yaw_to_quat(yaw_rad)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        
        # 共分散行列（対角成分のみ設定など、必要に応じて）
        msg.pose.covariance = [0.0] * 36
        self._initpose_pub.publish(msg)


# ==========================================
# GUI Class (Tkinter)
# ==========================================

class AppGUI:
    """Tkinterベースの操作盤"""
    
    # Constants for Styles
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
        # 1. Controller Selection
        frm_ctrl = tk.Frame(self.root)
        frm_ctrl.pack(pady=15)
        
        tk.Label(frm_ctrl, text="Controller:").pack(side=tk.LEFT, padx=5)
        
        self.controller_var = tk.StringVar(value="PP")
        controllers = ["PP", "APP", "RPP", "DWPP"]
        cb = ttk.Combobox(frm_ctrl, textvariable=self.controller_var, 
                          values=controllers, state="readonly", width=10, 
                          font=self.FONT_L_BOLD)
        cb.pack(side=tk.LEFT, padx=5)

        # 2. Control Buttons (Update / Clear)
        frm_btns = tk.Frame(self.root)
        frm_btns.pack(pady=10)
        
        tk.Button(frm_btns, text="Update Path Origin", font=self.FONT_L_BOLD, 
                  command=self._on_update_path_origin).pack(side=tk.LEFT, padx=10)
        tk.Button(frm_btns, text="Clear Trajectory", font=self.FONT_L_BOLD, 
                  command=self._on_clear_traj).pack(side=tk.LEFT, padx=10)

        # 3. Path Selection Buttons
        tk.Label(self.root, text="Select Path to Start").pack(pady=(20, 5))
        frm_paths = tk.Frame(self.root)
        frm_paths.pack()

        # Grid layout for path buttons
        keys = list(self.paths_dict.keys())
        for i, name in enumerate(keys):
            tk.Button(frm_paths, text=name, width=20, font=self.FONT_L_BOLD,
                      command=lambda n=name: self._on_send(n)).grid(
                          row=i // 2, column=i % 2, padx=10, pady=10)

        # 4. Cancel Button
        tk.Button(self.root, text="STOP / CANCEL", font=self.FONT_L_BOLD, 
                  fg="red", command=self._on_cancel).pack(pady=20)

    # --- GUI Events ---
    def _on_clear_traj(self):
        self.node.clear_trajectory()

    def _on_update_path_origin(self):
        self.node._update_path_frame_origin()
        messagebox.showinfo("Info", "Path origin updated to current robot pose.")

    def _on_send(self, path_name: str):
        ctrl_id = self.controller_var.get()
        # Goal Checker IDは必要に応じて変更可能
        goal_checker = "general_goal_checker"
        
        self.node.get_logger().info(f"UI: Send '{path_name}' with '{ctrl_id}'")
        try:
            path_msg = self.paths_dict[path_name]
            self.node.send_path(path_msg, ctrl_id, goal_checker)
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

    # パス生成 (基準フレーム: start_pose)
    # 実行時に tf によって map座標系へ変換されるため、ここではローカル定義でOK
    frame_id = "start_pose"
    path_A, path_B, path_C = make_path(frame_id)

    paths_registry = {
        "Path A (45 deg)": path_A,
        "Path B (90 deg)": path_B,
        "Path C (135 deg)": path_C,
    }

    # ノード作成
    node = FollowPathClient(path_frame_id=frame_id)

    # Executorの設定 (ROSコールバックとGUIの共存のためMultiThreaded)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    # ROSスレッドの開始
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        # GUI起動 (Main Thread)
        gui = AppGUI(node, paths_registry)
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()