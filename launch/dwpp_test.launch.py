#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    Command,
    TextSubstitution,
    PathJoinSubstitution,
    FindExecutable,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ====== パス解決（Rolling推奨の書き方） ======
    pkg_this      = FindPackageShare('dwpp_test_simulation')
    pkg_nav2      = FindPackageShare('nav2_bringup')
    pkg_tb3_desc  = FindPackageShare('turtlebot3_description')
    pkg_ros_gz    = FindPackageShare('ros_gz_sim')

    default_params_file = PathJoinSubstitution([pkg_this,     'config', 'test_params.yaml'])
    default_world_file  = PathJoinSubstitution([pkg_this,     'world',  'test.world'])     # SDF/.world を想定
    default_robot_sdf   = PathJoinSubstitution([pkg_this,     'model',  'tb3_wo_odom.sdf'])
    default_rviz_config = PathJoinSubstitution([pkg_this,     'rviz',   'dwpp_test.rviz'])
    default_urdf_xacro  = PathJoinSubstitution([pkg_tb3_desc, 'urdf',   'turtlebot3_waffle.urdf'])

    # ====== Launch 引数 ======
    params_file  = LaunchConfiguration('params_file')
    world_file   = LaunchConfiguration('world_file')
    robot_sdf    = LaunchConfiguration('robot_sdf')
    urdf_xacro   = LaunchConfiguration('urdf_xacro')
    tf_namespace = LaunchConfiguration('tf_namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz_config  = LaunchConfiguration('rviz_config')
    use_rviz     = LaunchConfiguration('use_rviz')
    robot_entity = LaunchConfiguration('robot_entity')

    declare_params   = DeclareLaunchArgument('params_file',   default_value=default_params_file, description='Nav2 parameters YAML')
    declare_world    = DeclareLaunchArgument('world_file',    default_value=default_world_file,  description='Gazebo Sim world (ros_gz)')
    declare_robot    = DeclareLaunchArgument('robot_sdf',     default_value=default_robot_sdf,   description='Robot SDF file to spawn')
    declare_urdfx    = DeclareLaunchArgument('urdf_xacro',    default_value=default_urdf_xacro,  description='TB3 xacro for RSP')
    declare_ns       = DeclareLaunchArgument('tf_namespace',  default_value=TextSubstitution(text=''), description='TF/xacro namespace (empty for none)')
    declare_use_sim  = DeclareLaunchArgument('use_sim_time',  default_value='True', description='Use /clock from Gazebo Sim')
    declare_rvizcf   = DeclareLaunchArgument('rviz_config',   default_value=default_rviz_config, description='RViz2 config file')
    declare_use_rv   = DeclareLaunchArgument('use_rviz',      default_value='True', description='Launch RViz2')
    declare_entity   = DeclareLaunchArgument('robot_entity',  default_value='tb3', description='Gazebo entity name')

    # ====== TB3 環境変数 ======
    set_tb3_model = SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle')

    # ====== Gazebo Sim 起動（ros_gz_sim） ======
    # -r: run（headless可） / -v 4: verbose、必要に応じて変更
    gz_launch = PathJoinSubstitution([pkg_ros_gz, 'launch', 'gz_sim.launch.py'])
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch),
        launch_arguments={
            # world を引数として渡す（-r -v を同時指定）
            'gz_args': [TextSubstitution(text='-r -v 3 '), world_file]
        }.items()
    )

    # ====== Gazebo Sim に TB3（SDF）をスポーン ======
    # ros_gz_sim の create を用い、SDFファイルを直接投入
    spawn_tb3 = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-file', robot_sdf,
            '-name', robot_entity
            # 必要なら `-allow_renaming true` や `-x/-y/-z` を追加
        ],
        output='screen'
    )

    # ====== map -> odom を恒等に固定 ======
    static_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        name='map_to_odom_static_tf'
    )

    # ====== 真値 → /odom ＆ TF(odom->base_link) ======
    # 既存ノードが Gazebo Classic の /gazebo/* に依存している場合は、ros_gz_bridge 側のトピック名に合わせた実装が必要です。
    ground_truth_odom = Node(
        package='dwpp_test_simulation',
        executable='ground_truth_odom_tf.py',
        name='ground_truth_odom_tf',
        output='screen',
        parameters=[{
            'model_name': robot_entity,
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'use_sim_time': use_sim_time
        }]
    )

    # ====== GUI（任意ツール） ======
    gui_node = Node(
        package='dwpp_test_simulation',
        executable='follow_path_test_gui.py',
        name='follow_path_gui',
        output='screen',
        emulate_tty=True,
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ====== robot_state_publisher（xacro 展開） ======
    robot_description = Command([
        FindExecutable(name='xacro'), ' ',
        urdf_xacro,
        ' namespace:=', tf_namespace,
        ' use_sim_time:=', use_sim_time
    ])

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time
        }]
    )

    # ====== Nav2 ======
    nav2_nav_launch = PathJoinSubstitution([pkg_nav2, 'launch', 'navigation_launch.py'])
    nav2_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_nav_launch),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file
        }.items()
    )

    # ====== RViz2 ======
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        condition=IfCondition(use_rviz)
    )

    return LaunchDescription([
        declare_params, declare_world, declare_robot, declare_urdfx, declare_ns,
        declare_use_sim, declare_rvizcf, declare_use_rv, declare_entity,
        set_tb3_model,
        gazebo,
        spawn_tb3,
        static_map_to_odom,
        ground_truth_odom,
        robot_state_publisher,
        nav2_navigation,
        rviz,
        gui_node,
    ])
