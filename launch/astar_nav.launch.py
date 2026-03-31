"""
astar_navigation launch file
=============================
Starts:
  1. map_server           — serves the pre-built map YAML
  2. lifecycle_manager    — auto-activates map_server
  3. static_tf map→odom   — makes the 'map' frame visible in TF/RViz2
                            (identity; safe for Gazebo sim or when MCL is not running)
  4. astar_node           — A* planner + pure-pursuit controller
  5. rviz2                — visualise map, path, visited cells

Usage (run AFTER launching Gazebo separately):

  # Terminal 1 — Gazebo + articubot_one
  ros2 launch articubot_one launch_sim.launch.py

  # Terminal 2 — A* navigation (with pre-built map)
  ros2 launch astar_navigation astar_nav.launch.py use_sim_time:=true

Override map at runtime:
  ros2 launch astar_navigation astar_nav.launch.py \\
      map:=/absolute/path/to/other_map.yaml

If you are running MCL alongside, disable the static TF so MCL's
dynamic map→odom correction is used instead:
  ros2 launch astar_navigation astar_nav.launch.py \\
      static_map_tf:=false

Then in RViz2 click the "2D Nav Goal" button to set a goal.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, LifecycleNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg = FindPackageShare('astar_navigation')

    # ── Launch arguments ─────────────────────────────────────────────────────
    map_arg = DeclareLaunchArgument(
        'map',
        default_value=os.path.expanduser(
            '~/robotics/ros2_ws/maps/my_map_save.yaml'),
        description='Absolute path to the saved map YAML file',
    )

    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([pkg, 'config', 'astar_params.yaml']),
        description='Full path to the astar_node parameters YAML file',
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([pkg, 'config', 'astar_nav.rviz']),
        description='Full path to the RViz2 configuration file',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true',
    )

    static_map_tf_arg = DeclareLaunchArgument(
        'static_map_tf',
        default_value='true',
        description=(
            'Publish a static identity transform map→odom. '
            'Set false when MCL/AMCL is already publishing that transform.'
        ),
    )

    use_sim_time  = LaunchConfiguration('use_sim_time')
    static_map_tf = LaunchConfiguration('static_map_tf')

    # ── 1. map_server (lifecycle node) ───────────────────────────────────────
    map_server = LifecycleNode(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace='',
        output='screen',
        parameters=[{
            'yaml_filename': LaunchConfiguration('map'),
            'frame_id': 'map',
            'use_sim_time': use_sim_time,
        }],
    )

    # ── 2. lifecycle_manager ─────────────────────────────────────────────────
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server'],
        }],
    )

    # ── 3. Static TF: map → odom (identity) ─────────────────────────────────
    # map_server only publishes the /map OccupancyGrid topic — it does NOT
    # broadcast any TF.  Without a map→odom transform the 'map' frame does not
    # exist in the TF tree, so RViz2 and astar_node's TF lookups both fail.
    #
    # This static publisher bridges map→odom as an identity transform, which is
    # correct when:
    #   • running in Gazebo with no accumulated odometry drift (perfect sim), or
    #   • testing A* planning in isolation before wiring up MCL.
    #
    # Disable with static_map_tf:=false when MCL is publishing map→odom.
    static_map_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(static_map_tf),
        output='screen',
    )

    # ── 4. astar_node ────────────────────────────────────────────────────────
    astar_node = Node(
        package='astar_navigation',
        executable='astar_node',
        name='astar_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'use_sim_time': use_sim_time},
        ],
    )

    # ── 5. RViz2 ─────────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    return LaunchDescription([
        map_arg,
        params_arg,
        rviz_arg,
        use_sim_time_arg,
        static_map_tf_arg,
        map_server,
        lifecycle_manager,
        static_map_tf_node,
        astar_node,
        rviz_node,
    ])
