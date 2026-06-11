from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    use_sim_time = LaunchConfiguration("use_sim_time")
    nav2_params_file = LaunchConfiguration("nav2_params_file")
    map_yaml_file = LaunchConfiguration("map_yaml_file")
    autostart = LaunchConfiguration("autostart")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument(
                "nav2_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("rgb_lidar_bev"), "config", "nav2_params.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "map_yaml_file",
                default_value="/home/ubuntu/ros2_ws/src/slam/maps/map_01.yaml",
            ),
            Node(
                package="rgb_lidar_bev",
                executable="bev_node",
                name="bev_node",
                output="screen",
                parameters=[
                    {
                        "show": False,
                        "sector_viz": "ascii",
                        "lidar_topic": "/scan",
                        "sector_topic": "/bev/sector_proximity",
                    }
                ],
            ),
            Node(
                package="rgb_lidar_bev",
                executable="bev_obstacle_bridge",
                name="bev_obstacle_bridge",
                output="screen",
                parameters=[
                    {
                        "input_topic": "/bev/sector_proximity",
                        "output_topic": "/bev/obstacle_scan",
                        "publish_human_scan": True,
                        "human_output_topic": "/bev/human_obstacle_scan",
                    }
                ],
            ),
            Node(
                package="rgb_lidar_bev",
                executable="joy_estop_gate",
                name="joy_estop_gate",
                output="screen",
                parameters=[
                    {
                        "joy_topic": "/ros_robot_controller/joy",
                        "input_cmd_vel_topic": "/cmd_vel_nav",
                        "output_cmd_vel_topic": "/cmd_vel",
                        "btn_drive_index": 4,  # Y
                        "btn_estop_index": 3,  # X
                        "require_enable_button": True,
                    }
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("nav2_bringup"), "launch", "localization_launch.py"]
                    )
                ),
                launch_arguments={
                    "map": map_yaml_file,
                    "use_sim_time": use_sim_time,
                    "params_file": nav2_params_file,
                    "autostart": autostart,
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("nav2_bringup"), "launch", "navigation_launch.py"]
                    )
                ),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "params_file": nav2_params_file,
                    "autostart": autostart,
                }.items(),
            ),
        ]
    )
