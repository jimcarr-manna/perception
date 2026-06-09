from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="rgb_lidar_bev",
            executable="bev_node",
            name="bev_node",
            output="screen",
            parameters=[{
                "show": False,
                "sector_viz": "ascii",
                "sector_topic": "/bev/sector_proximity",
            }],
        ),
        Node(
            package="rgb_lidar_bev",
            executable="nav_node",
            name="nav_node",
            output="screen",
            parameters=[{
                "joy_topic": "/ros_robot_controller/joy",
                "cmd_vel_topic": "/cmd_vel",
                "sector_topic": "/bev/sector_proximity",
                "btn_drive_index": 4,   # Y
                "btn_estop_index": 3,   # X
            }],
        ),
    ])
