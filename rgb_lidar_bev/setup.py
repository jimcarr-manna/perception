from glob import glob
from setuptools import find_packages, setup

package_name = "rgb_lidar_bev"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/rgb_lidar_bev"]),
        (f"share/rgb_lidar_bev", ["package.xml"]),
        (f"share/rgb_lidar_bev/launch", glob("launch/*.launch.py")),
        (f"share/rgb_lidar_bev/config", glob("config/*.yaml")),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "opencv-python-headless",
        "pyyaml",
        "scipy",
    ],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="YOLO monocular RGB + 2D lidar to BEV and nav stack",
    license="MIT",
    entry_points={
        "console_scripts": [
            "bev_node = rgb_lidar_bev.bev_node:main",
            "nav_node = rgb_lidar_bev.nav_node:main",
            "bev_nav_stack = rgb_lidar_bev.bev_nav_stack:main",
            "bev_obstacle_bridge = rgb_lidar_bev.bev_obstacle_bridge:main",
            "joy_estop_gate = rgb_lidar_bev.joy_estop_gate:main",
            "goal_client = rgb_lidar_bev.goal_client:main",
        ],
    },
)
