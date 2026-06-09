from setuptools import setup, find_packages

package_name = "rgb_lidar_bev"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/rgb_lidar_bev"]),
        (f"share/rgb_lidar_bev", ["package.xml"]),
        (f"share/rgb_lidar_bev/launch", ["launch/bev_nav.launch.py"]),
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
        ],
    },
)
