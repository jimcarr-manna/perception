"""Run BevNode and NavNode together in one process."""
from __future__ import annotations

import cv2
import rclpy
from rclpy.executors import MultiThreadedExecutor

from rgb_lidar_bev.nav_node import NavNode
from rgb_lidar_bev.bev_node import BevNode

def main(args=None) -> None:
    rclpy.init(args=args)
    bev = BevNode()
    nav = NavNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(bev)
    executor.add_node(nav)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        nav._publish_stop()
        if bev._sector_viz is not None:
            bev._sector_viz.close()
        executor.shutdown()
        nav.destroy_node()
        bev.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
