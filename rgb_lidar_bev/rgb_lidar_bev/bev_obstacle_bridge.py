"""
Bridge BEV sector proximity output into planner-friendly LaserScan topics.

Input:
  - /bev/sector_proximity (ranges + class IDs encoded in intensities)

Outputs:
  - /bev/obstacle_scan: all non-clear sectors as obstacles
  - /bev/human_obstacle_scan: optional human-only obstacles
"""
from __future__ import annotations

import math
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class BevObstacleBridge(Node):
    def __init__(self) -> None:
        super().__init__("bev_obstacle_bridge")

        self.declare_parameter("input_topic", "/bev/sector_proximity")
        self.declare_parameter("output_topic", "/bev/obstacle_scan")
        self.declare_parameter("publish_human_scan", True)
        self.declare_parameter("human_output_topic", "/bev/human_obstacle_scan")
        self.declare_parameter("clear_class_id", 0)
        self.declare_parameter("human_class_id", 2)
        self.declare_parameter("use_inf_for_clear", True)
        self.declare_parameter("default_unknown_class_id", 1)
        self.declare_parameter("static_inflation_m", 0.10)
        self.declare_parameter("human_inflation_m", 0.25)

        input_topic = self._get_str_param("input_topic", "/bev/sector_proximity")
        output_topic = self._get_str_param("output_topic", "/bev/obstacle_scan")
        self._publish_human_scan = self._get_bool_param("publish_human_scan", True)
        human_output_topic = self._get_str_param("human_output_topic", "/bev/human_obstacle_scan")
        self._clear_class_id = self._get_int_param("clear_class_id", 0)
        self._human_class_id = self._get_int_param("human_class_id", 2)
        self._use_inf_for_clear = self._get_bool_param("use_inf_for_clear", True)
        self._default_unknown_class_id = self._get_int_param("default_unknown_class_id", 1)
        self._static_inflation_m = self._get_float_param("static_inflation_m", 0.10)
        self._human_inflation_m = self._get_float_param("human_inflation_m", 0.25)

        self._obstacle_pub = self.create_publisher(LaserScan, output_topic, 10)
        self._human_pub = (
            self.create_publisher(LaserScan, human_output_topic, 10)
            if self._publish_human_scan
            else None
        )
        self.create_subscription(LaserScan, input_topic, self._sector_cb, 10)

        self.get_logger().info(
            "Bridge ready: %s -> %s%s"
            % (
                input_topic,
                output_topic,
                f", {human_output_topic}" if self._publish_human_scan else "",
            )
        )

    def _sector_cb(self, msg: LaserScan) -> None:
        out_scan = self._clone_scan_meta(msg)
        out_scan.ranges = self._to_obstacle_ranges(msg, human_only=False)
        self._obstacle_pub.publish(out_scan)

        if self._human_pub is not None:
            human_scan = self._clone_scan_meta(msg)
            human_scan.ranges = self._to_obstacle_ranges(msg, human_only=True)
            self._human_pub.publish(human_scan)

    def _to_obstacle_ranges(self, msg: LaserScan, human_only: bool) -> List[float]:
        out_ranges: List[float] = []
        range_min = float(msg.range_min)
        range_max = float(msg.range_max) if msg.range_max > 0.0 else 100.0

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or r <= 0.0:
                out_ranges.append(math.inf if self._use_inf_for_clear else range_max)
                continue

            class_id = self._default_unknown_class_id
            if i < len(msg.intensities):
                class_id = int(round(msg.intensities[i]))

            is_human = class_id == self._human_class_id
            is_clear = class_id == self._clear_class_id
            is_obstacle = (not is_clear) and (is_human if human_only else True)

            if not is_obstacle:
                out_ranges.append(math.inf if self._use_inf_for_clear else range_max)
                continue

            inflation = self._human_inflation_m if is_human else self._static_inflation_m
            obstacle_range = max(range_min, r - max(0.0, inflation))
            out_ranges.append(min(obstacle_range, range_max))

        return out_ranges

    @staticmethod
    def _clone_scan_meta(msg: LaserScan) -> LaserScan:
        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        return out

    def _get_str_param(self, name: str, default: str) -> str:
        value = self.get_parameter(name).value
        return default if value is None else str(value)

    def _get_bool_param(self, name: str, default: bool) -> bool:
        value = self.get_parameter(name).value
        return default if value is None else bool(value)

    def _get_int_param(self, name: str, default: int) -> int:
        value = self.get_parameter(name).value
        return default if value is None else int(value)

    def _get_float_param(self, name: str, default: float) -> float:
        value = self.get_parameter(name).value
        return default if value is None else float(value)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BevObstacleBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
