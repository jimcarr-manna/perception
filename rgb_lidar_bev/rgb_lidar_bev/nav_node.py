"""
ROS 2 navigation node for JetAuto controller-driven straight motion.

Controls:
  - Joy button[4] (Y): enable forward driving and clear manual estop
  - Joy button[3] (X): latch manual estop and stop immediately

Safety:
  - Subscribes to /bev/sector_proximity (LaserScan-encoded sector map)
  - Stops immediately when an obstacle appears in a configurable forward cone
"""
from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy, LaserScan

import time

class NavNode(Node):
    def __init__(self) -> None:
        super().__init__("rgb_lidar_nav")

        self.declare_parameter("joy_topic", "/ros_robot_controller/joy")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("sector_topic", "/bev/sector_proximity")
        self.declare_parameter("forward_speed", 0.20)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("btn_drive_index", 4)   # Y button
        self.declare_parameter("btn_estop_index", 3)   # X button
        self.declare_parameter("front_half_angle_deg", 15.0)
        self.declare_parameter("estop_dist_static_m", 0.45)
        self.declare_parameter("estop_dist_human_m", 0.70)

        joy_topic = str(self.get_parameter("joy_topic").value)
        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        sector_topic = str(self.get_parameter("sector_topic").value)
        self._forward_speed = float(self.get_parameter("forward_speed").value or 0.2)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value or 20.0)
        self._btn_drive = int(self.get_parameter("btn_drive_index").value or 4)
        self._btn_estop = int(self.get_parameter("btn_estop_index").value or 3)
        self._front_half_angle = math.radians(
            float(self.get_parameter("front_half_angle_deg").value or 15.0)
        )
        self._estop_dist_static = float(self.get_parameter("estop_dist_static_m").value or 0.45)
        self._estop_dist_human = float(self.get_parameter("estop_dist_human_m").value or 0.70)

        self._manual_estop = False
        self._forward_enabled = False
        self._obstacle_blocked = False
        self._prev_drive_pressed = False
        self._prev_estop_pressed = False

        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.create_subscription(Joy, joy_topic, self._joy_cb, 10)
        self.create_subscription(LaserScan, sector_topic, self._sector_cb, 10)
        self.create_timer(1.0 / publish_rate_hz, self._tick)

        self.get_logger().info(
            "NavNode ready — joy: %s, cmd_vel: %s, sector: %s, "
            "Y(button[%d])=drive, X(button[%d])=estop"
            % (joy_topic, cmd_vel_topic, sector_topic, self._btn_drive, self._btn_estop)
        )
        
        self._tick_period_s = 1.0 / publish_rate_hz
        self._tick_prev_t = None
        self._tick_count = 0
        self._tick_overrun_count = 0
        self._tick_max_dt_s = 0.0
        # Logging behavior (tune as needed)
        self._tick_log_every = 40          # info every ~2s at 20Hz
        self._tick_warn_factor = 1.5       # warn if dt > 1.5 * expected
        self._tick_error_factor = 3.0      # error if dt > 3.0 * expected

    def _joy_cb(self, msg: Joy) -> None:
        if self._btn_drive >= len(msg.buttons) or self._btn_estop >= len(msg.buttons):
            return

        drive_pressed = msg.buttons[self._btn_drive] == 1
        estop_pressed = msg.buttons[self._btn_estop] == 1

        if estop_pressed and not self._prev_estop_pressed:
            self._manual_estop = True
            self._forward_enabled = False
            self._publish_stop()
            self.get_logger().warn("E-STOP latched by controller (X pressed).")

        if drive_pressed and not self._prev_drive_pressed:
            self._manual_estop = False
            if self._obstacle_blocked:
                self._forward_enabled = False
                self.get_logger().warn("Drive requested (Y), but path ahead is blocked.")
            else:
                self._forward_enabled = True
                self.get_logger().info("Forward drive enabled (Y pressed).")

        self._prev_drive_pressed = drive_pressed
        self._prev_estop_pressed = estop_pressed

    def _sector_cb(self, msg: LaserScan) -> None:
        blocked = False
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or r <= 0.0:
                continue

            angle = msg.angle_min + (i + 0.5) * msg.angle_increment
            if abs(angle) > self._front_half_angle:
                continue

            obj_type = 1
            if i < len(msg.intensities):
                obj_type = int(round(msg.intensities[i]))

            stop_threshold = self._estop_dist_human if obj_type == 2 else self._estop_dist_static
            if r <= stop_threshold:
                blocked = True
                break

        if blocked and not self._obstacle_blocked:
            self.get_logger().warn("SECTOR E-STOP: obstacle detected in immediate forward cone.")
            self._publish_stop()

        self._obstacle_blocked = blocked
        if blocked:
            self._forward_enabled = False

    def _tick_orig(self) -> None:
        if self._manual_estop or self._obstacle_blocked or not self._forward_enabled:
            self._publish_stop()
            return

        cmd = Twist()
        cmd.linear.x = float(self._forward_speed)
        cmd.angular.z = 0.0
        self._cmd_pub.publish(cmd)

    def _tick(self) -> None:
        now = time.monotonic()
        self._tick_count += 1

        dt_s = None
        jitter_s = None
        missed_est = 0

        if self._tick_prev_t is not None:
            dt_s = now - self._tick_prev_t
            jitter_s = dt_s - self._tick_period_s

            if dt_s > self._tick_max_dt_s:
                self._tick_max_dt_s = dt_s

            # Rough estimate of skipped timer intervals
            ratio = dt_s / self._tick_period_s if self._tick_period_s > 0.0 else 1.0
            missed_est = max(0, int(round(ratio)) - 1)

            if dt_s > self._tick_period_s * self._tick_warn_factor:
                self._tick_overrun_count += 1
                level_msg = (
                    "Tick delayed: dt=%.1fms expected=%.1fms jitter=+%.1fms missed~%d "
                    "(overruns=%d/%d, max_dt=%.1fms)"
                    % (
                        dt_s * 1e3,
                        self._tick_period_s * 1e3,
                        max(0.0, jitter_s) * 1e3,
                        missed_est,
                        self._tick_overrun_count,
                        self._tick_count,
                        self._tick_max_dt_s * 1e3,
                    )
                )
                if dt_s > self._tick_period_s * self._tick_error_factor:
                    self.get_logger().error(level_msg)
                else:
                    self.get_logger().warn(level_msg)

            elif self._tick_count % self._tick_log_every == 0:
                # Periodic health log (low spam)
                self.get_logger().info(
                    "Tick stats: dt=%.1fms jitter=%+.1fms overruns=%d/%d max_dt=%.1fms"
                    % (
                        dt_s * 1e3,
                        jitter_s * 1e3,
                        self._tick_overrun_count,
                        self._tick_count,
                        self._tick_max_dt_s * 1e3,
                    )
                )

        self._tick_prev_t = now

        if self._manual_estop or self._obstacle_blocked or not self._forward_enabled:
            self._publish_stop()
            return

        cmd = Twist()
        cmd.linear.x = float(self._forward_speed)
        cmd.angular.z = 0.0
        self._cmd_pub.publish(cmd)

    def _publish_stop(self) -> None:
        self._cmd_pub.publish(Twist())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
