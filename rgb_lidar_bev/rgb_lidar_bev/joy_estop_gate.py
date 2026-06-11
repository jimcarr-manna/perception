"""Joystick-gated cmd_vel passthrough for autonomous navigation safety."""
from __future__ import annotations

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy


class JoyEstopGate(Node):
    def __init__(self) -> None:
        super().__init__("joy_estop_gate")

        self.declare_parameter("joy_topic", "/ros_robot_controller/joy")
        self.declare_parameter("input_cmd_vel_topic", "/cmd_vel_nav")
        self.declare_parameter("output_cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("btn_drive_index", 4)  # Y
        self.declare_parameter("btn_estop_index", 3)  # X
        self.declare_parameter("require_enable_button", True)

        joy_topic = str(self.get_parameter("joy_topic").value or "/ros_robot_controller/joy")
        input_topic = str(self.get_parameter("input_cmd_vel_topic").value or "/cmd_vel_nav")
        output_topic = str(self.get_parameter("output_cmd_vel_topic").value or "/cmd_vel")
        self._btn_drive = int(self.get_parameter("btn_drive_index").value or 4)
        self._btn_estop = int(self.get_parameter("btn_estop_index").value or 3)
        self._require_enable_button = bool(
            self.get_parameter("require_enable_button").value
        )

        # Start in latched-stop mode when enable button is required.
        self._manual_estop = bool(self._require_enable_button)
        self._enabled = not self._require_enable_button
        self._prev_drive_pressed = False
        self._prev_estop_pressed = False

        self._cmd_pub = self.create_publisher(Twist, output_topic, 10)
        self.create_subscription(Joy, joy_topic, self._joy_cb, 10)
        self.create_subscription(Twist, input_topic, self._cmd_in_cb, 10)

        self.get_logger().info(
            "Joy gate ready: %s -> %s, joy=%s, Y[%d]=enable, X[%d]=estop, start_enabled=%s"
            % (
                input_topic,
                output_topic,
                joy_topic,
                self._btn_drive,
                self._btn_estop,
                str(self._enabled),
            )
        )

    def _joy_cb(self, msg: Joy) -> None:
        if self._btn_drive >= len(msg.buttons) or self._btn_estop >= len(msg.buttons):
            return

        drive_pressed = msg.buttons[self._btn_drive] == 1
        estop_pressed = msg.buttons[self._btn_estop] == 1

        if estop_pressed and not self._prev_estop_pressed:
            self._manual_estop = True
            self._enabled = False
            self._publish_stop()
            self.get_logger().warn("E-STOP latched by controller (X pressed).")

        if drive_pressed and not self._prev_drive_pressed:
            self._manual_estop = False
            self._enabled = True
            self.get_logger().info("Drive gate enabled (Y pressed).")

        self._prev_drive_pressed = drive_pressed
        self._prev_estop_pressed = estop_pressed

    def _cmd_in_cb(self, msg: Twist) -> None:
        if self._manual_estop or not self._enabled:
            self._publish_stop()
            return
        self._cmd_pub.publish(msg)

    def _publish_stop(self) -> None:
        self._cmd_pub.publish(Twist())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JoyEstopGate()
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
