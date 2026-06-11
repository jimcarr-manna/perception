"""Simple NavigateToPose action client with optional initial pose publish."""
from __future__ import annotations

import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


class GoalClientNode(Node):
    def __init__(self) -> None:
        super().__init__("bev_goal_client")

        self.declare_parameter("action_name", "/navigate_to_pose")
        self.declare_parameter("goal_frame_id", "map")
        self.declare_parameter("goal_x", 0.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_yaw", 0.0)
        self.declare_parameter("behavior_tree", "")
        self.declare_parameter("wait_for_server_sec", 20.0)
        self.declare_parameter("result_timeout_sec", 180.0)
        self.declare_parameter("send_initial_pose", False)
        self.declare_parameter("initial_pose_topic", "/initialpose")
        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_yaw", 0.0)
        self.declare_parameter("initial_covariance_xy", 0.25)
        self.declare_parameter("initial_covariance_yaw", 0.20)

        action_name = str(self.get_parameter("action_name").value)
        self._goal_frame = str(self.get_parameter("goal_frame_id").value)
        self._goal_x = float(self.get_parameter("goal_x").value)
        self._goal_y = float(self.get_parameter("goal_y").value)
        self._goal_yaw = float(self.get_parameter("goal_yaw").value)
        self._behavior_tree = str(self.get_parameter("behavior_tree").value)
        self._wait_for_server_sec = float(self.get_parameter("wait_for_server_sec").value)
        self._result_timeout_sec = float(self.get_parameter("result_timeout_sec").value)

        self._send_initial_pose = bool(self.get_parameter("send_initial_pose").value)
        self._initial_pose_topic = str(self.get_parameter("initial_pose_topic").value)
        self._initial_x = float(self.get_parameter("initial_x").value)
        self._initial_y = float(self.get_parameter("initial_y").value)
        self._initial_yaw = float(self.get_parameter("initial_yaw").value)
        self._initial_cov_xy = float(self.get_parameter("initial_covariance_xy").value)
        self._initial_cov_yaw = float(self.get_parameter("initial_covariance_yaw").value)

        self._action_client = ActionClient(self, NavigateToPose, action_name)
        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, self._initial_pose_topic, 10
        )
        self._last_feedback_log_t = 0.0

    def run(self) -> int:
        if self._send_initial_pose:
            self._publish_initial_pose()
            # Give localization stack a brief chance to process the estimate.
            time.sleep(0.5)

        self.get_logger().info("Waiting for action server...")
        if not self._action_client.wait_for_server(timeout_sec=self._wait_for_server_sec):
            self.get_logger().error("NavigateToPose action server not available.")
            return 1

        goal = self._make_goal()
        self.get_logger().info(
            "Sending goal: frame=%s x=%.3f y=%.3f yaw=%.3f"
            % (self._goal_frame, self._goal_x, self._goal_y, self._goal_yaw)
        )
        send_future = self._action_client.send_goal_async(goal, feedback_callback=self._feedback_cb)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None:
            self.get_logger().error("Failed to send goal (timeout or communication error).")
            return 1
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by navigation stack.")
            return 1

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=max(1.0, self._result_timeout_sec)
        )
        wrapped_result = result_future.result()
        if wrapped_result is None:
            self.get_logger().error("Goal result timed out; cancelling goal.")
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
            return 1

        if wrapped_result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Goal reached successfully.")
            return 0

        self.get_logger().error("Navigation failed with status code: %d" % wrapped_result.status)
        return 1

    def _publish_initial_pose(self) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = self._initial_x
        msg.pose.pose.position.y = self._initial_y
        qx, qy, qz, qw = _yaw_to_quat(self._initial_yaw)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance[0] = self._initial_cov_xy
        msg.pose.covariance[7] = self._initial_cov_xy
        msg.pose.covariance[35] = self._initial_cov_yaw
        self._initial_pose_pub.publish(msg)
        self.get_logger().info(
            "Published initial pose estimate: x=%.3f y=%.3f yaw=%.3f"
            % (self._initial_x, self._initial_y, self._initial_yaw)
        )

    def _make_goal(self) -> NavigateToPose.Goal:
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self._goal_frame
        pose.pose.position.x = self._goal_x
        pose.pose.position.y = self._goal_y
        qx, qy, qz, qw = _yaw_to_quat(self._goal_yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        goal.pose = pose
        if self._behavior_tree:
            goal.behavior_tree = self._behavior_tree
        return goal

    def _feedback_cb(self, feedback_msg: NavigateToPose.FeedbackMessage) -> None:
        now = time.monotonic()
        if now - self._last_feedback_log_t < 1.0:
            return
        self._last_feedback_log_t = now
        eta = feedback_msg.feedback.estimated_time_remaining.sec
        self.get_logger().info("Feedback: distance_remaining=%.2f eta=%ds" % (
            feedback_msg.feedback.distance_remaining,
            eta,
        ))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GoalClientNode()
    try:
        rc = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
