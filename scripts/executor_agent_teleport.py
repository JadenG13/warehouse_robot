#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from warehouse_robot.srv import ExecutePath, ExecutePathResponse
from warehouse_robot.msg import RobotStatus
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class ExecutorAgentTeleport:
    def __init__(self):
        rospy.init_node('executor_agent_teleport')

        # Load grid cell size from your world params
        self.world_name = rospy.get_param("~world_name")
        self.config = rospy.get_param(self.world_name)
        self.cell_size = self.config["cell_size"]

        # —— status publisher so Manager/OLLAMA know when we’re free/busy ——
        self.status_pub = rospy.Publisher(
            '/robot_1/status', RobotStatus, queue_size=1, latch=True
        )
        # signal “idle” at startup
        idle = RobotStatus(
            robot_id='robot_1',
            state='idle',
            task_id='',
            battery_pct=100
        )
        self.status_pub.publish(idle)
        # — end status publisher setup —

        # Initialize Gazebo services
        rospy.wait_for_service('/gazebo/get_model_state')
        rospy.wait_for_service('/gazebo/set_model_state')
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Service for path execution
        self.srv = rospy.Service('execute_path', ExecutePath, self.execute_callback)
        rospy.loginfo("[EXEC] Ready.")

    def execute_callback(self, req):
        """Handle execution of a path defined by waypoints"""
        # — publish “busy” at the very start —
        busy_msg = RobotStatus(robot_id='robot_1',
                               state='busy',
                               task_id=req.task_id if hasattr(req, 'task_id') else '',
                               battery_pct=0)
        self.status_pub.publish(busy_msg)


        if not req.waypoints:
            rospy.logerr("[EXEC] No waypoints provided")
            return ExecutePathResponse(success=False)

        # signal “busy” so no one reassigns us mid-teleport
        busy = RobotStatus(
            robot_id='robot_1',
            state='busy',
            task_id=req.task_id,
            battery_pct=100
        )
        self.status_pub.publish(busy)

        rospy.loginfo(f"[EXEC] Executing path with {len(req.waypoints)} waypoints")

        # Print all waypoints at start for debugging
        rospy.loginfo("[EXEC] Full waypoint list:")
        for i, wp in enumerate(req.waypoints):
            rospy.loginfo(f"[EXEC] Waypoint {i}: ({wp.pose.position.x:.2f}, {wp.pose.position.y:.2f})")

        for i, (waypoint, action) in enumerate(zip(req.waypoints, req.suggested_actions)):
            # Get current state
            current = self.get_model_state(
                model_name='turtlebot3_burger',
                relative_entity_name='map'
            )
            if not current.success:
                rospy.logerr("[EXEC] Failed to get robot state")
                return ExecutePathResponse(success=False)

            # Detailed waypoint logging
            rospy.loginfo(f"\n[EXEC] === Executing waypoint {i} ===")
            rospy.loginfo(f"[EXEC] Target: ({waypoint.pose.position.x:.2f}, {waypoint.pose.position.y:.2f})")
            rospy.loginfo(f"[EXEC] Action: {action}")
            if i < len(req.descriptions) and req.descriptions[i]:
                rospy.loginfo(f"[EXEC] Description: {req.descriptions[i]}")

            # Validate move distance
            if action in ['F', 'B']:
                dx = waypoint.pose.position.x - current.pose.position.x
                dy = waypoint.pose.position.y - current.pose.position.y
                dist = math.hypot(dx, dy)
                rospy.loginfo(f"[EXEC] Movement distance: {dist:.3f}m (should be {self.cell_size}m)")
                if abs(dist - self.cell_size) > 0.01:
                    rospy.logwarn(f"[EXEC] Distance {dist:.3f}m ≠ cell_size {self.cell_size}m")

            # Teleport to waypoint
            if not self.teleport_robot(waypoint.pose):
                rospy.logerr(f"[EXEC] Failed to teleport to waypoint {i}")
                return ExecutePathResponse(success=False)

            rospy.sleep(0.5)

        rospy.loginfo("[EXEC] Path execution completed")

        # signal “idle” again now that we’re done
        idle = RobotStatus(
            robot_id='robot_1',
            state='idle',
            task_id='',
            battery_pct=100
        )
        self.status_pub.publish(idle)

        return ExecutePathResponse(success=True)

    def teleport_robot(self, pose):
        """Teleport robot to given pose"""
        try:
            rospy.loginfo(f"[EXEC] Target pose: x={pose.position.x:.2f}, y={pose.position.y:.2f}")
            current = self.get_model_state(
                model_name='turtlebot3_burger',
                relative_entity_name='map'
            )
            rospy.loginfo(f"[EXEC] Current pose before teleport: "
                          f"x={current.pose.position.x:.2f}, y={current.pose.position.y:.2f}")

            state = ModelState()
            state.model_name = 'turtlebot3_burger'
            state.pose = pose
            state.reference_frame = 'map'
            # zero velocities
            state.twist.linear.x = 0
            state.twist.linear.y = 0
            state.twist.linear.z = 0
            state.twist.angular.x = 0
            state.twist.angular.y = 0
            state.twist.angular.z = 0

            resp = self.set_model_state(state)
            if not resp.success:
                rospy.logwarn("[EXEC] Teleport failed!")
                return False

            after = self.get_model_state(
                model_name='turtlebot3_burger',
                relative_entity_name='map'
            )
            rospy.loginfo(f"[EXEC] Pose after teleport: "
                          f"x={after.pose.position.x:.2f}, y={after.pose.position.y:.2f}")

            rospy.sleep(0.5)
            return True

        except rospy.ServiceException as e:
            rospy.logerr(f"[EXEC] Service call failed: {e}")
            return False


if __name__ == '__main__':
    try:
        ExecutorAgentTeleport()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
