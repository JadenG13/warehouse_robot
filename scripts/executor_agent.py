#!/usr/bin/env python3
import rospy
import math
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from warehouse_robot.srv import ExecutePath, ExecutePathResponse, CheckMovement, GetPath
from warehouse_robot.msg import RobotStatus
from tf.transformations import euler_from_quaternion


class ExecutorAgent:
    def __init__(self):
        rospy.init_node('executor_agent')

        # Load grid cell size from your world params
        self.world_name = rospy.get_param("~world_name")
        self.config = rospy.get_param(self.world_name)
        self.cell_size = self.config["cell_size"]

        # Status publisher setup
        self.status_pub = rospy.Publisher(
            '/robot_1/status', RobotStatus, queue_size=1, latch=True
        )
        # Signal "idle" at startup
        idle = RobotStatus(
            robot_id='robot_1',
            state='idle',
            task_id='',
        )
        self.status_pub.publish(idle)

        # Initialize Gazebo services
        rospy.wait_for_service('/gazebo/get_model_state')
        rospy.wait_for_service('/gazebo/set_model_state')
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Wait for movement validation service
        rospy.wait_for_service('check_movement')
        self.check_movement = rospy.ServiceProxy('check_movement', CheckMovement)

        # Wait for path planning service
        rospy.wait_for_service('/get_path')
        self.get_path = rospy.ServiceProxy('/get_path', GetPath)
        
        # Service for path execution
        self.srv = rospy.Service('execute_path', ExecutePath, self.execute_callback)
        rospy.loginfo("[EXEC] Ready.")

    def execute_callback(self, req):
        """Handle execution of a path defined by waypoints"""
        rospy.loginfo("[EXEC] Received path execution request")
        # Publish "busy" state
        busy_msg = RobotStatus(
            robot_id='robot_1',
            state='busy',
            task_id=req.task_id if hasattr(req, 'task_id') else ''
        )
        self.status_pub.publish(busy_msg)
        
        rospy.loginfo(f"[EXEC] Task ID: {req.task_id}")

        if not req.waypoints:
            rospy.logerr("[EXEC] No waypoints provided")
            return ExecutePathResponse(success=False)

        rospy.loginfo(f"[EXEC] Executing path with {len(req.waypoints)} waypoints")

        # Print all waypoints at start for debugging
        rospy.loginfo("[EXEC] Full waypoint list:")
        for i, wp in enumerate(req.waypoints):
            quat = [
                wp.pose.orientation.x,
                wp.pose.orientation.y,
                wp.pose.orientation.z,
                wp.pose.orientation.w
            ]
            _, _, yaw = euler_from_quaternion(quat)
            rospy.loginfo(f"[EXEC] Waypoint {i}: ({wp.pose.position.x:.2f}, {wp.pose.position.y:.2f}), Orientation: {math.degrees(yaw):.0f}°")

        # Get initial robot state
        current = self.get_model_state(
            model_name='turtlebot3_burger',
            relative_entity_name='map'
        )
        if not current.success:
            rospy.logerr("[EXEC] Failed to get initial robot state")
            return ExecutePathResponse(success=False)

        for i, (waypoint, action) in enumerate(zip(req.waypoints, req.suggested_actions)):
            rospy.loginfo(f"\n[EXEC] === Executing waypoint {i} ===")
            rospy.loginfo(f"[EXEC] Target: ({waypoint.pose.position.x:.2f}, {waypoint.pose.position.y:.2f})")
            rospy.loginfo(f"[EXEC] Action: {action}")
            if i < len(req.descriptions) and req.descriptions[i]:
                rospy.loginfo(f"[EXEC] Description: {req.descriptions[i]}")

            # Check movement safety
            check_result = self.check_movement(
                current_pose=current.pose,
                target_pose=waypoint.pose,
                action=action
            )

            if not check_result.success:
                rospy.logwarn(f"[EXEC] Movement validation failed: {check_result.message}")
                
                # Request a replan by calling get_path with current position
                rospy.loginfo("[EXEC] Requesting new path")
                replan_result = self.get_path(
                    start_pose=current.pose,
                    goal_pose=req.waypoints[-1].pose  # Final goal position
                )
                
                if replan_result.waypoints:
                    # Don't return failure - the manager will get a new path and try again
                    rospy.loginfo("[EXEC] New path received, awaiting execution")
                    return ExecutePathResponse(success=True)
                else:
                    rospy.logerr("[EXEC] Failed to get new path")
                    # Return failure if path planning failed
                    return ExecutePathResponse(success=False)

            rospy.loginfo("[EXEC] Movement validated as safe")

            # Validate move distance for forward/backward moves
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

            # Update current state for next iteration
            current.pose = waypoint.pose
            rospy.sleep(0.2)

        rospy.loginfo("[EXEC] Path execution completed")

        # Signal "idle" again now that we're done
        idle = RobotStatus(
            robot_id='robot_1',
            state='idle',
            task_id='',
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
            # Zero velocities
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

            rospy.sleep(0.2)
            return True

        except rospy.ServiceException as e:
            rospy.logerr(f"[EXEC] Service call failed: {e}")
            return False


if __name__ == '__main__':
    try:
        ExecutorAgent()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
