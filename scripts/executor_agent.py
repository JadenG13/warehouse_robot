#!/usr/bin/env python3
import rospy
import math
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from warehouse_robot.srv import ExecutePath, ExecutePathResponse, GetPath
from warehouse_robot.msg import RobotStatus
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import OccupancyGrid


class ExecutorAgent:
    def __init__(self):
        rospy.init_node('executor_agent')

        # Load grid cell size from your world params
        self.world_name = rospy.get_param("~world_name")
        self.config = rospy.get_param(self.world_name)
        self.cell_size = self.config["cell_size"]

        # Costmap data
        self.global_costmap = None
        self.costmap_info = None
        
        # Subscribe to global costmap
        self.costmap_sub = rospy.Subscriber(
            '/move_base/global_costmap/costmap',
            OccupancyGrid,
            self.costmap_callback,
            queue_size=10
        )
        
        # Wait for initial costmap data
        rospy.loginfo("[EXEC] Waiting for costmap...")
        while self.global_costmap is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("[EXEC] Got initial costmap")

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

        # Wait for path planning service
        rospy.wait_for_service('/get_path')
        self.get_path = rospy.ServiceProxy('/get_path', GetPath)
        
        # Service for path execution
        self.srv = rospy.Service('execute_path', ExecutePath, self.execute_callback)
        rospy.loginfo("[EXEC] Ready.")

    def costmap_callback(self, msg):
        """Store incoming costmap data"""
        self.global_costmap = list(msg.data)
        self.costmap_info = msg.info

    def is_cell_blocked(self, x, y):
        """Check if the target cell (x,y) is occupied"""
        if not self.global_costmap or not self.costmap_info:
            rospy.logwarn("[EXEC] No costmap data available")
            return True

        if (0 <= x < self.costmap_info.width and 
            0 <= y < self.costmap_info.height):
            idx = y * self.costmap_info.width + x
            if idx < 0 or idx >= len(self.global_costmap):
                rospy.logwarn(f"[EXEC] Invalid index {idx} for costmap at ({x}, {y})")
                return True
            cost = self.global_costmap[idx]
            if cost > 0:  # ROS costmaps typically use 0-100 range, with >= 50 being obstacles
                rospy.loginfo(f"[EXEC] Detected obstacle at ({x}, {y}) with cost {cost}")
                return True
        else:
            # Treat out-of-bounds as occupied for safety
            rospy.loginfo(f"[EXEC] Cell ({x}, {y}) is out of bounds")
            return True
        return False

    def check_position_safety(self, target_pose):
        """Check if a target pose is safe to move to"""
        if not self.global_costmap or not self.costmap_info:
            return False, "No costmap data available"

        try:
            # Convert world coordinates to grid coordinates
            target_x = int(round((target_pose.position.x - self.costmap_info.origin.position.x) / self.cell_size))
            target_y = int(round((target_pose.position.y - self.costmap_info.origin.position.y) / self.cell_size))
            
            rospy.loginfo(f"[EXEC] Checking safety of position ({target_x}, {target_y})")

            # Check if target position is blocked
            if self.is_cell_blocked(target_x, target_y):
                return False, "Target position is blocked by an obstacle"

            # Get local occupancy grid around target
            map_section = []
            for dy in range(-1, 2):
                row = []
                for dx in range(-1, 2):
                    x, y = target_x + dx, target_y + dy
                    if (0 <= x < self.costmap_info.width and 
                        0 <= y < self.costmap_info.height):
                        idx = y * self.costmap_info.width + x
                        value = self.global_costmap[idx]
                        # row.append(' X ' if value > 0 else ' _ ')
                        row.append(f' {value} ')
                    else:
                        row.append(' ? ')
                map_section.append(row)
            
            # Print the 3x3 grid around target
            rospy.loginfo("Local map section around target (3x3 grid):")
            for row in map_section:
                rospy.loginfo(' '.join(row))

            return True, "Position is safe"

        except Exception as e:
            rospy.logerr(f"[EXEC] Error checking position safety: {str(e)}")
            return False, f"Error checking position: {str(e)}"

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

            # Check next target position
            is_safe, message = self.check_position_safety(waypoint.pose)

            if not is_safe:
                rospy.logwarn(f"[EXEC] Movement validation failed: {message}")
                
                # Request a replan by calling get_path with current position
                rospy.loginfo("[EXEC] Requesting new path")
                replan_result = self.get_path(
                    start_pose=current.pose,
                    goal_pose=req.waypoints[-1].pose  # Final goal position
                )
                
                if replan_result.waypoints:
                    rospy.loginfo("[EXEC] New path received, awaiting execution")
                    return ExecutePathResponse(success=True)
                else:
                    rospy.logerr("[EXEC] Failed to get new path")
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