#!/usr/bin/env python3
import rospy
import math
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState #  uses full names for service proxies
from warehouse_robot.srv import ExecutePath, ExecutePathResponse
from warehouse_robot.msg import RobotStatus
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker
import time # For timing blocks

# Helper to format pose for logging
# Returns a string with x, y, and heading in degrees
# Accepts either a geometry_msgs.Pose or a dict with x, y, theta
def format_pose_for_log(pose_obj_from_waypoint): # pose_obj_from_waypoint is waypoint.pose
    # This function now correctly uses np.degrees
    if hasattr(pose_obj_from_waypoint, 'orientation') and \
            all(hasattr(pose_obj_from_waypoint.orientation, attr) for attr in ['x', 'y', 'z', 'w']):
        try:
            q_list = [
                pose_obj_from_waypoint.orientation.x,
                pose_obj_from_waypoint.orientation.y,
                pose_obj_from_waypoint.orientation.z,
                pose_obj_from_waypoint.orientation.w
            ]
            if any(math.isnan(val) or math.isinf(val) for val in q_list): # math is usually imported
                raise ValueError("NaN or Inf in quaternion components")

            if all(abs(val) < 1e-9 for val in q_list):
                yaw_deg = 0.0
            else:
                _, _, yaw_rad = euler_from_quaternion(q_list)
                yaw_deg = np.degrees(yaw_rad)

            return f"x:{pose_obj_from_waypoint.position.x:.2f}_y:{pose_obj_from_waypoint.position.y:.2f}_th:{yaw_deg:.0f}"
        except Exception as e:
            rospy.logwarn_throttle(5,
                                   f"Error converting waypoint pose to log string ({type(e).__name__}): {e}. Pose: {pose_obj_from_waypoint}")
            return f"x:{pose_obj_from_waypoint.position.x:.2f}_y:{pose_obj_from_waypoint.position.y:.2f}_th:ERR_CONV_QUAT"
    elif isinstance(pose_obj_from_waypoint, dict) and all(k in pose_obj_from_waypoint for k in ['x','y','theta']): # For dict style pose
        return f"x:{pose_obj_from_waypoint['x']:.2f}_y:{pose_obj_from_waypoint['y']:.2f}_th:{np.degrees(pose_obj_from_waypoint['theta']):.0f}" # np is now defined
    return "POSE_FORMAT_ERROR_STRUCTURE"



class ExecutorAgent:
    def __init__(self):
        # Initialize node and parameters
        rospy.init_node('executor_agent')
        self.node_start_time = rospy.Time.now().to_sec()
        rospy.loginfo(f"[PERF_LOG] event=EXECUTOR_INIT, timestamp={self.node_start_time:.3f}")

        # Load grid cell size from world params
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
        wait_for_costmap_start_time = time.monotonic()
        while self.global_costmap is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        wait_for_costmap_duration = time.monotonic() - wait_for_costmap_start_time
        if self.global_costmap is not None:
            rospy.loginfo("[EXEC] Got initial costmap")
            rospy.loginfo(f"[PERF_LOG] event=EXECUTOR_COSTMAP_READY, timestamp={rospy.Time.now().to_sec():.3f}, wait_duration_sec={wait_for_costmap_duration:.3f}")
        else:
            rospy.logwarn("[EXEC] Shutting down before costmap was received.")
            rospy.loginfo(f"[PERF_LOG] event=EXECUTOR_COSTMAP_TIMEOUT_OR_SHUTDOWN, timestamp={rospy.Time.now().to_sec():.3f}, wait_duration_sec={wait_for_costmap_duration:.3f}")

        # Status publisher
        self.status_pub = rospy.Publisher(
            '/robot_1/status', RobotStatus, queue_size=1, latch=True
        )
        # Publish idle at startup
        idle = RobotStatus(
            robot_id='robot_1',
            state='idle',
            task_id='',
        )
        self.status_pub.publish(idle)
        rospy.loginfo(f"[PERF_LOG] event=EXECUTOR_ROBOT_STATUS_PUBLISHED, timestamp={rospy.Time.now().to_sec():.3f}, robot_id=robot_1, state=idle, task_id=''")

        # Initialize Gazebo services
        rospy.wait_for_service('/gazebo/get_model_state')
        rospy.wait_for_service('/gazebo/set_model_state')
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Service for path execution
        self.srv = rospy.Service('execute_path', ExecutePath, self.execute_callback)
        rospy.loginfo("[EXEC] Ready.")

        # Publisher for planned path visualization
        self.path_pub = rospy.Publisher(
            '/planned_path', Path, queue_size=1, latch=True
        )
        # Publishers for visualization
        self.start_marker_pub = rospy.Publisher('/start_position', Marker, queue_size=1, latch=True)
        self.goal_marker_pub = rospy.Publisher('/goal_position', Marker, queue_size=1, latch=True)

    def costmap_callback(self, msg):
        # Store incoming costmap data
        is_first_time = self.global_costmap is None
        self.global_costmap = list(msg.data)
        self.costmap_info = msg.info
        if is_first_time and self.costmap_info:
            rospy.loginfo(f"[PERF_LOG] event=EXECUTOR_FIRST_COSTMAP_DATA_RECEIVED, timestamp={rospy.Time.now().to_sec():.3f}, width={msg.info.width}, height={msg.info.height}, res={msg.info.resolution:.3f}")
            rospy.loginfo(f"[PERF_LOG] event=EXECUTOR_CONFIG_CELL_SIZE, timestamp={rospy.Time.now().to_sec():.3f}, conceptual_cell_size={self.cell_size:.3f}")

    def is_cell_blocked(self, x, y):
        # Return True if cell (x, y) is occupied or out of bounds
        if not self.global_costmap or not self.costmap_info:
            rospy.logwarn("[EXEC] No costmap data available")
            return True
        if (0 <= x < self.costmap_info.width and 0 <= y < self.costmap_info.height):
            idx = y * self.costmap_info.width + x
            if idx < 0 or idx >= len(self.global_costmap):
                rospy.logwarn(f"[EXEC] Invalid index {idx} for costmap at ({x}, {y})")
                return True
            cost = self.global_costmap[idx]
            if cost > 0:
                rospy.loginfo(f"[EXEC] Detected obstacle at ({x}, {y}) with cost {cost}")
                return True
        else:
            rospy.loginfo(f"[EXEC] Cell ({x}, {y}) is out of bounds")
            return True
        return False

    def check_position_safety(self, target_pose):
        # Check if a target pose is safe to move to
        safety_check_start_time_monotonic = time.monotonic()
        safety_check_start_time_ros = rospy.Time.now().to_sec()
        rospy.loginfo(f"[PERF_LOG] event=EXEC_SAFETY_CHECK_START, timestamp={safety_check_start_time_ros:.3f}, target_world_pose={format_pose_for_log(target_pose)}")
        if not self.global_costmap or not self.costmap_info:
            safety_check_duration_sec = time.monotonic() - safety_check_start_time_monotonic
            rospy.loginfo(f"[PERF_LOG] event=EXEC_SAFETY_CHECK_END, timestamp={rospy.Time.now().to_sec():.3f}, duration_sec={safety_check_duration_sec:.4f}, is_safe=False, reason='No costmap data'")
            return False, "No costmap data available"
        target_x_grid, target_y_grid = -1, -1
        try:
            # Convert world coordinates to grid coordinates
            target_x_grid = int(round((target_pose.position.x - self.costmap_info.origin.position.x) / self.cell_size))
            target_y_grid = int(round((target_pose.position.y - self.costmap_info.origin.position.y) / self.cell_size))
            rospy.loginfo(f"[PERF_LOG] event=EXEC_SAFETY_CHECK_GRID_CONVERSION, timestamp={rospy.Time.now().to_sec():.3f}, world_x={target_pose.position.x:.2f}, world_y={target_pose.position.y:.2f}, grid_x={target_x_grid}, grid_y={target_y_grid}, using_cell_size={self.cell_size:.3f}")
            rospy.loginfo(f"[EXEC] Checking safety of position ({target_x_grid}, {target_y_grid})")
            is_blocked = self.is_cell_blocked(target_x_grid, target_y_grid)
            if is_blocked:
                safety_check_duration_sec = time.monotonic() - safety_check_start_time_monotonic
                rospy.loginfo(f"[PERF_LOG] event=EXEC_SAFETY_CHECK_END, timestamp={rospy.Time.now().to_sec():.3f}, duration_sec={safety_check_duration_sec:.4f}, is_safe=False, reason='Target position blocked', target_grid=({target_x_grid},{target_y_grid})")
                return False, "Target position is blocked by an obstacle"
            # Show local 3x3 grid for debugging
            map_section = []
            for dy in range(-1, 2):
                row = []
                for dx in range(-1, 2):
                    x, y = target_x_grid + dx, target_y_grid + dy
                    if (0 <= x < self.costmap_info.width and 0 <= y < self.costmap_info.height):
                        idx = y * self.costmap_info.width + x
                        if 0 <= idx < len(self.global_costmap):
                            value = self.global_costmap[idx]
                            row.append(f' {value} ')
                        else:
                            row.append(' ! ')
                    else:
                        row.append(' ? ')
                map_section.append(row)
            rospy.loginfo("Local map section around target (3x3 grid):")
            for row_idx, map_row_data in enumerate(map_section):
                rospy.loginfo(' '.join(map_row_data))
            safety_check_duration_sec = time.monotonic() - safety_check_start_time_monotonic
            rospy.loginfo(f"[PERF_LOG] event=EXEC_SAFETY_CHECK_END, timestamp={rospy.Time.now().to_sec():.3f}, duration_sec={safety_check_duration_sec:.4f}, is_safe=True, target_grid=({target_x_grid},{target_y_grid})")
            return True, "Position is safe"
        except Exception as e:
            rospy.logerr(f"[EXEC] Error checking position safety: {str(e)}")
            safety_check_duration_sec = time.monotonic() - safety_check_start_time_monotonic
            rospy.loginfo(f"[PERF_LOG] event=EXEC_SAFETY_CHECK_END, timestamp={rospy.Time.now().to_sec():.3f}, duration_sec={safety_check_duration_sec:.4f}, is_safe=False, reason='Exception: {str(e)}', target_grid=({target_x_grid},{target_y_grid})")
            return False, f"Error checking position: {str(e)}"

    def execute_callback(self, req):
        # Handle execution of a path defined by waypoints
        handler_start_time_ros = rospy.Time.now().to_sec()
        task_id_for_log = req.task_id if hasattr(req, 'task_id') else 'UNKNOWN_TASK'
        num_waypoints_for_log = len(req.waypoints) if req.waypoints else 0
        rospy.loginfo(f"[PERF_LOG] event=EXEC_PATH_REQUEST_RECEIVED, timestamp={handler_start_time_ros:.3f}, task_id='{task_id_for_log}', num_waypoints={num_waypoints_for_log}")
        rospy.loginfo("[EXEC] Received path execution request")
        # Publish start and goal markers if waypoints exist
        if req.waypoints:
            # Publish start position (first waypoint)
            start_marker = self.create_pose_marker(req.waypoints[0].pose, is_start=True)
            self.start_marker_pub.publish(start_marker)
            
            # Publish goal position (last waypoint)
            goal_marker = self.create_pose_marker(req.waypoints[-1].pose, is_start=False)
            self.goal_marker_pub.publish(goal_marker)
            
            rospy.loginfo("[EXEC] Published start and goal markers for visualization")
        # Publish the planned path for visualization if present
        if hasattr(req, 'path') and req.path and hasattr(req.path, 'poses') and req.path.poses:
            if not req.path.header.frame_id:
                req.path.header.frame_id = "map"
            self.path_pub.publish(req.path)
            rospy.loginfo("[EXEC] Published planned path for visualization.")
        # Publish busy state
        busy_msg = RobotStatus(
            robot_id='robot_1',
            state='busy',
            task_id=task_id_for_log 
        )
        self.status_pub.publish(busy_msg)
        rospy.loginfo(f"[PERF_LOG] event=EXECUTOR_ROBOT_STATUS_PUBLISHED, timestamp={rospy.Time.now().to_sec():.3f}, robot_id=robot_1, state=busy, task_id='{task_id_for_log}'")
        rospy.loginfo(f"[EXEC] Task ID: {req.task_id}")
        if not req.waypoints:
            rospy.logerr("[EXEC] No waypoints provided")
            planning_duration_sec = rospy.Time.now().to_sec() - handler_start_time_ros
            rospy.loginfo(f"[PERF_LOG] event=EXEC_PATH_FAIL, timestamp={rospy.Time.now().to_sec():.3f}, reason='No waypoints provided', task_id='{task_id_for_log}', exec_duration_sec={planning_duration_sec:.3f}")
            return ExecutePathResponse(success=False)
        rospy.loginfo(f"[EXEC] Executing path with {len(req.waypoints)} waypoints")
        rospy.loginfo("[EXEC] Full waypoint list:")
        for i_wp_log, wp_log in enumerate(req.waypoints):
            quat_log = [wp_log.pose.orientation.x, wp_log.pose.orientation.y, wp_log.pose.orientation.z, wp_log.pose.orientation.w]
            _, _, yaw_log = euler_from_quaternion(quat_log)
            rospy.loginfo(f"[EXEC] Waypoint {i_wp_log}: ({wp_log.pose.position.x:.2f}, {wp_log.pose.position.y:.2f}), Orientation: {math.degrees(yaw_log):.0f}°")
        # Get initial robot state
        get_model_state_call_start_time = time.monotonic()
        current = self.get_model_state(
            model_name='turtlebot3_burger',
            relative_entity_name='map'
        )
        get_model_state_call_duration = time.monotonic() - get_model_state_call_start_time
        rospy.loginfo(f"[PERF_LOG] event=EXEC_GET_MODEL_STATE_INITIAL_CALL, timestamp={rospy.Time.now().to_sec():.3f}, duration_sec={get_model_state_call_duration:.4f}, success={current.success}")
        if not current.success:
            rospy.logerr("[EXEC] Failed to get initial robot state")
            exec_duration_sec = rospy.Time.now().to_sec() - handler_start_time_ros
            rospy.loginfo(f"[PERF_LOG] event=EXEC_PATH_FAIL, timestamp={rospy.Time.now().to_sec():.3f}, reason='Failed to get initial robot state', task_id='{task_id_for_log}', exec_duration_sec={exec_duration_sec:.3f}")
            return ExecutePathResponse(success=False)
        # Loop through waypoints
        num_safety_checks_done = 0
        num_replans_triggered = 0
        for i, (waypoint, action) in enumerate(zip(req.waypoints, req.suggested_actions)):
            wp_exec_start_time_ros = rospy.Time.now().to_sec()
            rospy.loginfo(f"[PERF_LOG] event=EXEC_WAYPOINT_START, timestamp={wp_exec_start_time_ros:.3f}, task_id='{task_id_for_log}', waypoint_idx={i}, target_pose={format_pose_for_log(waypoint.pose)}, action='{action}'")
            rospy.loginfo(f"\n[EXEC] === Executing waypoint {i} ===")
            rospy.loginfo(f"[EXEC] Target: ({waypoint.pose.position.x:.2f}, {waypoint.pose.position.y:.2f})")
            rospy.loginfo(f"[EXEC] Action: {action}")
            if i < len(req.descriptions) and req.descriptions[i]:
                rospy.loginfo(f"[EXEC] Description: {req.descriptions[i]}")
            num_safety_checks_done += 1
            is_safe, message = self.check_position_safety(waypoint.pose)
            if not is_safe:
                rospy.logwarn(f"[EXEC] Movement validation failed: {message}")
                num_replans_triggered += 1
                rospy.loginfo(f"[PERF_LOG] event=EXEC_REPLAN_TRIGGERED, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_for_log}', waypoint_idx={i}, reason='{message}'")
                replan_msg = RobotStatus(
                    robot_id='robot_1',
                    state='replan',
                    task_id=task_id_for_log 
                )
                self.status_pub.publish(replan_msg)
                rospy.loginfo(f"[PERF_LOG] event=EXECUTOR_ROBOT_STATUS_PUBLISHED, timestamp={rospy.Time.now().to_sec():.3f}, robot_id=robot_1, state=replan, task_id='{task_id_for_log}'")
                rospy.loginfo("[EXEC] Requesting new path")
                return ExecutePathResponse(success=True)
            rospy.loginfo("[EXEC] Movement validated as safe")
            # Validate move distance for forward moves only
            if action == 'F':
                dx = waypoint.pose.position.x - current.pose.position.x
                dy = waypoint.pose.position.y - current.pose.position.y
                dist = math.hypot(dx, dy)
                rospy.loginfo(f"[EXEC] Movement distance: {dist:.3f}m (should be {self.cell_size}m)")
                if abs(dist - self.cell_size) > 0.01:
                    rospy.logwarn(f"[EXEC] Distance {dist:.3f}m ≠ cell_size {self.cell_size}m")
                    rospy.loginfo(f"[PERF_LOG] event=EXEC_WAYPOINT_DIST_VALIDATION_WARN, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_for_log}', waypoint_idx={i}, actual_dist={dist:.3f}, expected_conceptual_dist={self.cell_size:.3f}")
            # Teleport to waypoint
            teleport_call_start_time_monotonic = time.monotonic()
            teleport_ok = self.teleport_robot(waypoint.pose)
            teleport_call_duration_sec = time.monotonic() - teleport_call_start_time_monotonic
            rospy.loginfo(f"[PERF_LOG] event=EXEC_TELEPORT_CALL_END, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_for_log}', waypoint_idx={i}, duration_sec={teleport_call_duration_sec:.4f}, success={teleport_ok}")
            if not teleport_ok:
                rospy.logerr(f"[EXEC] Failed to teleport to waypoint {i}")
                exec_duration_sec = rospy.Time.now().to_sec() - handler_start_time_ros
                rospy.loginfo(f"[PERF_LOG] event=EXEC_PATH_FAIL, timestamp={rospy.Time.now().to_sec():.3f}, reason='Teleport failed', task_id='{task_id_for_log}', waypoint_idx={i}, exec_duration_sec={exec_duration_sec:.3f}, num_safety_checks={num_safety_checks_done}, num_replans={num_replans_triggered}")
                return ExecutePathResponse(success=False)
            # Update current state for next iteration
            current.pose = waypoint.pose
            rospy.sleep(0.2)
            rospy.loginfo(f"[PERF_LOG] event=EXEC_WAYPOINT_END, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_for_log}', waypoint_idx={i}, duration_sec={rospy.Time.now().to_sec() - wp_exec_start_time_ros:.3f}")
        rospy.loginfo("[EXEC] Path execution completed")
        # Clear visualizations since we're done
        self.clear_visualizations()
        # Signal "idle" again now that we're done
        idle = RobotStatus(
            robot_id='robot_1',
            state='idle',
            task_id='',
        )
        self.status_pub.publish(idle)
        rospy.loginfo(f"[PERF_LOG] event=EXECUTOR_ROBOT_STATUS_PUBLISHED, timestamp={rospy.Time.now().to_sec():.3f}, robot_id=robot_1, state=idle, task_id=''")
        exec_duration_sec = rospy.Time.now().to_sec() - handler_start_time_ros
        rospy.loginfo(f"[PERF_LOG] event=EXEC_PATH_SUCCESS_COMPLETE, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_for_log}', exec_duration_sec={exec_duration_sec:.3f}, num_waypoints_processed={i+1 if req.waypoints else 0}, num_safety_checks={num_safety_checks_done}, num_replans={num_replans_triggered}")
        return ExecutePathResponse(success=True)

    def create_pose_marker(self, pose, is_start=True):
        # Create a marker for start or goal position
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "start_goal_markers"
        marker.id = 0 if is_start else 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = 0.0 if is_start else 1.0
        marker.color.g = 1.0 if is_start else 0.0
        marker.color.b = 0.0
        return marker

    def clear_visualizations(self):
        # Clear all visualization markers and paths
        empty_path = Path()
        empty_path.header.frame_id = "map"
        empty_path.header.stamp = rospy.Time.now()
        self.path_pub.publish(empty_path)
        start_marker = Marker()
        start_marker.header.frame_id = "map"
        start_marker.header.stamp = rospy.Time.now()
        start_marker.ns = "start_goal_markers"
        start_marker.id = 0
        start_marker.action = Marker.DELETE
        self.start_marker_pub.publish(start_marker)
        goal_marker = Marker()
        goal_marker.header.frame_id = "map"
        goal_marker.header.stamp = rospy.Time.now()
        goal_marker.ns = "start_goal_markers"
        goal_marker.id = 1
        goal_marker.action = Marker.DELETE
        self.goal_marker_pub.publish(goal_marker)

    def teleport_robot(self, pose):
        # Teleport robot to given pose
        try:
            rospy.loginfo(f"[EXEC] Target pose: x={pose.position.x:.2f}, y={pose.position.y:.2f}")
            current_teleport_check = self.get_model_state(
                model_name='turtlebot3_burger',
                relative_entity_name='map'
            )
            rospy.loginfo(f"[EXEC] Current pose before teleport: "
                          f"x={current_teleport_check.pose.position.x:.2f}, y={current_teleport_check.pose.position.y:.2f}")
            state = ModelState()
            state.model_name = 'turtlebot3_burger'
            state.pose = pose
            state.reference_frame = 'map'
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
            after_teleport_check = self.get_model_state(
                model_name='turtlebot3_burger',
                relative_entity_name='map'
            )
            rospy.loginfo(f"[EXEC] Pose after teleport: "
                          f"x={after_teleport_check.pose.position.x:.2f}, y={after_teleport_check.pose.position.y:.2f}")
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
        rospy.loginfo(f"[PERF_LOG] event=EXECUTOR_ROS_INTERRUPT, timestamp={rospy.Time.now().to_sec():.3f}")
        pass
    finally:
        rospy.loginfo(f"[PERF_LOG] event=EXECUTOR_SHUTDOWN, timestamp={rospy.Time.now().to_sec():.3f}")