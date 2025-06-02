#!/usr/bin/env python3
import rospy, heapq
import numpy as np
from warehouse_robot.srv import GetPath, GetPathResponse
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
# from gazebo_msgs.srv import GetModelState #  was correct, this is not used by planner
from tf.transformations import euler_from_quaternion
import time # For timing blocks

# Helper to format pose for logging
# Returns a string with x, y, and heading in degrees
# Accepts either a geometry_msgs.Pose or a dict with x, y, theta
def format_pose_for_log(pose_obj):
    if hasattr(pose_obj, 'orientation') and all(hasattr(pose_obj.orientation, attr) for attr in ['x', 'y', 'z', 'w']):
        try:
            _, _, yaw = euler_from_quaternion([pose_obj.orientation.x, pose_obj.orientation.y, pose_obj.orientation.z, pose_obj.orientation.w])
            return f"x:{pose_obj.position.x:.2f}_y:{pose_obj.position.y:.2f}_th:{np.degrees(yaw):.0f}"
        except Exception: # pylint: disable=broad-except
            # Fallback if quaternion is malformed for some reason
            return f"x:{pose_obj.position.x:.2f}_y:{pose_obj.position.y:.2f}_th:ERR_CONVERTING_QUAT"
    elif isinstance(pose_obj, dict) and all(k in pose_obj for k in ['x','y','theta']): # For dict style pose
        return f"x:{pose_obj['x']:.2f}_y:{pose_obj['y']:.2f}_th:{np.degrees(pose_obj['theta']):.0f}"
    return "POSE_FORMAT_ERROR"

class PlannerAgent:
    def __init__(self):
        # Initialize node and parameters
        rospy.init_node('planner_agent')
        self.node_start_time = rospy.Time.now().to_sec()
        rospy.loginfo(f"[PERF_LOG] event=PLANNER_INIT, timestamp={self.node_start_time:.3f}")

        # Subscribe to global costmap
        rospy.Subscriber(
            '/move_base/global_costmap/costmap',
            OccupancyGrid,
            self.global_costmap_callback
        )
        self.global_costmap = None
        self.global_costmap_info = None

        # Get map configuration and cell size
        self.world_name = rospy.get_param("~world_name")
        self.config = rospy.get_param(self.world_name)
        self.cell_size = self.config["cell_size"]

        # Wait for costmap to be available
        rospy.loginfo("[Planner] Waiting for costmap...")
        wait_for_costmap_start_time = time.monotonic()
        while self.global_costmap is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        wait_for_costmap_duration = time.monotonic() - wait_for_costmap_start_time
        if self.global_costmap is not None:
            rospy.loginfo(f"[Planner] Got initial costmap. Resolution: {self.global_costmap_info.resolution if self.global_costmap_info else 'N/A - Info Missing'}")
            rospy.loginfo(f"[PERF_LOG] event=PLANNER_COSTMAP_READY, timestamp={rospy.Time.now().to_sec():.3f}, wait_duration_sec={wait_for_costmap_duration:.3f}")
        else:
            rospy.logwarn("[Planner] Shutting down before costmap was received.")
            rospy.loginfo(f"[PERF_LOG] event=PLANNER_COSTMAP_TIMEOUT_OR_SHUTDOWN, timestamp={rospy.Time.now().to_sec():.3f}, wait_duration_sec={wait_for_costmap_duration:.3f}")

        # Expose the GetPath service
        self.srv = rospy.Service('get_path', GetPath, self.handle_get_path)
        rospy.loginfo("[Planner] Ready.")

    def global_costmap_callback(self, msg):
        # Store incoming global costmap data
        is_first_time = self.global_costmap is None
        self.global_costmap = list(msg.data)
        self.global_costmap_info = msg.info
        if is_first_time and self.global_costmap_info:
             rospy.loginfo(f"[PERF_LOG] event=PLANNER_FIRST_COSTMAP_DATA_RECEIVED, timestamp={rospy.Time.now().to_sec():.3f}, width={msg.info.width}, height={msg.info.height}, res={msg.info.resolution:.3f}")
        # Count non-zero cells to check if map has obstacles
        occupied = sum(1 for cell in msg.data if cell > 0)
        # Optional: log occupied cell count for debugging

    def _yaw_to_quaternion(self, yaw):
        # Convert yaw angle to geometry_msgs/Quaternion
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = np.sin(yaw / 2)
        q.w = np.cos(yaw / 2)
        return q

    def handle_get_path(self, req):
        # Handle path planning requests
        handler_start_time_ros = rospy.Time.now().to_sec()
        rospy.loginfo(f"[PERF_LOG] event=PLANNER_GET_PATH_REQUEST_RECEIVED, timestamp={handler_start_time_ros:.3f}, start_pose={format_pose_for_log(req.start_pose)}, goal_pose={format_pose_for_log(req.goal_pose)}")

        # For logging even in case of early failure due to no costmap
        start_x_grid, start_y_grid, start_th_grid = -1, -1, -1
        goal_x_grid, goal_y_grid, goal_th_grid = -1, -1, -1

        try:
            if not self.global_costmap_info or not self.global_costmap: # Check both for safety before use
                rospy.logerr("[Planner] No global_costmap_info or global_costmap data available for planning.")
                planning_duration_sec = rospy.Time.now().to_sec() - handler_start_time_ros
                rospy.loginfo(f"[PERF_LOG] event=PLANNER_GET_PATH_FAIL, timestamp={rospy.Time.now().to_sec():.3f}, reason='No costmap data', planning_duration_sec={planning_duration_sec:.3f}")
                return GetPathResponse(path=Path(), waypoints=[], suggested_actions=[], descriptions=[]) #  empty response structure

            # Convert start pose to grid coordinates
            start_x_grid = int(round((req.start_pose.position.x - self.global_costmap_info.origin.position.x) / self.global_costmap_info.resolution))
            start_y_grid = int(round((req.start_pose.position.y - self.global_costmap_info.origin.position.y) / self.global_costmap_info.resolution))

            # Get start orientation
            start_quat = [
                req.start_pose.orientation.x,
                req.start_pose.orientation.y,
                req.start_pose.orientation.z,
                req.start_pose.orientation.w
            ]
            _, _, start_yaw = euler_from_quaternion(start_quat)
            start_th_grid = int(np.round(start_yaw / (np.pi/2))) % 4

            # Convert goal pose to grid coordinates
            goal_x_grid = int(round((req.goal_pose.position.x - self.global_costmap_info.origin.position.x) / self.global_costmap_info.resolution))
            goal_y_grid = int(round((req.goal_pose.position.y - self.global_costmap_info.origin.position.y) / self.global_costmap_info.resolution))

            # Get goal orientation
            goal_quat = [
                req.goal_pose.orientation.x,
                req.goal_pose.orientation.y,
                req.goal_pose.orientation.z,
                req.goal_pose.orientation.w
            ]
            _, _, goal_yaw = euler_from_quaternion(goal_quat)
            goal_th_grid = int(np.round(goal_yaw / (np.pi/2))) % 4

            rospy.loginfo(f"[Planner] Planning path from ({start_x_grid}, {start_y_grid}, {start_th_grid}°) to ({goal_x_grid}, {goal_y_grid}, {goal_th_grid}°)") #  log

            # Run A* path planning
            astar_call_start_time = time.monotonic()
            actions = self.astar((start_x_grid, start_y_grid, start_th_grid), (goal_x_grid, goal_y_grid, goal_th_grid))
            astar_call_duration_sec = time.monotonic() - astar_call_start_time
            rospy.loginfo(f"[PERF_LOG] event=PLANNER_ASTAR_CALL_END, timestamp={rospy.Time.now().to_sec():.3f}, astar_duration_sec={astar_call_duration_sec:.4f}, success={actions is not None}, num_actions={len(actions) if actions else 0}")
            rospy.loginfo(f"[Planner] Planned actions: {actions}") #  log

            # If no path found, return empty response
            if actions is None:
                rospy.logwarn("[Planner] No path found") #  log
                empty_path = Path()
                empty_path.header.frame_id = "map"
                empty_path.header.stamp = rospy.Time.now()
                planning_duration_sec = rospy.Time.now().to_sec() - handler_start_time_ros
                rospy.loginfo(f"[PERF_LOG] event=PLANNER_GET_PATH_FAIL, timestamp={rospy.Time.now().to_sec():.3f}, reason='A* found no path', planning_duration_sec={planning_duration_sec:.3f}, start_grid=({start_x_grid},{start_y_grid},{start_th_grid}), goal_grid=({goal_x_grid},{goal_y_grid},{goal_th_grid})")
                return GetPathResponse(
                    path=empty_path,
                    waypoints=[],
                    suggested_actions=[],
                    descriptions=[]
                ) #  response

            # Build path response nav_msgs/Path
            path = Path() #  variable name
            path.header.frame_id = "map"
            path.header.stamp = rospy.Time.now()

            # stored PoseStamped objects for each step
            waypoints = []
            # stored actions taken (F, B, L, R)
            suggested_actions = []
            # written descriptions for step
            descriptions = []

            # start from current pose and header in cells
            i = start_x_grid # Use grid versions for clarity from here
            j = start_y_grid
            th = start_th_grid

            # add starting pose - grid-aligned like other waypoints
            start_pose_wp = PoseStamped() # Renamed to avoid confusion with req.start_pose
            start_pose_wp.header.frame_id = "map"
            start_pose_wp.header.stamp = rospy.Time.now()

            # Set position
            start_pose_wp.pose = req.start_pose #  assignment

            # Convert grid coordinates to world coordinates
            start_pose_wp.pose.position.x = self.global_costmap_info.origin.position.x + start_x_grid * self.global_costmap_info.resolution #  logic
            start_pose_wp.pose.position.y = self.global_costmap_info.origin.position.y + start_y_grid * self.global_costmap_info.resolution #  logic

            path.poses.append(start_pose_wp)
            waypoints.append(start_pose_wp)
            suggested_actions.append('')  # No action for start pose
            descriptions.append(f"Starting at grid ({i}, {j}), orientation: {th * 90}°") #  description

            path_length_meters = 0.0 # For logging metric

            # Process each action (F, L, R)
            for act in actions:
                # Set change in x and y directions to 0
                di = 0
                dj = 0

                # Update position based on action and facing direction
                if act == 'F':
                    if th == 0:  # Facing east
                        di = 1
                    elif th == 1:  # Facing north
                        dj = 1
                    elif th == 2:  # Facing west
                        di = -1
                    else:  # Facing south
                        dj = -1
                    path_length_meters += self.global_costmap_info.resolution # Accumulate length for F moves
                # Turning
                elif act == 'L':
                    th = (th + 1) % 4  # Turn left 90°
                else:  # 'R'
                    th = (th - 1 + 4) % 4  # Turn right 90° (ensure positive modulo,  was (th-1)%4 which is fine for Python)

                # Create a PoseStamped for visualization and execution
                new_pose = PoseStamped()
                new_pose.header.frame_id = "map"
                new_pose.header.stamp = rospy.Time.now()

                # Forward movement, update grid position
                i = i + di
                j = j + dj

                # Convert grid coordinates to world coordinates
                new_pose.pose.position.x = self.global_costmap_info.origin.position.x + i * self.global_costmap_info.resolution
                new_pose.pose.position.y = self.global_costmap_info.origin.position.y + j * self.global_costmap_info.resolution
                new_pose.pose.position.z = 0.0  # Ground height

                # Set orientation
                yaw = th * np.pi / 2  # Convert grid orientation to radians
                new_pose.pose.orientation = self._yaw_to_quaternion(yaw)

                # Add to paths and descriptions
                path.poses.append(new_pose)
                waypoints.append(new_pose)
                suggested_actions.append(act)
                descriptions.append(f"Move to grid ({i}, {j}), orientation: {th * 90}°") #  description

            rospy.loginfo(f"[Planner] Generated path with {len(waypoints)} waypoints") #  log
            planning_duration_sec = rospy.Time.now().to_sec() - handler_start_time_ros
            rospy.loginfo(f"[PERF_LOG] event=PLANNER_GET_PATH_SUCCESS, timestamp={rospy.Time.now().to_sec():.3f}, planning_duration_sec={planning_duration_sec:.3f}, num_waypoints={len(waypoints)}, num_actions={len(actions)}, path_length_meters={path_length_meters:.3f}, start_grid=({start_x_grid},{start_y_grid},{start_th_grid}), goal_grid=({goal_x_grid},{goal_y_grid},{goal_th_grid})")

            return GetPathResponse(
                path=path, # 
                waypoints=waypoints,
                suggested_actions=suggested_actions,
                descriptions=descriptions
            ) #  response

        except Exception as e:
            rospy.logerr(f"[Planner] Path planning failed: {str(e)}") #  log
            empty_path = Path()
            empty_path.header.frame_id = "map"
            empty_path.header.stamp = rospy.Time.now()
            planning_duration_sec = rospy.Time.now().to_sec() - handler_start_time_ros
            rospy.loginfo(f"[PERF_LOG] event=PLANNER_GET_PATH_FAIL, timestamp={rospy.Time.now().to_sec():.3f}, reason='Exception: {str(e)}', planning_duration_sec={planning_duration_sec:.3f}, start_grid=({start_x_grid},{start_y_grid},{start_th_grid}), goal_grid=({goal_x_grid},{goal_y_grid},{goal_th_grid})")
            return GetPathResponse(
                path=empty_path,
                waypoints=[],
                suggested_actions=[],
                descriptions=[]
            ) #  response

    def check_cell_safety(self, map_i, map_j):
        # Return True if cell (map_i, map_j) is occupied or out of bounds
        if not self.global_costmap_info or not self.global_costmap:
            return True
        if (0 <= map_i < self.global_costmap_info.width and 0 <= map_j < self.global_costmap_info.height):
            cell_index = map_j * self.global_costmap_info.width + map_i
            if cell_index < 0 or cell_index >= len(self.global_costmap):
                return True
            cost = self.global_costmap[cell_index]
            if cost > 0:
                return True
        else:
            return True
        return False

    def astar(self, start, goal):
        # A* search returning sequence of grid actions
        goal_pos = goal[: 2]
        start_h = abs(goal_pos[0] - start[0]) + abs(goal_pos[1] - start[1])
        rot_diff = abs((goal[2] - start[2]) % 4)
        rot_cost = min(rot_diff, 4 - rot_diff) * 0.5
        start_h += rot_cost
        frontier = [(start_h, start_h, start, [])]
        heapq.heapify(frontier)
        came_from = {start: None}
        g_score = {start: 0}
        while frontier:
            _, _, current, actions = heapq.heappop(frontier)
            current_pos = current[:2]
            if current_pos == goal_pos and current[2] == goal[2]:
                return actions
            for act in ['F', 'L', 'R']:
                i, j, th = current
                di, dj = 0, 0
                new_th = th
                if act == 'F':
                    if th == 0:
                        di = 1
                    elif th == 1:
                        dj = 1
                    elif th == 2:
                        di = -1
                    else:
                        dj = -1
                elif act == 'L':
                    new_th = (th + 1) % 4
                else:
                    new_th = (th - 1 + 4) % 4
                new_i = i + di
                new_j = j + dj
                neighbor = (new_i, new_j, new_th)
                if act == 'F':
                    map_i = new_i
                    map_j = new_j
                    if self.check_cell_safety(map_i, map_j):
                        continue
                move_cost = 1.0 if act == 'F' else 0.5
                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = (current, act)
                    g_score[neighbor] = tentative_g
                    h = abs(goal_pos[0] - new_i) + abs(goal_pos[1] - new_j)
                    rot_diff_h = abs((goal[2] - new_th) % 4)
                    rot_cost_h = min(rot_diff_h, 4 - rot_diff_h) * 0.5
                    h += rot_cost_h
                    f = tentative_g + h
                    new_actions = actions + [act]
                    heapq.heappush(frontier, (f, h, neighbor, new_actions))
        return None

    def spin(self):
        rospy.spin()

if __name__=='__main__':
    try:
        PlannerAgent().spin()
    except rospy.ROSInterruptException:
        rospy.loginfo(f"[PERF_LOG] event=PLANNER_ROS_INTERRUPT, timestamp={rospy.Time.now().to_sec():.3f}")
    finally:
        rospy.loginfo(f"[PERF_LOG] event=PLANNER_SHUTDOWN, timestamp={rospy.Time.now().to_sec():.3f}")