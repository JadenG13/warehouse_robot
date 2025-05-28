# planner_agent.py
#!/usr/bin/env python3
import rospy, heapq
import numpy as np
from warehouse_robot.srv import GetPath, GetPathResponse
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from gazebo_msgs.srv import GetModelState
from tf.transformations import euler_from_quaternion

class PlannerAgent:
    def __init__(self):
        # initialise node
        rospy.init_node('planner_agent')
        
        # Subscribe to global costmap
        rospy.Subscriber(
            '/move_base/global_costmap/costmap',
            OccupancyGrid,
            self.global_costmap_callback
        )
        self.global_costmap = None
        self.global_costmap_info = None

        # get map configuration and cell size (resolution) of map grid
        self.world_name = rospy.get_param("~world_name")
        self.config = rospy.get_param(self.world_name)
        self.cell_size = self.config["cell_size"]

        # Wait for costmap to be available
        while self.global_costmap is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
            
        # Expose the GetPath service
        self.srv = rospy.Service('get_path', GetPath, self.handle_get_path)
        rospy.loginfo("[Planner] Ready.")

    def global_costmap_callback(self, msg):
        """Store incoming global costmap data"""
        self.global_costmap = list(msg.data)
        self.global_costmap_info = msg.info
        # Count non-zero cells to check if map has obstacles
        occupied = sum(1 for cell in msg.data if cell > 0)
        
    # convert robot heading to a quarternion 
    def _yaw_to_quaternion(self, yaw):
        # convert yaw angle to geometry_msgs/Quaternion
        q = Quaternion()
        # encodes rotation axis
        q.x = 0.0
        q.y = 0.0
        # only around z axis 
        q.z = np.sin(yaw / 2)
        # encodes rotation angle
        q.w = np.cos(yaw / 2)
        return q

    def handle_get_path(self, req):
        """Handle path planning requests"""
        try:
            # Convert start pose to grid coordinates
            start_x = int(round((req.start_pose.position.x - self.global_costmap_info.origin.position.x) / self.global_costmap_info.resolution))
            start_y = int(round((req.start_pose.position.y - self.global_costmap_info.origin.position.y) / self.global_costmap_info.resolution))
            
            # Get start orientation
            start_quat = [
                req.start_pose.orientation.x,
                req.start_pose.orientation.y,
                req.start_pose.orientation.z,
                req.start_pose.orientation.w
            ]
            _, _, start_yaw = euler_from_quaternion(start_quat)
            start_th = int(np.round(start_yaw / (np.pi/2))) % 4
            
            # Convert goal pose to grid coordinates 
            goal_x = int(round((req.goal_pose.position.x - self.global_costmap_info.origin.position.x) / self.global_costmap_info.resolution))
            goal_y = int(round((req.goal_pose.position.y - self.global_costmap_info.origin.position.y) / self.global_costmap_info.resolution))
            
            # Get goal orientation
            goal_quat = [
                req.goal_pose.orientation.x,
                req.goal_pose.orientation.y,
                req.goal_pose.orientation.z,
                req.goal_pose.orientation.w
            ]
            _, _, goal_yaw = euler_from_quaternion(goal_quat)
            goal_th = int(np.round(goal_yaw / (np.pi/2))) % 4

            rospy.loginfo(f"[Planner] Planning path from ({start_x}, {start_y}, {start_th}°) to ({goal_x}, {goal_y}, {goal_th}°)")

            # Run A* path planning 
            actions = self.astar((start_x, start_y, start_th), (goal_x, goal_y, goal_th))
            rospy.loginfo(f"[Planner] Planned actions: {actions}")

            # If no path found, return empty response
            if actions is None:
                rospy.logwarn("[Planner] No path found")
                empty_path = Path()
                empty_path.header.frame_id = "map"
                empty_path.header.stamp = rospy.Time.now()
                return GetPathResponse(
                    path=empty_path,
                    waypoints=[],
                    suggested_actions=[],
                    descriptions=[]
                )

            # Build path response nav_msgs/Path
            path = Path()
            path.header.frame_id = "map"
            path.header.stamp = rospy.Time.now()
            
            # stored PoseStamped objects for each step
            waypoints = []
            # stored actions taken (F, B, L, R)
            suggested_actions = []
            # written descriptions for step
            descriptions = []

            # start from current pose and header in cells
            i = start_x
            j = start_y 
            th = start_th

            # add starting pose - grid-aligned like other waypoints
            start_pose = PoseStamped()
            start_pose.header.frame_id = "map"
            start_pose.header.stamp = rospy.Time.now()
            
            # Set position 
            start_pose.pose = req.start_pose
            
            # Convert grid coordinates to world coordinates
            start_pose.pose.position.x = self.global_costmap_info.origin.position.x + start_x * self.global_costmap_info.resolution
            start_pose.pose.position.y = self.global_costmap_info.origin.position.y + start_y * self.global_costmap_info.resolution
            
            path.poses.append(start_pose)
            waypoints.append(start_pose)
            suggested_actions.append('')  # No action for start pose
            descriptions.append(f"Starting at grid ({i}, {j}), orientation: {th * 90}°")

            # Process each action (F, B, L, R)
            for act in actions:
                # Set change in x and y directions to 0
                di = 0
                dj = 0
                
                # Update position based on action and facing direction
                # Move forward/backwards
                if act == 'F':
                    if th == 0:  # Facing east
                        di = 1
                    elif th == 1:  # Facing north
                        dj = 1
                    elif th == 2:  # Facing west
                        di = -1
                    else:  # Facing south
                        dj = -1

                # Turning
                elif act == 'L':
                    th = (th + 1) % 4  # Turn left 90°
                else:  # 'R'
                    th = (th - 1) % 4  # Turn right 90°

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
                descriptions.append(f"Move to grid ({i}, {j}), orientation: {th * 90}°")

            rospy.loginfo(f"[Planner] Generated path with {len(waypoints)} waypoints")

            return GetPathResponse(
                path=path,
                waypoints=waypoints,
                suggested_actions=suggested_actions,
                descriptions=descriptions
            )
            
        except Exception as e:
            rospy.logerr(f"[Planner] Path planning failed: {str(e)}")
            empty_path = Path()
            empty_path.header.frame_id = "map"
            empty_path.header.stamp = rospy.Time.now()
            return GetPathResponse(
                path=empty_path,
                waypoints=[],
                suggested_actions=[],
                descriptions=[]
            )

    def check_cell_safety(self, map_i, map_j):    
        """Check if the target cell (x,y) is occupied"""
        # rospy.loginfo(f"[Planner] Checking cell ({map_i}, {map_j}) for safety")
        if (0 <= map_i < self.global_costmap_info.width and 
            0 <= map_j < self.global_costmap_info.height):
            cell_index = map_j * self.global_costmap_info.width + map_i
            if cell_index < 0 or cell_index >= len(self.global_costmap):
                # rospy.logwarn(f"[Planner] Invalid index {cell_index} for costmap at ({map_i}, {map_j})")
                return True
            cost = self.global_costmap[cell_index]
            if cost > 0:  # ROS costmaps typically use 0-100 range, with >= 50 being obstacles
                # rospy.loginfo(f"[Planner] Detected obstacle at ({map_i}, {map_j}) with cost {cost}")
                return True
        else:
            # Treat out-of-bounds as occupied for safety
            # rospy.loginfo(f"[Planner] Cell ({map_i}, {map_j}) is out of bounds")
            return True
        return False
    

    # A* path finding returning sequence of actions
    def astar(self, start, goal):
        # A* search returning sequence of grid actions
        # Use goal position for distance heuristic, but keep orientation for final check
        goal_pos = goal[: 2]
        
        # Priority queue of nodes to explore: (f-score, h-score, position, actions)
        # Heuristic defined by Manhattan (taxicab) distance (admissible heuristic)
        # Plus minimum turns needed to reach goal orientation (0.5 cost per turn)
        start_h = abs(goal_pos[0] - start[0]) + abs(goal_pos[1] - start[1])
        # Add estimated rotation cost
        rot_diff = abs((goal[2] - start[2]) % 4)
        rot_cost = min(rot_diff, 4 - rot_diff) * 0.5
        start_h += rot_cost
        frontier = [(start_h, start_h, start, [])]
        heapq.heapify(frontier)
        
        # Keep track of best paths and scores
        # Parent nodes and actions
        came_from = {start: None}
        # Lowest known cost to reach each node
        g_score = {start: 0}
        
        # Main loop
        while frontier:
            # pops lowest cost node 
            _, _, current, actions = heapq.heappop(frontier)
            current_pos = current[:2]
            
            # check if we've reached the goal position and orientation
            if current_pos == goal_pos and current[2] == goal[2]:
                return actions
                
            # for each possible action
            # only try F, L, R (no backwards movement)
            for act in ['F', 'L', 'R']:
                i, j, th = current
                di, dj = 0, 0
                new_th = th
                
                # calculate resulting position and orientation
                # forward movement
                if act == 'F':
                    if th == 0:  # Facing east
                        di = 1
                    elif th == 1:  # Facing north
                        dj = 1
                    elif th == 2:  # Facing west
                        di = -1
                    else:  # Facing south
                        dj = -1

                # turning
                elif act == 'L':
                    new_th = (th + 1) % 4  # Turn left 90°
                
                # else R
                else:
                    new_th = (th - 1) % 4  # Turn right 90°
                    
                # calculate new state
                new_i = i + di
                new_j = j + dj
                neighbor = (new_i, new_j, new_th)
                
                # check if move would hit an obstacle or boundary
                if act == 'F':
                    map_i = new_i 
                    map_j = new_j
                    
                    # Check if target position and its immediate neighbors are safe
                    if self.check_cell_safety(map_i, map_j):
                        continue
                
                # calculate new path score (1 for moves, 0.5 for turns)
                move_cost = 1.0 if act == 'F' else 0.5
                tentative_g = g_score[current] + move_cost
                
                # if we haven't been here before, or this is a better path
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # record this path
                    came_from[neighbor] = (current, act)
                    g_score[neighbor] = tentative_g
                    
                    # calculate new f-score and add to frontier
                    h = abs(goal_pos[0] - new_i) + abs(goal_pos[1] - new_j)
                    # Add rotation cost to heuristic
                    rot_diff = abs((goal[2] - new_th) % 4)
                    rot_cost = min(rot_diff, 4 - rot_diff) * 0.5
                    h += rot_cost
                    f = tentative_g + h
                    new_actions = actions + [act]
                    # push into p-queue
                    heapq.heappush(frontier, (f, h, neighbor, new_actions))
                    
        # no path found
        return None

    def spin(self):
        rospy.spin()

if __name__=='__main__':
    PlannerAgent().spin()
