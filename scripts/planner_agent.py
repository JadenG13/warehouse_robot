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
        rospy.init_node('planner_agent')
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.map_info = None
        self.occupancy = []

        self.world_name = rospy.get_param("~world_name")
        self.config = rospy.get_param(self.world_name)
        self.cell_size = self.config["cell_size"]

        # Wait for gazebo service
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        # Expose the new GetPath service
        self.srv = rospy.Service('get_path', GetPath, self.handle_get_path)
        rospy.loginfo("[Planner] Ready.")

    def map_callback(self, msg):
        self.map_info = msg.info
        self.occupancy = list(msg.data)

    def _yaw_to_quaternion(self, yaw):
        """Convert yaw angle to geometry_msgs/Quaternion"""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = np.sin(yaw/2)
        q.w = np.cos(yaw/2)
        return q

    def handle_get_path(self, req):
        # 1) Lookup goal cell        
        goal_cell = self.config["locations"][req.goal_name]
        goal = tuple(goal_cell)
        
        # 2) Get current pose from Gazebo
        try:
            model_state = self.get_model_state(model_name='turtlebot3_burger', relative_entity_name='map')
            if not model_state.success:
                rospy.logerr("[Planner] Failed to get robot state from Gazebo")
                return GetPathResponse()
                
            # Convert to grid coordinates
            sx = int(round((model_state.pose.position.x - self.map_info.origin.position.x) / self.cell_size))
            sy = int(round((model_state.pose.position.y - self.map_info.origin.position.y) / self.cell_size))
            
            # Extract current yaw from quaternion
            quat = [
                model_state.pose.orientation.x,
                model_state.pose.orientation.y,
                model_state.pose.orientation.z,
                model_state.pose.orientation.w
            ]
            _, _, start_yaw = euler_from_quaternion(quat)
            
            # Log the conversion for debugging
            rospy.loginfo(f"[Planner] World pos: ({model_state.pose.position.x}, {model_state.pose.position.y}) -> Grid: ({sx}, {sy})")
        except rospy.ServiceException as e:
            rospy.logerr(f"[Planner] Failed to get model state: {e}")
            return GetPathResponse()
        # Convert to grid orientation (0-3)
        start_th = int(np.round(start_yaw / (np.pi/2))) % 4
        start = (sx, sy, start_th)
        
        rospy.loginfo(f"[Planner] Start: {start}, Goal: {goal}")

        # 3) Plan on grid (A*)
        actions = self.astar(start, goal)
        rospy.loginfo(f"[Planner] Planned actions: {actions}")

        # 4) Build path response
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()
        
        waypoints = []
        suggested_actions = []
        descriptions = []

        # Start from current pose
        i = sx
        j = sy
        th = start_th

        # Add starting pose
        start_pose = PoseStamped()
        start_pose.header.frame_id = "map"
        start_pose.header.stamp = rospy.Time.now()
        
        # Set position using exact current position
        start_pose.pose.position = model_state.pose.position
        start_pose.pose.orientation = model_state.pose.orientation
        
        path.poses.append(start_pose)
        waypoints.append(start_pose)
        suggested_actions.append('')  # No action for start pose
        descriptions.append(f"Starting at grid ({i}, {j}), orientation: {th * 90}°")

        # Process each action
        for act in actions:
            di = 0
            dj = 0
            
            # Update position based on action and facing direction
            if act == 'F':
                if th == 0:
                    di = 1
                elif th == 1:
                    dj = 1
                elif th == 2:
                    di = -1
                else:  # th == 3
                    dj = -1
            elif act == 'B':
                if th == 0:
                    di = -1
                elif th == 1:
                    dj = -1
                elif th == 2:
                    di = 1
                else:  # th == 3
                    dj = 1
            elif act == 'L':
                th = (th + 1) % 4
            else:  # act == 'R'
                th = (th - 1) % 4

            # Create a PoseStamped for visualization and execution
            new_pose = PoseStamped()
            new_pose.header.frame_id = "map"
            new_pose.header.stamp = rospy.Time.now()
            
            # For moves, update position
            i = i + di
            j = j + dj
            
            # Set position
            new_pose.pose.position.x = self.map_info.origin.position.x + i * self.cell_size
            new_pose.pose.position.y = self.map_info.origin.position.y + j * self.cell_size
            new_pose.pose.position.z = 0.0

            # Set orientation (cardinal directions)
            yaw = th * np.pi/2  # Convert grid orientation to radians
            new_pose.pose.orientation = self._yaw_to_quaternion(yaw)
            
            # Add to path and waypoints
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

    def astar(self, start, goal):
        """A* search returning sequence of grid actions"""
        # Convert goal to just x,y for easier comparison
        goal_pos = goal[:2]
        
        # Priority queue of nodes to explore: (f-score, h-score, position, actions)
        start_h = abs(goal_pos[0] - start[0]) + abs(goal_pos[1] - start[1])
        frontier = [(start_h, start_h, start, [])]
        heapq.heapify(frontier)
        
        # Keep track of best paths and scores
        came_from = {start: None}
        g_score = {start: 0}
        
        while frontier:
            _, _, current, actions = heapq.heappop(frontier)
            current_pos = current[:2]
            
            # Check if we've reached the goal
            if current_pos == goal_pos:
                return actions
                
            # For each possible action
            for act in ['F', 'B', 'L', 'R']:
                i, j, th = current
                di, dj = 0, 0
                new_th = th
                
                # Calculate resulting position and orientation
                if act == 'F':
                    if th == 0:
                        di = 1
                    elif th == 1:
                        dj = 1
                    elif th == 2:
                        di = -1
                    else:  # th == 3
                        dj = -1
                elif act == 'B':
                    if th == 0:
                        di = -1
                    elif th == 1:
                        dj = -1
                    elif th == 2:
                        di = 1
                    else:  # th == 3
                        dj = 1
                elif act == 'L':
                    new_th = (th + 1) % 4
                else:  # act == 'R'
                    new_th = (th - 1) % 4
                    
                # Calculate new state
                new_i = i + di
                new_j = j + dj
                neighbor = (new_i, new_j, new_th)
                
                # Check if move would hit an obstacle or boundary
                if act in ['F', 'B']:
                    map_i = new_i
                    map_j = new_j
                    
                    # Skip if out of bounds
                    if map_i < 0 or map_i >= self.map_info.width or \
                       map_j < 0 or map_j >= self.map_info.height:
                        continue
                        
                    # Skip if hitting obstacle (>50 means occupied)
                    cell_index = map_j * self.map_info.width + map_i
                    if cell_index >= len(self.occupancy) or \
                       self.occupancy[cell_index] > 50:
                        continue
                
                # Calculate new path score (1 for moves, 0.5 for turns)
                move_cost = 1.0 if act in ['F', 'B'] else 0.5
                tentative_g = g_score[current] + move_cost
                
                # If we haven't been here before, or this is a better path
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Record this path
                    came_from[neighbor] = (current, act)
                    g_score[neighbor] = tentative_g
                    
                    # Calculate new f-score and add to frontier
                    h = abs(goal_pos[0] - new_i) + abs(goal_pos[1] - new_j)
                    f = tentative_g + h
                    new_actions = actions + [act]
                    heapq.heappush(frontier, (f, h, neighbor, new_actions))
        
        # No path found
        return None

    def spin(self):
        rospy.spin()

if __name__=='__main__':
    PlannerAgent().spin()
