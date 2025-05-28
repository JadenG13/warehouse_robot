# validator_agent.py
#!/usr/bin/env python3
import rospy
import numpy as np
from warehouse_robot.srv import ValidatePath, ValidatePathResponse, CheckMovement, CheckMovementResponse
from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import OccupancyGrid
from gazebo_msgs.srv import GetModelState
from tf.transformations import euler_from_quaternion

class ValidatorAgent:
    def __init__(self):
        rospy.init_node('validator_agent')
        self.global_costmap = None
        self.costmap_info = None
        self.cell_size = None
        
        # Subscribe to global costmap
        # Subscribe to global costmap with a larger queue size to not miss updates
        self.costmap_sub = rospy.Subscriber(
            '/move_base/global_costmap/costmap',
            OccupancyGrid,
            self.costmap_callback,
            queue_size=10
        )
        
        # Wait for gazebo service to get robot position
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        # Wait for initial costmap data
        rospy.loginfo("[Validator] Waiting for costmap...")
        while self.global_costmap is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("[Validator] Got initial costmap")
        
        # Expose services
        self.validate_srv = rospy.Service('validate_path', ValidatePath, self.handle)
        self.check_srv = rospy.Service('check_movement', CheckMovement, self.check_movement)
        rospy.loginfo("[Validator] Ready.")

    def handle(self, req):
        # For each waypoint, check against a dynamic costmap or LIDAR data
        for act in req.actions:
            # placeholder: no dynamic obstacles
            pass
        return ValidatePathResponse(True)

    def spin(self):
        rospy.spin()

    def costmap_callback(self, msg):
        """Store incoming costmap data"""
        self.global_costmap = list(msg.data)
        self.costmap_info = msg.info
        self.cell_size = msg.info.resolution

    def check_movement(self, req):
        """Service handler to check if movement is safe"""
        if not self.global_costmap or not self.costmap_info:
            return CheckMovementResponse(success=False, message="No costmap data available", occupancy_grid=[False]*9)

        try:
            # Convert world coordinates to grid coordinates for current position
            robot_x = int(round((req.current_pose.position.x - self.costmap_info.origin.position.x) / self.cell_size))
            robot_y = int(round((req.current_pose.position.y - self.costmap_info.origin.position.y) / self.cell_size))
            
            # Calculate target grid coordinates
            target_x = int(round((req.target_pose.position.x - self.costmap_info.origin.position.x) / self.cell_size))
            target_y = int(round((req.target_pose.position.y - self.costmap_info.origin.position.y) / self.cell_size))
            
            rospy.loginfo(f"[Validator] Checking movement from ({robot_x}, {robot_y}) to ({target_x}, {target_y})")

            # First check if target position is blocked
            if self.is_cell_blocked(target_x, target_y):
                return CheckMovementResponse(
                    success=False,
                    message="Target position is blocked by an obstacle",
                    occupancy_grid=[True]*9 # Show as blocked
                )

            # For moves, only check cells in front of the robot
            if req.action in ['F']:
                # Get robot's orientation
                quat = [
                    req.current_pose.orientation.x,
                    req.current_pose.orientation.y,
                    req.current_pose.orientation.z,
                    req.current_pose.orientation.w
                ]
                _, _, yaw = euler_from_quaternion(quat)
                heading = int(round(yaw / (np.pi/2))) % 4  # 0:+x, 1:+y, 2:-x, 3:-y

                # Get movement direction
                dx = target_x - robot_x
                dy = target_y - robot_y
                
                # Determine number of steps to check
                steps = max(abs(dx), abs(dy))
                if steps > 0:
                    step_x = dx / steps
                    step_y = dy / steps
                    
                    # Check each intermediate position
                    for i in range(1, steps):
                        check_x = int(robot_x + step_x * i)
                        check_y = int(robot_y + step_y * i)
                        if self.is_cell_blocked(check_x, check_y):
                            rospy.loginfo(f"[Validator] Path blocked at step {i} ({check_x}, {check_y})")
                            return CheckMovementResponse(
                                success=False,
                                message=f"Path to target is blocked",
                                occupancy_grid=[True]*9 # Show as blocked
                            )

            # Get occupancy grid for response
            occupancy_grid = []
            map_section = []
            for dy in range(-1, 2):
                row = []
                for dx in range(-1, 2):
                    x, y = target_x + dx, target_y + dy
                    if (0 <= x < self.costmap_info.width and 
                        0 <= y < self.costmap_info.height):
                        idx = y * self.costmap_info.width + x
                        value = self.global_costmap[idx]
                        is_occupied = value > 0
                        occupancy_grid.append(is_occupied)
                        row.append(' X ' if is_occupied else ' _ ')
                    else:
                        row.append(' ? ')
                        occupancy_grid.append(True)  # Treat out of bounds as occupied
                map_section.append(row)
            
            # Print the 3x3 grid around target
            rospy.loginfo("Local map section around target (3x3 grid):")
            for row in map_section:
                rospy.loginfo(' '.join(row))

            return CheckMovementResponse(
                success=True,
                message="Movement is safe",
                occupancy_grid=occupancy_grid
            )

        except Exception as e:
            rospy.logerr(f"[Validator] Error checking movement: {str(e)}")
            return CheckMovementResponse(
                success=False,
                message=f"Error checking movement: {str(e)}",
                occupancy_grid=[False]*9
            )

    def is_cell_blocked(self, x, y):
        """Check if the target cell (x,y) is occupied"""
        if (0 <= x < self.costmap_info.width and 
            0 <= y < self.costmap_info.height):
            idx = y * self.costmap_info.width + x
            if idx < 0 or idx >= len(self.global_costmap):
                rospy.logwarn(f"[Validator] Invalid index {idx} for costmap at ({x}, {y})")
                return True
            cost = self.global_costmap[idx]
            if cost > 0:  # ROS costmaps typically use 0-100 range, with >= 50 being obstacles
                rospy.loginfo(f"[Validator] Detected obstacle at ({x}, {y}) with cost {cost}")
                return True
        else:
            # Treat out-of-bounds as occupied for safety
            rospy.loginfo(f"[Validator] Cell ({x}, {y}) is out of bounds")
            return True
        return False

if __name__=='__main__':
    ValidatorAgent().spin()
