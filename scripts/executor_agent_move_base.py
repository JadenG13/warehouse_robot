#!/usr/bin/env python3
import rospy
import numpy as np
import actionlib
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from warehouse_robot.srv import ExecutePath, ExecutePathResponse
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import euler_from_quaternion

class ExecutorAgentMoveBase:
    def __init__(self):
        rospy.init_node('executor_agent_move_base')
        
        # Load grid cell size from your world params
        self.world_name = rospy.get_param("~world_name")
        self.config = rospy.get_param(self.world_name)
        self.cell_size = self.config["cell_size"]
        
        # Move base parameters
        self.max_planning_retries = rospy.get_param('~max_planning_retries', 3)
        self.goal_timeout = rospy.get_param('~goal_timeout', 60.0)  # seconds
        self.waypoint_tolerance = rospy.get_param('~waypoint_tolerance', 0.05)  # meters
        
        # Initialize move_base action client
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("[EXEC] Waiting for move_base action server...")
        if not self.move_base_client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("[EXEC] Move base action server not available!")
            rospy.signal_shutdown("Move base action server not available")
            return
        
        # Set up communication
        self.execute_service = rospy.Service('execute_path', ExecutePath, self.execute_callback)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.amcl_sub = rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self.amcl_callback)

        # Initialize state variables
        self.last_amcl_pose = None
        self.last_amcl_time = None
        self.amcl_yaw = 0
        
        rospy.loginfo("[EXEC] Move base executor agent initialized")

    def amcl_callback(self, msg):
        """Store latest AMCL pose"""
        quat = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        _, _, yaw = euler_from_quaternion(quat)
        self.amcl_yaw = yaw
        self.last_amcl_pose = msg.pose.pose
        self.last_amcl_time = rospy.Time.now()

    def execute_callback(self, req):
        """Handle execution of a path defined by waypoints and suggested actions"""
        if not self.last_amcl_pose:
            rospy.logerr("[EXEC] No AMCL pose available")
            return ExecutePathResponse(success=False)
        
        if not req.waypoints:
            rospy.logerr("[EXEC] No waypoints provided")
            return ExecutePathResponse(success=False)
            
        if len(req.waypoints) != len(req.suggested_actions):
            rospy.logerr("[EXEC] Mismatch between number of waypoints and actions")
            return ExecutePathResponse(success=False)

        # Print waypoint information
        rospy.loginfo("[EXEC] Executing path with waypoints:")
        for i, waypoint in enumerate(req.waypoints):
            x = waypoint.pose.position.x 
            y = waypoint.pose.position.y
            rospy.loginfo(f"[EXEC] Waypoint {i+1}: ({x:.2f}, {y:.2f})")
            if i < len(req.descriptions) and req.descriptions[i]:
                rospy.loginfo(f"[EXEC] Description: {req.descriptions[i]}")

        # Execute each waypoint
        for i, waypoint in enumerate(req.waypoints):
            step_info = {
                'waypoint': waypoint,
                'suggested_action': req.suggested_actions[i],
                'description': req.descriptions[i] if i < len(req.descriptions) else ''
            }
            
            success = self._do_action(step_info)
            if not success:
                desc = step_info['description'] if step_info['description'] else f"step {i+1}"
                rospy.logerr(f"[EXEC] Failed to execute {desc}")
                return ExecutePathResponse(success=False)

        rospy.loginfo("[EXEC] Path execution completed successfully")
        return ExecutePathResponse(success=True)

    def _do_action(self, step_info):
        """Execute a single step using move_base for navigation"""
        waypoint = step_info['waypoint']
        desc = step_info['description'] if step_info['description'] else 'current waypoint'
        
        # Create move_base goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        
        # Copy pose from waypoint
        goal.target_pose.pose = waypoint.pose
        
        rospy.loginfo(f"[EXEC] Sending goal for {desc}")
        rospy.loginfo(f"[EXEC] Target: ({waypoint.pose.position.x:.2f}, {waypoint.pose.position.y:.2f})")
        
        # Try multiple times in case planning fails
        for attempt in range(self.max_planning_retries):
            # Send goal to move_base
            self.move_base_client.send_goal(goal)
            
            # Wait for result with timeout
            success = self.move_base_client.wait_for_result(rospy.Duration(self.goal_timeout))
            
            if success:
                state = self.move_base_client.get_state()
                if state == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo(f"[EXEC] Goal reached successfully")
                    return True
                else:
                    rospy.logwarn(f"[EXEC] Goal failed with state: {state}")
            else:
                rospy.logwarn(f"[EXEC] Goal timed out")
            
            if attempt < self.max_planning_retries - 1:
                rospy.loginfo(f"[EXEC] Retrying... (attempt {attempt + 2}/{self.max_planning_retries})")
                rospy.sleep(1.0)  # Wait a bit before retrying
                
                # Cancel any existing goal
                self.move_base_client.cancel_goal()
                
                # Send a stop command to ensure robot is stationary
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                rospy.sleep(0.5)
        
        rospy.logerr(f"[EXEC] Failed to reach goal after {self.max_planning_retries} attempts")
        return False

if __name__ == '__main__':
    try:
        ExecutorAgentMoveBase()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
