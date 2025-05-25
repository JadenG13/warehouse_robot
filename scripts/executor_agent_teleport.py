#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from warehouse_robot.srv import ExecutePath, ExecutePathResponse
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class ExecutorAgentTeleport:
    def __init__(self):
        rospy.init_node('executor_agent_teleport')
        
        # Load grid cell size from your world params
        self.world_name = rospy.get_param("~world_name")
        self.config = rospy.get_param(self.world_name)
        self.cell_size = self.config["cell_size"]
        
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
        if not req.waypoints:
            rospy.logerr("[EXEC] No waypoints provided")
            return ExecutePathResponse(success=False)
            
        rospy.loginfo(f"[EXEC] Executing path with {len(req.waypoints)} waypoints")
        
        # Print all waypoints at start for debugging
        rospy.loginfo("[EXEC] Full waypoint list:")
        for i, waypoint in enumerate(req.waypoints):
            rospy.loginfo(f"[EXEC] Waypoint {i}: ({waypoint.pose.position.x:.2f}, {waypoint.pose.position.y:.2f})")
        
        for i, (waypoint, action) in enumerate(zip(req.waypoints, req.suggested_actions)):
            # Get current state
            current = self.get_model_state(model_name='turtlebot3_burger', relative_entity_name='map')
            if not current.success:
                rospy.logerr("[EXEC] Failed to get robot state")
                return ExecutePathResponse(success=False)

            # Detailed waypoint logging
            rospy.loginfo(f"\n[EXEC] === Executing waypoint {i} ===")
            rospy.loginfo(f"[EXEC] Target: ({waypoint.pose.position.x:.2f}, {waypoint.pose.position.y:.2f})")
            rospy.loginfo(f"[EXEC] Action: {action}")
            if i < len(req.descriptions) and req.descriptions[i]:
                rospy.loginfo(f"[EXEC] Description: {req.descriptions[i]}")

            # For movements ('F' or 'B'), validate distance is cell_size
            if action in ['F', 'B']:
                dx = waypoint.pose.position.x - current.pose.position.x
                dy = waypoint.pose.position.y - current.pose.position.y
                dist = math.sqrt(dx*dx + dy*dy)
                rospy.loginfo(f"[EXEC] Movement distance: {dist:.3f}m (should be {self.cell_size}m)")
                if abs(dist - self.cell_size) > 0.01:  # 1cm tolerance
                    rospy.logwarn(f"[EXEC] Movement distance {dist:.3f}m differs from cell size {self.cell_size}m")

            # Teleport to waypoint
            if not self.teleport_robot(waypoint.pose):
                rospy.logerr(f"[EXEC] Failed to teleport to waypoint {i}")
                return ExecutePathResponse(success=False)

            # Small delay between teleports
            rospy.sleep(0.5)

        rospy.loginfo("[EXEC] Path execution completed")
        return ExecutePathResponse(success=True)

    def teleport_robot(self, pose):
        """Teleport robot to given pose"""
        try:
            # Debug logging
            rospy.loginfo(f"[EXEC] Target pose: x={pose.position.x:.2f}, y={pose.position.y:.2f}")
            
            # Get current state to verify the change
            current = self.get_model_state(model_name='turtlebot3_burger', relative_entity_name='map')
            rospy.loginfo(f"[EXEC] Current pose before teleport: x={current.pose.position.x:.2f}, y={current.pose.position.y:.2f}")
            
            state = ModelState()
            state.model_name = 'turtlebot3_burger'
            state.pose = pose
            state.reference_frame = 'map'  # Using map frame consistently
            
            # Set zero velocity to ensure clean teleport
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
            
            # Verify the teleport
            after = self.get_model_state(model_name='turtlebot3_burger', relative_entity_name='map')
            rospy.loginfo(f"[EXEC] Pose after teleport: x={after.pose.position.x:.2f}, y={after.pose.position.y:.2f}")
            
            # Small delay to allow state to update
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
