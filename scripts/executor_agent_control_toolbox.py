#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from warehouse_robot.srv import ExecutePath, ExecutePathResponse, ValidatePath
import nav_msgs.msg
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion

class ExecutorAgent:
    def __init__(self):
        rospy.init_node('executor_agent')
        
        # Load grid cell size from your world params
        self.world_name = rospy.get_param("~world_name")
        self.config = rospy.get_param(self.world_name)
        self.cell_size = self.config["cell_size"]
        
        # Movement parameters
        self.linear_speed = rospy.get_param('~linear_speed', 0.2)
        self.angular_speed = rospy.get_param('~angular_speed', 0.5)
        
        # Initialize PID parameters
        # Position control gains - increased for faster response
        self.Kp_pos = rospy.get_param('~Kp_pos', 0.5)  # Increased for faster response
        self.Ki_pos = rospy.get_param('~Ki_pos', 0.1)  # Increased for better steady-state
        self.Kd_pos = rospy.get_param('~Kd_pos', 0.2)  # Increased for better damping
        self.integral_limit = rospy.get_param('~integral_limit', 0.3)  # Increased for stronger integral action

        # Rotation control gains - increased for faster response
        self.Kp = rospy.get_param('~Kp', 1.0)  # Increased from 0.01
        self.Kd = rospy.get_param('~Kd', 0.5)  # Increased from 0.00005

        # Publishers for debugging PID values
        self.pos_error_pub = rospy.Publisher('pos_pid/error', Float64, queue_size=1)
        self.rot_error_pub = rospy.Publisher('rot_pid/error', Float64, queue_size=1)

        # Initialize PID state
        self.pos_integral = 0.0
        self.rot_integral = 0.0
        self.last_pos_error = 0.0
        self.last_rot_error = 0.0
        self.last_time = None

        # Set up communication
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)
        self.amcl_sub = rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self.amcl_callback)
        self.execute_service = rospy.Service('execute_path', ExecutePath, self.execute_callback)

        # Initialize state variables
        self.last_odom = None
        self.last_odom_time = None
        self.odom_yaw = 0
        
        self.current_direction = 1
        
        # Initialize odometry correction values
        self.odom_correction_x = 0.0
        self.odom_correction_y = 0.0
        self.odom_correction_yaw = 0.0
        
        rospy.loginfo("[EXEC] Executor agent initialized")

    def execute_callback(self, req):
        """Handle execution of a path defined by waypoints and suggested actions"""
        if not self.last_odom:
            rospy.logerr("[EXEC] No odometry data available")
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

        # Reset PID states for new path  
        self.pos_integral = 0.0
        self.rot_integral = 0.0
        self.last_pos_error = 0.0 
        self.last_rot_error = 0.0
        self.last_time = None

        for i, waypoint in enumerate(req.waypoints):
            # Create step info dict for logging
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

    def amcl_callback(self, msg):
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
        
        # Calculate correction between AMCL and odometry
        if self.last_odom:
            self.odom_correction_x = msg.pose.pose.position.x - self.last_odom.pose.pose.position.x
            self.odom_correction_y = msg.pose.pose.position.y - self.last_odom.pose.pose.position.y
            self.odom_correction_yaw = self._normalize_angle(yaw - self.odom_yaw)
            rospy.logdebug(f"[EXEC] AMCL correction: x={self.odom_correction_x:.3f}, y={self.odom_correction_y:.3f}, yaw={np.degrees(self.odom_correction_yaw):.1f}°")

    def odom_callback(self, msg):
        self.last_odom = msg
        self.last_odom_time = rospy.Time.now()
        
        # Extract yaw from quaternion
        quat = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        _, _, yaw = euler_from_quaternion(quat)
        self.odom_yaw = yaw

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _do_action(self, step_info):
        """Execute a single step using waypoint-based navigation"""
        waypoint = step_info['waypoint']
        
        # Wait for initial AMCL pose to ensure accurate starting position
        start_time = rospy.Time.now()
        while not hasattr(self, 'last_amcl_pose'):
            if (rospy.Time.now() - start_time).to_sec() > 2.0:
                rospy.logwarn("[EXEC] No AMCL pose received, falling back to odometry")
                break
            rospy.sleep(0.1)

        # Extract target pose
        target_x = waypoint.pose.position.x
        target_y = waypoint.pose.position.y
        
        # Extract target yaw using euler_from_quaternion
        quat = [
            waypoint.pose.orientation.x,
            waypoint.pose.orientation.y,
            waypoint.pose.orientation.z,
            waypoint.pose.orientation.w
        ]
        _, _, target_yaw = euler_from_quaternion(quat)
        
        # Calculate the absolute minimum rotation needed
        current_yaw = self.odom_yaw
        total_rotation = self._normalize_angle(target_yaw - current_yaw)
        
        rospy.loginfo(f"[EXEC] Moving to waypoint: ({target_x:.2f}, {target_y:.2f}, {np.degrees(target_yaw):.1f}°)")
        rospy.loginfo(f"[EXEC] Current yaw: {np.degrees(current_yaw):.1f}°, Required rotation: {np.degrees(total_rotation):.1f}°")
        if step_info['description']:
            rospy.loginfo(f"[EXEC] Step description: {step_info['description']}")

        # Make sure we start from a complete stop
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        rospy.sleep(0.5)  # Wait for robot to stop completely

        # Rotate directly to target orientation
        while not rospy.is_shutdown():
            current_yaw = self.odom_yaw
            heading_error = self._normalize_angle(target_yaw - current_yaw)
            
            # Check if we've achieved target orientation within tolerance
            if abs(heading_error) < np.radians(2.0):  # 2 degree tolerance
                # Stop rotation when aligned
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                rospy.sleep(0.2)  # Short delay to ensure rotation has stopped
                rospy.loginfo(f"[EXEC] Aligned to target orientation: {np.degrees(current_yaw):.1f}°")
                break

            # Apply control for rotation
            cmd = Twist()
            # Use the minimum rotation direction
            cmd.angular.z = self.Kp * heading_error
            if abs(cmd.angular.z) > self.angular_speed:
                cmd.angular.z = self.angular_speed * (-1 if heading_error < 0 else 1)
            self.cmd_vel_pub.publish(cmd)
            rospy.Rate(10).sleep()

        # Store starting position for progress tracking
        self._movement_start = (
            self.last_odom.pose.pose.position.x,
            self.last_odom.pose.pose.position.y
        )

        # Then move to target position
        while not rospy.is_shutdown():
            if not self.last_odom:  # Safety check
                rospy.logerr("[EXEC] Lost odometry data")
                return False
                
            # Get current position using corrected odometry
            current_x = self.last_odom.pose.pose.position.x + self.odom_correction_x
            current_y = self.last_odom.pose.pose.position.y + self.odom_correction_y
            current_yaw = self._normalize_angle(self.odom_yaw + self.odom_correction_yaw)
                # rospy.logdebug(f"[EXEC] Using odom position: ({current_x:.2f}, {current_y:.2f})")
                
            rospy.loginfo(f"[EXEC] cur pos: amcl, odom: ({self.last_amcl_pose.position.x:.2f}, {self.last_amcl_pose.position.y:.2f}), ({self.last_odom.pose.pose.position.x:.2f}, {self.last_odom.pose.pose.position.y:.2f})")
            
            # Calculate distance and bearing to target
            dx = target_x - current_x
            dy = target_y - current_y
            distance = np.sqrt(dx*dx + dy*dy)
            rospy.loginfo(f"[EXEC] Distance to target: {distance:.3f}m at ({target_x:.2f}, {target_y:.2f})")
            
            # Calculate angle to target
            target_angle = np.arctan2(dy, dx)
            
            # Get current yaw from AMCL when available
            if hasattr(self, 'last_amcl_pose') and (rospy.Time.now() - self.last_amcl_time).to_sec() < 0.5:
                current_yaw = self.amcl_yaw
            else:
                current_yaw = self.odom_yaw
            
            # Compare current orientation with target direction
            angle_diff = self._normalize_angle(target_angle - current_yaw)
            if abs(angle_diff) > np.pi/2:  # If target is behind us
                # Stay in current orientation and move backwards
                self.current_direction = -1
                desired_yaw = current_yaw  # Maintain current orientation
            else:
                # Turn to face direction of travel and move forwards
                self.current_direction = 1
                desired_yaw = current_yaw  # Maintain current orientation
            
            heading_error = self._normalize_angle(desired_yaw - current_yaw)
            
            # Check if we've reached the target
            if distance < 0.05:  # 5cm tolerance
                # Verify position with AMCL if available
                if hasattr(self, 'last_amcl_pose') and (rospy.Time.now() - self.last_amcl_time).to_sec() < 0.5:
                    amcl_dx = target_x - self.last_amcl_pose.position.x
                    amcl_dy = target_y - self.last_amcl_pose.position.y
                    amcl_distance = np.sqrt(amcl_dx*amcl_dx + amcl_dy*amcl_dy)
                    if amcl_distance > 0.1:  # 10cm tolerance for AMCL
                        rospy.logwarn(f"[EXEC] AMCL reports different position, continuing. AMCL distance: {amcl_distance:.3f}m")
                        continue
                
                # Stop the robot completely
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                rospy.sleep(0.5)  # Wait for robot to stop completely
                rospy.loginfo(f"[EXEC] Waypoint reached at ({current_x:.2f}, {current_y:.2f})")
                break
            
            # Generate velocity commands
            cmd = Twist()
            
            # Adjust heading
            cmd.angular.z = self.Kp * heading_error
            if abs(cmd.angular.z) > self.angular_speed:
                cmd.angular.z = self.angular_speed * (-1 if cmd.angular.z < 0 else 1)
            
            # Move if heading error is small
            if abs(heading_error) < np.radians(10):  # Only move if roughly pointing right way
                cmd.linear.x = self.current_direction * min(float(self.linear_speed), float(distance))
            
            self.cmd_vel_pub.publish(cmd)
            rospy.Rate(10).sleep()

        # Ensure we're stopped before returning
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        rospy.sleep(0.5)  # Make sure robot has stopped completely
        
        return True
if __name__ == '__main__':
    try:
        ExecutorAgent()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
