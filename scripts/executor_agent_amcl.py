#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from warehouse_robot.srv import ExecutePath, ExecutePathResponse
from tf.transformations import euler_from_quaternion

class ExecutorAgentAMCL:
    def __init__(self):
        rospy.init_node('executor_agent_amcl')
        
        # Load grid cell size from your world params
        self.world_name = rospy.get_param("~world_name")
        self.config = rospy.get_param(self.world_name)
        self.cell_size = self.config["cell_size"]
        
        # Movement parameters
        self.linear_speed = rospy.get_param('~linear_speed', 0.2)
        self.angular_speed = rospy.get_param('~angular_speed', 0.5)
        
        # PID control parameters
        self.Kp_linear = rospy.get_param('~Kp_linear', 1.0)
        self.Ki_linear = rospy.get_param('~Ki_linear', 0.1)
        self.Kd_linear = rospy.get_param('~Kd_linear', 0.2)
        
        self.Kp_angular = rospy.get_param('~Kp_angular', 1.0)
        self.Ki_angular = rospy.get_param('~Ki_angular', 0.1)
        self.Kd_angular = rospy.get_param('~Kd_angular', 0.3)
        
        # Error integration limits
        self.linear_integral_limit = rospy.get_param('~linear_integral_limit', 0.5)
        self.angular_integral_limit = rospy.get_param('~angular_integral_limit', 1.0)
        
        # Initialize PID state variables
        self.linear_integral = 0.0
        self.angular_integral = 0.0
        self.last_linear_error = 0.0
        self.last_angular_error = 0.0
        self.last_error_time = None
        
        # AMCL reliability thresholds
        self.max_pose_covariance = rospy.get_param('~max_pose_covariance', 0.25)  # m^2
        self.max_orientation_covariance = rospy.get_param('~max_orientation_covariance', 0.5)  # rad^2
        
        # Set up communication at higher rates
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.amcl_sub = rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self.amcl_callback, queue_size=1)
        self.execute_service = rospy.Service('execute_path', ExecutePath, self.execute_callback)

        # Initialize state variables
        self.last_amcl_pose = None
        self.last_amcl_time = None
        self.amcl_yaw = 0
        self.current_direction = 1
        self.pose_covariance = None
        
        rospy.loginfo("[EXEC] AMCL-based executor agent initialized")

    def amcl_callback(self, msg):
        """Handle new AMCL pose updates with reliability checks"""
        # Store covariance for reliability checks
        # Extract position and orientation covariance
        pose_cov = max(msg.pose.covariance[0], msg.pose.covariance[7])  # max of x and y variance
        orientation_cov = msg.pose.covariance[35]  # yaw variance
        
        # Check if pose is reliable
        if pose_cov > self.max_pose_covariance:
            rospy.logwarn_throttle(1.0, f"[EXEC] High position uncertainty: {pose_cov:.3f} m^2")
            return
            
        if orientation_cov > self.max_orientation_covariance:
            rospy.logwarn_throttle(1.0, f"[EXEC] High orientation uncertainty: {orientation_cov:.3f} rad^2")
            return
        
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
        self.pose_covariance = pose_cov  # Store for monitoring
        
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def calibrate_amcl(self):
        """Perform a slow 360-degree rotation to improve AMCL pose estimation"""
        rospy.loginfo("[EXEC] Starting AMCL calibration spin...")
        
        # Make sure we start from a complete stop
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        rospy.sleep(1.0)
        
        # Set up rotation command
        cmd = Twist()
        cmd.angular.z = self.angular_speed * 0.5  # Use half speed for better scan matching
        
        start_time = rospy.Time.now()
        start_yaw = self.amcl_yaw
        total_rotation = 0.0
        last_reliable_time = rospy.Time.now()
        
        # Rotate until we complete a full circle
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            
            # Check if we have recent AMCL data
            if self.last_amcl_time and (current_time - self.last_amcl_time).to_sec() < 0.5:
                # Update total rotation
                total_rotation = abs(self._normalize_angle(self.amcl_yaw - start_yaw))
                last_reliable_time = current_time
                
                # Log progress
                if int(total_rotation * 180/np.pi) % 45 == 0:  # Log every 45 degrees
                    rospy.loginfo(f"[EXEC] Calibration spin progress: {int(total_rotation * 180/np.pi)}°")
            
            # Check completion conditions
            if total_rotation >= 2.0 * np.pi:  # Completed full rotation
                break
            elif (current_time - last_reliable_time).to_sec() > 5.0:  # No reliable poses for 5 seconds
                rospy.logwarn("[EXEC] Lost AMCL tracking during calibration")
                break
            elif (current_time - start_time).to_sec() > 30.0:  # Timeout after 30 seconds
                rospy.logwarn("[EXEC] Calibration spin timeout")
                break
                
            # Send rotation command
            self.cmd_vel_pub.publish(cmd)
            rospy.Rate(20).sleep()
        
        # Stop rotation
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        rospy.sleep(1.0)
        
        if total_rotation >= 2.0 * np.pi:
            rospy.loginfo("[EXEC] Calibration spin completed successfully")
            return True
        else:
            rospy.logwarn("[EXEC] Calibration spin did not complete")
            return False

    def execute_callback(self, req):
        """Handle execution of a path defined by waypoints and suggested actions"""
        if not self.last_amcl_pose:
            rospy.logerr("[EXEC] No AMCL pose available")
            return ExecutePathResponse(success=False)
        
        # # Perform calibration spin before starting navigation
        # if not self.calibrate_amcl():
        #     rospy.logwarn("[EXEC] Proceeding without complete calibration")
        
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
        """Execute a single step using waypoint-based navigation with PID control"""
        waypoint = step_info['waypoint']
        
        # Reset PID state for new waypoint
        self.linear_integral = 0.0
        self.angular_integral = 0.0
        self.last_linear_error = 0.0
        self.last_angular_error = 0.0
        self.last_error_time = None
        
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
        
        rospy.loginfo(f"[EXEC] Moving to waypoint: ({target_x:.2f}, {target_y:.2f}, {np.degrees(target_yaw):.1f}°)")
        rospy.loginfo(f"[EXEC] Current yaw: {np.degrees(self.amcl_yaw):.1f}°")

        # Make sure we start from a complete stop
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        rospy.sleep(0.5)  # Wait for robot to stop completely

        # First rotate to target orientation
        while not rospy.is_shutdown():
            heading_error = self._normalize_angle(target_yaw - self.amcl_yaw)
            
            rospy.loginfo(f"[EXEC] Current yaw: {np.degrees(self.amcl_yaw):.1f}°, Target yaw: {np.degrees(target_yaw):.1f}°")
            rospy.loginfo(f"[EXEC] Heading error: {np.degrees(heading_error):.1f}°")
            
            # Check if we've achieved target orientation within tolerance
            if abs(heading_error) < np.radians(2.0):  # 2 degree tolerance
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                rospy.sleep(0.2)
                rospy.loginfo(f"[EXEC] Aligned to target orientation: {np.degrees(self.amcl_yaw):.1f}°")
                break

            # Calculate time delta for PID
            current_time = rospy.Time.now()
            if self.last_error_time is None:
                self.last_error_time = current_time
                dt = 0.1  # Initial assumption
            else:
                dt = (current_time - self.last_error_time).to_sec()
                
            # Update angular PID terms
            self.angular_integral += heading_error * dt
            self.angular_integral = np.clip(self.angular_integral, -self.angular_integral_limit, self.angular_integral_limit)
            
            if dt > 0:
                angular_derivative = (heading_error - self.last_angular_error) / dt
            else:
                angular_derivative = 0
                
            # Calculate PID control output
            cmd = Twist()
            cmd.angular.z = (self.Kp_angular * heading_error + 
                           self.Ki_angular * self.angular_integral +
                           self.Kd_angular * angular_derivative)
                           
            # Limit angular velocity
            if abs(cmd.angular.z) > self.angular_speed:
                cmd.angular.z = self.angular_speed * (-1 if cmd.angular.z < 0 else 1)
                
            # Update state variables
            self.last_angular_error = heading_error
            self.last_error_time = current_time
            
            # Send command and maintain loop rate
            self.cmd_vel_pub.publish(cmd)
            rospy.Rate(20).sleep()  # Increased update rate

        # Then move to target position
        while not rospy.is_shutdown():
            # Get current position from AMCL
            current_x = self.last_amcl_pose.position.x
            current_y = self.last_amcl_pose.position.y
            
            rospy.loginfo(f"[EXEC] Current pos: ({current_x:.2f}, {current_y:.2f})")
            
            # Calculate distance and bearing to target
            dx = target_x - current_x
            dy = target_y - current_y
            distance = np.sqrt(dx*dx + dy*dy)
            target_angle = np.arctan2(dy, dx)
            
            # Compare current orientation with target direction
            angle_diff = self._normalize_angle(target_angle - self.amcl_yaw)
            if abs(angle_diff) > np.pi/2:  # If target is behind us
                # Stay in current orientation and move backwards
                self.current_direction = -1
                desired_yaw = self.amcl_yaw  # Maintain current orientation
            else:
                # Move forwards
                self.current_direction = 1
                desired_yaw = self.amcl_yaw  # Maintain current orientation
            
            heading_error = self._normalize_angle(desired_yaw - self.amcl_yaw)
            
            # Log current position and distance
            rospy.loginfo(f"[EXEC] Current pos: ({current_x:.2f}, {current_y:.2f}), Distance: {distance:.3f}m")
            
            # Check if we've reached the target
            if distance < 0.05:  # 5cm tolerance
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                rospy.sleep(0.5)
                rospy.loginfo(f"[EXEC] Waypoint reached at ({current_x:.2f}, {current_y:.2f})")
                break
            
            # Calculate time delta for PID
            current_time = rospy.Time.now()
            if self.last_error_time is None:
                self.last_error_time = current_time
                dt = 0.1  # Initial assumption
            else:
                dt = (current_time - self.last_error_time).to_sec()
                
            # Update PID terms
            # Linear control
            linear_error = distance
            self.linear_integral += linear_error * dt
            self.linear_integral = np.clip(self.linear_integral, -self.linear_integral_limit, self.linear_integral_limit)
            
            if dt > 0:
                linear_derivative = (linear_error - self.last_linear_error) / dt
            else:
                linear_derivative = 0
                
            # Angular control
            self.angular_integral += heading_error * dt
            self.angular_integral = np.clip(self.angular_integral, -self.angular_integral_limit, self.angular_integral_limit)
            
            if dt > 0:
                angular_derivative = (heading_error - self.last_angular_error) / dt
            else:
                angular_derivative = 0
                
            # Generate command with PID control
            cmd = Twist()
            
            # Angular control with PID
            cmd.angular.z = (self.Kp_angular * heading_error + 
                           self.Ki_angular * self.angular_integral +
                           self.Kd_angular * angular_derivative)
                           
            # Limit angular velocity
            if abs(cmd.angular.z) > self.angular_speed:
                cmd.angular.z = self.angular_speed * (-1 if cmd.angular.z < 0 else 1)
            
            # Linear control with PID if roughly aligned
            if abs(heading_error) < np.radians(10):
                cmd.linear.x = self.current_direction * (
                    self.Kp_linear * linear_error +
                    self.Ki_linear * self.linear_integral +
                    self.Kd_linear * linear_derivative
                )
                # Limit linear velocity
                cmd.linear.x = np.clip(cmd.linear.x, -self.linear_speed, self.linear_speed)
            
            # Debug info
            rospy.logdebug(f"[EXEC] Linear PID: error={linear_error:.3f}, I={self.linear_integral:.3f}, D={linear_derivative:.3f}")
            rospy.logdebug(f"[EXEC] Angular PID: error={heading_error:.3f}, I={self.angular_integral:.3f}, D={angular_derivative:.3f}")
            
            # Update state variables
            self.last_linear_error = linear_error
            self.last_angular_error = heading_error
            self.last_error_time = current_time
            
            # Send command and maintain loop rate
            self.cmd_vel_pub.publish(cmd)
            rospy.Rate(20).sleep()  # Increased update rate

        # Ensure we're stopped before returning
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        rospy.sleep(0.5)
        
        return True

if __name__ == '__main__':
    try:
        ExecutorAgentAMCL()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
