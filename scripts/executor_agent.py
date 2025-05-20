#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from warehouse_robot.srv import ExecutePath, ExecutePathResponse, ValidatePath
import nav_msgs.msg

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
        
        # Control parameters - rotation
        self.Kp = rospy.get_param('~Kp', 0.01)  # Proportional gain for rotation
        self.Kd = rospy.get_param('~Kd', 0.00005)  # Derivative gain for rotation
        
        # Control parameters - position
        self.Kp_pos = rospy.get_param('~Kp_pos', 0.1)  # Proportional gain for position
        self.Ki_pos = rospy.get_param('~Ki_pos', 0.01)  # Integral gain for position
        self.Kd_pos = rospy.get_param('~Kd_pos', 0.05)  # Derivative gain for position
        self.integral_limit = 0.2  # Limit integral windup
        
        # Error thresholds - tighter for grid alignment
        self.angle_threshold = rospy.get_param('~angle_threshold', 2.0)  # degrees
        self.position_threshold = rospy.get_param('~position_threshold', 0.05)  # meters - tighter threshold
        self.completion_threshold = self.position_threshold * 1.5  # slightly more lenient for completion
        
        # AMCL topic for pose validation
        self.amcl_topic = rospy.get_param('~amcl_topic', '/amcl_pose')
        self.last_amcl_pose = None
        self.last_amcl_time = None
        self._last_valid_amcl = None  # Store last valid AMCL reading
        
        # Odometry for control feedback
        self.odom_topic = rospy.get_param('~odom_topic', '/odom')
        self.last_odom = None
        self.odom_yaw = None
        self.last_odom_time = None
        
        # Subscribe to both pose sources
        rospy.Subscriber(self.amcl_topic,
                        PoseWithCovarianceStamped,
                        self.amcl_callback)
        rospy.Subscriber(self.odom_topic,
                        nav_msgs.msg.Odometry,
                        self.odom_callback)
        
        # Publisher for velocity commands
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.loginfo("[EXEC] cmd_vel publisher ready.")

        # Validator service
        rospy.wait_for_service('/validate_path')
        self.validate_path = rospy.ServiceProxy('/validate_path', ValidatePath)
        rospy.loginfo("[EXEC] validate_path service ready.")

        # Executor service
        self.service = rospy.Service('/execute_path', ExecutePath, self.execute_callback)
        rospy.loginfo("[EXEC] execute_path service ready.")

    def execute_callback(self, req):
        # 1) Decode actions from the incoming request
        actions = req.actions
        rospy.loginfo(f"[EXEC] Received actions: {actions}")

        # 2) Validate entire action sequence
        valid = self.validate_path(actions=actions)
        if not valid.is_valid:
            rospy.logerr(f"[EXEC] Path rejected: {valid.message}")
            return ExecutePathResponse(success=False)

        # 3) Execute each action in turn
        for idx, act in enumerate(actions, 1):
            rospy.loginfo(f"[EXEC] Performing action {idx}/{len(actions)}: {act}")
            self._do_action(act)

        rospy.loginfo("[EXEC] All actions completed.")
        return ExecutePathResponse(success=True)
        
    def amcl_callback(self, msg):
        self.last_amcl_pose = msg.pose.pose
        self.last_amcl_time = rospy.Time.now()
        
    def odom_callback(self, msg):
        self.last_odom = msg
        self.last_odom_time = rospy.Time.now()
        o = msg.pose.pose.orientation
        self.odom_yaw = np.arctan2(2*(o.w*o.z), 1 - 2*(o.z**2))

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _do_action(self, act):
        cmd = Twist()
        v = self.linear_speed
        w = self.angular_speed
        rate = rospy.Rate(50)  # 50Hz control rate

        if act in ('L', 'R'):
            # Wait for odometry
            while not self.last_odom and not rospy.is_shutdown():
                rospy.sleep(0.1)
                
            initial_yaw = self.odom_yaw
            # Convert current yaw to 0-360 degrees for easier cardinal direction determination
            current_degrees = np.degrees(initial_yaw) % 360
            
            # Determine closest cardinal direction and next target based on turn direction
            if act == 'L':
                # Left turns go counterclockwise: E->N->W->S->E
                target_degrees = np.ceil((current_degrees + 45) / 90) * 90 % 360
            else:  # act == 'R'
                # Right turns go clockwise: E->S->W->N->E
                target_degrees = np.floor((current_degrees - 45) / 90) * 90 % 360
            
            # Convert back to radians
            target_yaw = np.radians(target_degrees)
            
            # Get cardinal direction names for logging
            cardinal_names = {0: "East", 90: "North", 180: "West", 270: "South"}
            current_cardinal = cardinal_names[round(current_degrees / 90) * 90 % 360]
            target_cardinal = cardinal_names[target_degrees]
            
            rospy.loginfo(f"[EXEC] Rotating from {current_cardinal} ({np.degrees(initial_yaw):.1f}°) to {target_cardinal} ({target_degrees:.1f}°)")
            
            last_error = 0
            last_time = rospy.Time.now()

            while not rospy.is_shutdown():
                if self.odom_yaw is None:
                    continue
                
                current_time = rospy.Time.now()
                dt = (current_time - last_time).to_sec()
                dt = max(dt, 0.0001)  # Prevent division by zero
                
                # Calculate error and derivative
                error = np.degrees(self._normalize_angle(target_yaw - self.odom_yaw))
                derivative = (error - last_error) / dt
                
                # PD control with smoother response
                base_velocity = w * 0.5  # Use half max speed as base
                correction = self.Kp * error + self.Kd * derivative
                
                # Set rotation direction and apply correction
                direction = 1 if error > 0 else -1
                angular_velocity = direction * base_velocity + correction
                
                # Apply limits
                angular_velocity = max(-w, min(w, angular_velocity))
                
                # Update command
                cmd = Twist()
                cmd.angular.z = angular_velocity
                self.cmd_pub.publish(cmd)
                
                # Check if we've reached target
                if abs(error) < self.angle_threshold:
                    rospy.loginfo(f"[EXEC] Rotation complete. Final error: {error:.2f}°")
                    break
                
                last_error = error
                last_time = current_time
                rate.sleep()

        else:  # Forward or Backward movement
            # Wait for both pose sources
            while (not self.last_odom or not self.last_amcl_pose) and not rospy.is_shutdown():
                rospy.sleep(0.1)
                
            # Wait for a stable AMCL reading
            stable_readings = 0
            last_amcl_x = None
            last_amcl_y = None
            start_wait = rospy.Time.now()
            
            while stable_readings < 3 and not rospy.is_shutdown():  # Need 3 consistent readings
                if (rospy.Time.now() - start_wait).to_sec() > 3.0:
                    rospy.logwarn("[EXEC] Timeout waiting for stable AMCL")
                    break
                    
                if not self.last_amcl_pose or (rospy.Time.now() - self.last_amcl_time).to_sec() > 0.5:
                    rospy.loginfo_throttle(1.0, "[EXEC] Waiting for fresh AMCL update...")
                    rospy.sleep(0.1)
                    continue
                    
                current_amcl_x = self.last_amcl_pose.position.x
                current_amcl_y = self.last_amcl_pose.position.y
                
                if (last_amcl_x is not None and
                    abs(current_amcl_x - last_amcl_x) < self.position_threshold and
                    abs(current_amcl_y - last_amcl_y) < self.position_threshold):
                    stable_readings += 1
                else:
                    stable_readings = 0
                    
                last_amcl_x = current_amcl_x
                last_amcl_y = current_amcl_y
                rospy.sleep(0.1)
                
            self._last_valid_amcl = self.last_amcl_pose  # Store stable reading
            
            current_x = self.last_amcl_pose.position.x
            current_y = self.last_amcl_pose.position.y
            initial_yaw = self.odom_yaw  # Use odom for heading
            
            # Round current position to nearest grid cell based on current heading
            # This ensures we use the correct grid cell based on the robot's orientation
            heading_cos = np.cos(initial_yaw)
            heading_sin = np.sin(initial_yaw)
            
            # Project position onto major axes based on heading
            if abs(heading_cos) > abs(heading_sin):  # Facing mostly east/west
                # For east/west movement, round X to grid, preserve Y
                grid_x = round(current_x / self.cell_size) * self.cell_size
                grid_y = current_y  # Keep exact Y
                # Log the rounding calculation for debugging
                rospy.loginfo(f"[EXEC] E/W Rounding: x={current_x:.3f}->{grid_x:.3f}, y={current_y:.3f}")
            else:  # Facing mostly north/south
                # For north/south movement, round Y to grid, preserve X
                grid_x = current_x  # Keep exact X
                # Use floor for negative Y and ceil for positive Y to ensure consistent rounding
                y_cells = current_y / self.cell_size
                if y_cells < 0:
                    grid_y = np.floor(y_cells) * self.cell_size
                else:
                    grid_y = np.ceil(y_cells) * self.cell_size
                # Log the rounding calculation for debugging
                rospy.loginfo(f"[EXEC] N/S Rounding: x={current_x:.3f}, y={current_y:.3f}->{grid_y:.3f}")
            
            # Calculate direction of movement
            direction = 1 if act == 'F' else -1 if act == 'B' else 0
            if direction == 0:
                rospy.logwarn(f"[EXEC] Unknown action '{act}'")
                return
            
            # Calculate target based on nearest grid position and validate with AMCL
            target_x = grid_x + direction * self.cell_size * np.cos(initial_yaw)
            target_y = grid_y + direction * self.cell_size * np.sin(initial_yaw)
            target_yaw = initial_yaw

            # Validate starting position with AMCL
            amcl_x_error = grid_x - current_x
            amcl_y_error = grid_y - current_y
            amcl_error = np.hypot(amcl_x_error, amcl_y_error)
            if amcl_error > self.position_threshold * 2:
                rospy.logwarn(f"[EXEC] Large initial position error from AMCL: {amcl_error:.3f}m")
                # Adjust grid position to actual AMCL position
                grid_x = current_x
                grid_y = current_y
                target_x = grid_x + direction * self.cell_size * np.cos(initial_yaw)
                target_y = grid_y + direction * self.cell_size * np.sin(initial_yaw)
            
            rospy.loginfo(f"[EXEC] Current grid pos: ({grid_x:.3f}, {grid_y:.3f})")
            rospy.loginfo(f"[EXEC] Moving {act} to ({target_x:.3f}, {target_y:.3f})")
            
            # Store initial position for progress validation
            self._movement_start = (current_x, current_y)
            
            # PID control variables
            last_time = rospy.Time.now()
            last_x_error = 0
            last_y_error = 0
            x_integral = 0
            y_integral = 0

            while not rospy.is_shutdown():
                current_time = rospy.Time.now()
                dt = (current_time - last_time).to_sec()
                dt = max(dt, 0.0001)
                
                # Use odometry for control feedback
                if not self.last_odom or (current_time - self.last_odom_time).to_sec() > 0.1:
                    rospy.logwarn_throttle(1, "[EXEC] No recent odometry updates")
                    rate.sleep()
                    continue
                    
                current_x = self.last_odom.pose.pose.position.x
                current_y = self.last_odom.pose.pose.position.y
                
                # Calculate all position and orientation variables in a logical order
                x_error = target_x - current_x
                y_error = target_y - current_y
                distance = np.hypot(x_error, y_error)
                distance_factor = min(1.0, distance / self.cell_size)
                
                # Calculate errors in robot frame first
                forward_error = (x_error * np.cos(self.odom_yaw) + 
                               y_error * np.sin(self.odom_yaw))
                lateral_error = (-x_error * np.sin(self.odom_yaw) + 
                               y_error * np.cos(self.odom_yaw))
                
                # Then calculate all heading-related angles
                current_heading = np.degrees(self.odom_yaw)
                path_heading = np.degrees(np.arctan2(y_error, x_error))
                if act == 'B':
                    path_heading = self._normalize_angle(path_heading + 180)
                
                # Calculate final heading errors and corrections
                steering_angle = self._normalize_angle(path_heading - current_heading)
                heading_error = np.degrees(self._normalize_angle(target_yaw - self.odom_yaw))
                lateral_correction = np.degrees(np.arctan2(lateral_error, self.cell_size * 0.5))
                
                # Get previous values for derivative control
                last_composite_error = getattr(self, '_last_composite_error', 0)
                last_lateral = getattr(self, '_last_lateral', 0)
                
                # Update integrals with anti-windup
                x_integral = max(-self.integral_limit, min(self.integral_limit, 
                               x_integral + x_error * dt))
                y_integral = max(-self.integral_limit, min(self.integral_limit, 
                               y_integral + y_error * dt))
                
                # Calculate derivatives
                x_derivative = (x_error - last_x_error) / dt
                y_derivative = (y_error - last_y_error) / dt
                
                # Calculate PID components
                proportional = self.Kp_pos * forward_error
                integral_term = self.Ki_pos * (x_integral * np.cos(self.odom_yaw) + 
                                             y_integral * np.sin(self.odom_yaw))
                derivative_term = self.Kd_pos * (x_derivative * np.cos(self.odom_yaw) + 
                                               y_derivative * np.sin(self.odom_yaw))
                
                # Combine all PID components for velocity
                pid_correction = proportional + integral_term + derivative_term
                
                # Remove duplicated angle calculations
                # The calculations are already done earlier in the code
                if act == 'B':
                    path_heading = self._normalize_angle(path_heading + 180)
                
                # Basic steering to follow path
                steering_angle = self._normalize_angle(path_heading - current_heading)
                
                # Calculate desired heading (where we want to end up)
                heading_error = np.degrees(self._normalize_angle(target_yaw - self.odom_yaw))
                
                # Calculate correction for lateral offset
                # Using atan2 to get proper sign and scaling
                lateral_correction = np.degrees(np.arctan2(lateral_error, self.cell_size * 0.5))
                
                # Blend between path following and final heading
                distance = np.hypot(x_error, y_error)
                distance_factor = min(1.0, distance / self.cell_size)
                
                # Calculate blended control error
                composite_error = (distance_factor * (steering_angle + 0.5 * lateral_correction) + 
                                 (1 - distance_factor) * heading_error)
                
                # Calculate derivative for damping
                composite_derivative = (composite_error - last_composite_error) / dt
                
                # Calculate angular velocity with adaptive gains
                effective_Kp = self.Kp * (0.5 + 0.5 * distance_factor)  # More aggressive when far
                angular_velocity = (effective_Kp * composite_error + 
                                  self.Kd * composite_derivative)
                angular_velocity = max(-w/2, min(w/2, angular_velocity))  # Limit angular velocity
                
                # Calculate linear velocity with heading-based scaling
                heading_factor = np.cos(np.radians(composite_error))
                linear_velocity = pid_correction * max(0.3, abs(heading_factor))
                linear_velocity = max(-v, min(v, linear_velocity))  # Apply velocity limits
                
                # Ensure minimum velocity near target when aligned
                if abs(composite_error) < 30 and abs(linear_velocity) < 0.05:
                    linear_velocity = 0.05 * np.sign(forward_error) * direction
                    
                # Add debug logging for movement components
                if (current_time.to_sec() % 1.0) < 0.02:  # Log every ~1 second
                    rospy.loginfo(f"[EXEC] Forward error: {forward_error:.3f}m, Lateral: {lateral_error:.3f}m")
                    rospy.loginfo(f"[EXEC] Target heading: {target_yaw:.1f}°, Current: {current_heading:.1f}°")
                    rospy.loginfo(f"[EXEC] Path heading: {path_heading:.1f}°, Correction: {lateral_correction:.1f}°")
                    rospy.loginfo(f"[EXEC] Composite error: {composite_error:.1f}°, Distance factor: {distance_factor:.2f}")
                    rospy.loginfo(f"[EXEC] Linear: {linear_velocity:.3f}, Angular: {angular_velocity:.3f}")
                
                # Add debug logging for movement components
                if (current_time.to_sec() % 1.0) < 0.02:  # Log every ~1 second
                    rospy.loginfo(f"[EXEC] Forward error: {forward_error:.3f}m, Lateral: {lateral_error:.3f}m")
                    rospy.loginfo(f"[EXEC] Target heading: {target_yaw:.1f}°, Current: {current_heading:.1f}°")
                    rospy.loginfo(f"[EXEC] Path heading: {path_heading:.1f}°, Correction: {lateral_correction:.1f}°")
                    rospy.loginfo(f"[EXEC] Composite error: {composite_error:.1f}°, Distance factor: {distance_factor:.2f}")
                    rospy.loginfo(f"[EXEC] Linear: {linear_velocity:.3f}, Angular: {angular_velocity:.3f}")
                
                # Store values for next iteration
                self._last_composite_error = composite_error
                self._last_lateral = lateral_error
                
                # Update command
                cmd = Twist()
                cmd.linear.x = linear_velocity
                cmd.angular.z = angular_velocity
                self.cmd_pub.publish(cmd)
                
                # Store values for next iteration
                last_x_error = x_error
                last_y_error = y_error
                last_time = current_time
                
                # Check completion with separate thresholds for forward and lateral errors
                forward_threshold = self.position_threshold
                lateral_threshold = self.position_threshold * 1.5  # Slightly more forgiving for lateral error
                
                # First validate total progress relative to cell size
                start_x, start_y = self._movement_start
                progress = np.hypot(current_x - start_x, current_y - start_y)
                expected_progress = self.cell_size  # Should move exactly one cell
                progress_error = abs(progress - expected_progress)
                
                # Get latest AMCL reading if available
                current_amcl = None
                if self.last_amcl_pose and (current_time - self.last_amcl_time).to_sec() < 0.5:
                    current_amcl = self.last_amcl_pose
                    
                if progress < expected_progress * 0.8:  # Less than 80% of a cell
                    if (current_time - last_time).to_sec() > 2.0:  # Stuck for 2 seconds
                        rospy.logwarn(f"[EXEC] Insufficient progress: {progress:.3f}m of {expected_progress:.3f}m")
                        # Use stronger correction if we have AMCL
                        if current_amcl:
                            amcl_progress = np.hypot(
                                current_amcl.position.x - self._movement_start[0],
                                current_amcl.position.y - self._movement_start[1]
                            )
                            if amcl_progress < expected_progress * 0.8:
                                # Verified by AMCL that we need more movement
                                linear_velocity = v * direction
                                angular_velocity = 0  # Focus on forward movement
                        continue
                
                if (abs(forward_error) < forward_threshold and 
                    abs(lateral_error) < lateral_threshold):
                    # Check heading only if forward error is small
                    heading_check = (abs(heading_error) < self.angle_threshold or 
                                   abs(forward_error) > forward_threshold)
                    
                    if heading_check:
                        if self.last_amcl_pose and (current_time - self.last_amcl_time).to_sec() < 1.0:
                            # Validate with AMCL, focusing on forward progress
                            amcl_x_error = target_x - self.last_amcl_pose.position.x
                            amcl_y_error = target_y - self.last_amcl_pose.position.y
                            amcl_forward_error = (amcl_x_error * np.cos(self.odom_yaw) + 
                                                amcl_y_error * np.sin(self.odom_yaw))
                            amcl_lateral_error = (-amcl_x_error * np.sin(self.odom_yaw) + 
                                                amcl_y_error * np.cos(self.odom_yaw))
                            
                            # Check both forward and lateral error with AMCL
                            if (abs(amcl_forward_error) > forward_threshold * 2 or
                                abs(amcl_lateral_error) > lateral_threshold * 2):
                                rospy.logwarn(f"[EXEC] AMCL validation failed - forward: {amcl_forward_error:.3f}m, lateral: {amcl_lateral_error:.3f}m")
                                continue
                            
                            # Final position validation with both odometry and AMCL
                            odom_progress = np.hypot(current_x - self._movement_start[0],
                                                   current_y - self._movement_start[1])
                            amcl_progress = np.hypot(self.last_amcl_pose.position.x - self._movement_start[0],
                                                   self.last_amcl_pose.position.y - self._movement_start[1])
                            
                            progress_diff = abs(odom_progress - amcl_progress)
                            if progress_diff > self.position_threshold:
                                rospy.logwarn(f"[EXEC] Odometry and AMCL disagree on progress: {progress_diff:.3f}m difference")
                                continue
                                
                            # Verify we've moved almost exactly one cell
                            if abs(amcl_progress - self.cell_size) > self.completion_threshold:
                                rospy.logwarn(f"[EXEC] Incorrect movement distance: {amcl_progress:.3f}m vs {self.cell_size:.3f}m expected")
                                continue
                    rospy.loginfo(f"[EXEC] Move complete. Final error: forward={forward_error:.3f}m, lateral={lateral_error:.3f}m")
                    break
                
                rate.sleep()

        # Stop movement
        self.cmd_pub.publish(Twist())
        rospy.sleep(0.2)  # Short delay to ensure robot has stopped

if __name__ == '__main__':
    ExecutorAgent()
    rospy.spin()

