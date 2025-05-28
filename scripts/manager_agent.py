#!/usr/bin/env python3
import rospy
import json
import re
import ollama
from warehouse_robot.srv import AssignTask, AssignTaskResponse
from warehouse_robot.srv import GetPath, ExecutePath
from warehouse_robot.msg import RobotStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Quaternion
import numpy as np
from tf.transformations import euler_from_quaternion

class ManagerAgent:
    def __init__(self):
        rospy.init_node('manager_agent')
        
        # Map and task configuration
        self.world_name = rospy.get_param('~world_name')
        self.map_cfg = rospy.get_param(self.world_name)
        self.task_list = [
            {'id': n, 'x': c[0], 'y': c[1]}
            for n,c in self.map_cfg['locations'].items()
        ]
        self.completed_tasks = set()
        
        # LLM setup
        self.client = ollama.Client()
        self.model = rospy.get_param('~model', 'llama3.2')
        
        # Robot state tracking
        self.current_pose = None
        self.is_busy = False
        self.is_replanning = False
        self.current_task = None
        self.robot_status = {}
        
        # ROS Communication setup
        self._setup_ros_communication()
        
        rospy.loginfo("[Manager] Manager ready.")
    
    def _setup_ros_communication(self):
        # Status publisher
        self.status_pub = rospy.Publisher(
            '/robot_1/status',
            RobotStatus,
            queue_size=1,
            latch=True
        )
        
        # Initial idle status
        idle = RobotStatus(
            robot_id='robot_1',
            state='idle',
            task_id=''
        )
        self.status_pub.publish(idle)
        
        # Subscribers
        rospy.Subscriber(
            '/amcl_pose',
            PoseWithCovarianceStamped,
            self.pose_cb,
            queue_size=1
        )
        rospy.Subscriber(
            '/robot_1/status',
            RobotStatus,
            self.status_cb,
            queue_size=1
        )
        
        # Services
        self.assign_srv = rospy.Service(
            'assign_task',
            AssignTask,
            self.handle_assign
        )
        
        # Timer for task decisions
        self.timer = rospy.Timer(
            rospy.Duration(2.0),
            self.decide_cb
        )
    
    def pose_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = self._yaw_from_quat(q)
        self.current_pose = {'x': p.x, 'y': p.y, 'theta': yaw}
    
    def _quat_from_yaw(self, yaw):
        """Convert yaw angle to geometry_msgs/Quaternion"""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = np.sin(yaw / 2)
        q.w = np.cos(yaw / 2)
        return q

    def _yaw_from_quat(self, q):
        """Extract yaw angle from geometry_msgs/Quaternion"""
        euler = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return euler[2]  # yaw is the third element

    def status_cb(self, msg):
        if msg.robot_id != 'robot_1':
            return
            
        # Store status update
        self.robot_status[msg.robot_id] = msg
        rospy.loginfo(f"[Manager] Status from {msg.robot_id}: {msg.state}, task={msg.task_id}")
        
        # Handle task completion
        was_busy = self.is_busy
        self.is_busy = (msg.state == 'busy')
        
        # Only handle completion in status callback if task exists and robot becomes idle
        if was_busy and not self.is_busy and self.current_task and self.current_task in [t['id'] for t in self.task_list]:
            rospy.loginfo(f"[Manager] Task {self.current_task} completed via status update")
            self.remove_task(self.current_task)
            self.current_task = None
            if self.task_list:
                # Wait longer before starting next task to ensure everything is settled
                rospy.Timer(
                    rospy.Duration(10.0),
                    self.decide_cb,
                    oneshot=True
                )
    
    def remove_task(self, task_id):
        self.completed_tasks.add(task_id)
        self.task_list = [
            t for t in self.task_list
            if t['id'] not in self.completed_tasks
        ]
    
    def decide_cb(self, _):
        if not self.current_pose or self.is_busy or not self.task_list:
            return
        
        # Ask LLM for next task
        prompt_payload = {
            'robot_pose': self.current_pose,
            'pending_tasks': self.task_list,
            'map_config': self.map_cfg,
            'instruction': (
                "Given your current pose, the pending tasks, and the map configuration "
                "(cell size, origins, named locations), choose the next best task. "
                "Reply STRICTLY with JSON: {\"next_task\": \"<task_id>\"}."
            )
        }
        
        text = self.client.generate(
            model=self.model,
            prompt=json.dumps(prompt_payload)
        ).response.strip()
        
        # Parse LLM response
        try:
            next_id = json.loads(text)['next_task']
        except Exception:
            m = re.search(r'"next_task"\s*:\s*"([^"]+)"', text)
            if not m:
                rospy.logerr(f"[Manager] Could not parse LLM response:\n>>{text}")
                return
            next_id = m.group(1)
        
        # Self-assign task
        if self._assign_task(next_id):
            self._plan_and_execute(next_id)
    
    def _assign_task(self, task_id):
        # Internal task assignment
        status = self.robot_status.get('robot_1')
        if not status or status.state != 'idle':
            rospy.logwarn("[Manager] Robot not available for task assignment")
            return False
        
        rospy.loginfo(f"[Manager] Assigning task {task_id}")
        self.is_busy = True
        self.current_task = task_id
        return True
    
    def _plan_and_execute(self, task_id):
        # Get path plan
        rospy.wait_for_service('/get_path')
        get_path = rospy.ServiceProxy('/get_path', GetPath)
        
        rospy.loginfo(f"[Manager] Planning path for task {task_id}")
        
        # Create start pose from current robot pose
        start_pose = Pose()
        start_pose.position.x = self.current_pose['x']
        start_pose.position.y = self.current_pose['y']
        start_pose.position.z = 0.0
        start_pose.orientation = self._quat_from_yaw(self.current_pose['theta'])
        
        # Create goal pose from task location
        goal_location = next((t for t in self.task_list if t['id'] == task_id), None)
        if not goal_location:
            rospy.logerr(f"[Manager] Task {task_id} not found in task list")
            return
        
        # Convert task location to global coordinates
        goal_pose = Pose()
        goal_pose.position.x = goal_location['x'] * 0.25 + (-10)
        goal_pose.position.y = goal_location['y'] * 0.25 + (-10)
        goal_pose.position.z = 0.0
        # Set goal orientation to face east (0 degrees) by default
        goal_pose.orientation = self._quat_from_yaw(0.0)
        
        path_resp = get_path(
            start_pose=start_pose,
            goal_pose=goal_pose
        )
        
        # Check planner response
        if not path_resp.waypoints:
            # Empty response means error/failure
            rospy.logerr(f"[Manager] Planner failed to find path for {task_id}")
            # Only reset state if we're not in replan mode
            if not self.is_replanning:
                if self.current_task:
                    self.is_busy = False
                    self.current_task = None
            return
        
        rospy.loginfo(f"[Manager] Planner found {len(path_resp.waypoints)} waypoints for task {task_id}")
        
        # Single waypoint with "Already at goal" means success
        if len(path_resp.waypoints) == 1 and path_resp.descriptions and path_resp.descriptions[0] == "Already at goal":
            rospy.loginfo(f"[Manager] Already at goal location for task {task_id}")
            self.remove_task(self.current_task)
            self.is_busy = False
            self.current_task = None
            return
        
        rospy.loginfo(f"[Manager] Executing path for task {task_id}")
        
        # Execute path
        rospy.wait_for_service('/execute_path')
        rospy.loginfo("[Manager] Waiting for execute_path service")
        exec_path = rospy.ServiceProxy('/execute_path', ExecutePath)
        rospy.loginfo("[Manager] Got execute_path service")
        ok = exec_path(
            task_id=task_id,
            waypoints=path_resp.waypoints,
            suggested_actions=path_resp.suggested_actions,
            descriptions=path_resp.descriptions
        ).success
        
        rospy.loginfo(f"[Manager] Execution result for {task_id}: {ok}")
        
        if ok:
            # First check if task was already removed
            if task_id not in [t['id'] for t in self.task_list]:
                rospy.loginfo(f"[Manager] Task {task_id} was completed during execution")
                # Clear current task if it matches
                if self.current_task == task_id:
                    self.current_task = None
                return
            
            # Get current robot status
            status = self.robot_status.get('robot_1')
            if status and status.state == 'idle':
                # If robot is idle and task still exists, complete it now
                rospy.loginfo(f"[Manager] Task {task_id} completed normally")
                self.remove_task(task_id)
                self.current_task = None
                return
                
            # If we get here, task is not complete and robot is busy - must be a replan
            rospy.loginfo("[Manager] Execution returned success but task not complete, retrying with new path")
            rospy.Timer(
                rospy.Duration(1.0),  # Short delay to let everything settle
                lambda _: self._plan_and_execute(task_id),
                oneshot=True
            )
            return
        else:
            rospy.logerr(f"[Manager] Execution failed for {task_id}")
            # Only reset state if execution actually failed
            if self.current_task:
                self.is_busy = False
                self.current_task = None
            return
            # Don't mark as complete, let status callback handle that
    
    def handle_assign(self, req):
        # External task assignment handler
        status = self.robot_status.get(req.robot_id)
        if not status or status.state != 'idle':
            return AssignTaskResponse(False, "Robot not available")
        
        rospy.loginfo(f"[Manager] External assignment: {req.task_id}@{req.location_name} to {req.robot_id}")
        return AssignTaskResponse(True, "Task accepted")
    
    def _quat_from_yaw(self, yaw):
        """Convert yaw angle to geometry_msgs/Quaternion"""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = np.sin(yaw / 2)
        q.w = np.cos(yaw / 2)
        return q

    def _yaw_from_quat(self, q):
        """Extract yaw angle from geometry_msgs/Quaternion"""
        # Use existing quaternion to euler conversion
        import numpy as np
        return np.arctan2(2*(q.w*q.z + q.x*q.y),
                         1 - 2*(q.y*q.y + q.z*q.z))
    
    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        ManagerAgent().spin()
    except rospy.ROSInterruptException:
        pass
