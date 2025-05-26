#!/usr/bin/env python3
import rospy
import json
import re
import ollama
from warehouse_robot.srv import AssignTask, AssignTaskRequest, AssignTaskResponse
from warehouse_robot.srv import GetPath, ExecutePath
from warehouse_robot.msg import RobotStatus
from geometry_msgs.msg import PoseWithCovarianceStamped

class CombinedManager:
    def __init__(self):
        rospy.init_node('manager_combined')
        
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
        self.current_task = None
        self.robot_status = {}
        
        # ROS Communication setup
        self._setup_ros_communication()
        
        rospy.loginfo("[Manager] Combined manager ready.")
    
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
            task_id='',
            battery_pct=100
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
    
    def status_cb(self, msg):
        if msg.robot_id != 'robot_1':
            return
            
        # Store status update
        self.robot_status[msg.robot_id] = msg
        rospy.loginfo(f"[Manager] Status from {msg.robot_id}: {msg.state}, task={msg.task_id}")
        
        # Handle task completion
        was_busy = self.is_busy
        self.is_busy = (msg.state == 'busy')
        
        if was_busy and not self.is_busy and self.current_task:
            rospy.loginfo(f"[Manager] Completed {self.current_task}")
            self.remove_task(self.current_task)
            self.current_task = None
            if self.task_list:
                rospy.Timer(
                    rospy.Duration(5.0),
                    self.decide_cb,
                    oneshot=True
                )
    
    def remove_task(self, task_id):
        self.completed_tasks.add(task_id)
        self.task_list = [
            t for t in self.task_list
            if t['id'] not in self.completed_tasks
        ]
        rospy.loginfo(f"[Manager] Removed {task_id}. {len(self.task_list)} left")
    
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
        path_resp = get_path(goal_name=task_id)
        
        # Execute path
        rospy.wait_for_service('/execute_path')
        exec_path = rospy.ServiceProxy('/execute_path', ExecutePath)
        ok = exec_path(
            task_id=task_id,
            waypoints=path_resp.waypoints,
            suggested_actions=path_resp.suggested_actions,
            descriptions=path_resp.descriptions
        ).success
        
        if ok:
            rospy.loginfo(f"[Manager] Execution started for {task_id}")
        else:
            rospy.logerr(f"[Manager] Execution failed for {task_id}")
            self.is_busy = False
            self.current_task = None
    
    def handle_assign(self, req):
        # External task assignment handler
        status = self.robot_status.get(req.robot_id)
        if not status or status.state != 'idle':
            return AssignTaskResponse(False, "Robot not available")
        
        rospy.loginfo(f"[Manager] External assignment: {req.task_id}@{req.location_name} to {req.robot_id}")
        return AssignTaskResponse(True, "Task accepted")
    
    @staticmethod
    def _yaw_from_quat(q):
        import numpy as np
        return np.arctan2(2*(q.w*q.z + q.x*q.y),
                         1 - 2*(q.y*q.y + q.z*q.z))
    
    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        CombinedManager().spin()
    except rospy.ROSInterruptException:
        pass
