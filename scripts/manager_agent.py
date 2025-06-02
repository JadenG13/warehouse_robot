#!/usr/bin/env python3
import rospy
import json
import re
import ollama
from gazebo_msgs.srv import GetModelState
from warehouse_robot.srv import GetPath, ExecutePath, ValidatePathWithPat
from warehouse_robot.msg import RobotStatus
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from nav_msgs.msg import Path  # Added for path visualization
import numpy as np
from tf.transformations import euler_from_quaternion
import time # For timing blocks

# Helper to format pose for logging
def format_pose_for_log(pose_obj):
    if isinstance(pose_obj, dict): # For self.current_pose
        return f"x:{pose_obj['x']:.2f}_y:{pose_obj['y']:.2f}_th:{np.degrees(pose_obj['theta']):.0f}"
    # For geometry_msgs.Pose
    # Ensure q.x, q.y, q.z, q.w are valid numbers before passing to euler_from_quaternion
    if hasattr(pose_obj, 'orientation') and all(hasattr(pose_obj.orientation, attr) for attr in ['x', 'y', 'z', 'w']):
        try:
            _, _, yaw = euler_from_quaternion([pose_obj.orientation.x, pose_obj.orientation.y, pose_obj.orientation.z, pose_obj.orientation.w])
            return f"x:{pose_obj.position.x:.2f}_y:{pose_obj.position.y:.2f}_th:{np.degrees(yaw):.0f}"
        except Exception as e: # pylint: disable=broad-except
            return f"x:{pose_obj.position.x:.2f}_y:{pose_obj.position.y:.2f}_th:ERR_CONVERTING_QUAT"
    return "POSE_FORMAT_ERROR"


class ManagerAgent:
    def __init__(self):
        rospy.init_node('manager_agent')
        self.node_start_time = rospy.Time.now().to_sec()
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_INIT, timestamp={self.node_start_time:.3f}")

        # Map and task configuration
        self.world_name = rospy.get_param('~world_name')
        self.map_cfg = rospy.get_param(self.world_name)
        self.task_list = [
            {'id': n, 'x': c[0], 'y': c[1]}
            for n,c in self.map_cfg['locations'].items()
        ]
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_TASK_LIST_INIT, timestamp={rospy.Time.now().to_sec():.3f}, num_initial_tasks={len(self.task_list)}")
        self.completed_tasks = set()

        # LLM setup
        self.client = ollama.Client()
        self.model = rospy.get_param('~model', 'llama3.2')

        # Robot state tracking
        self.current_pose = None
        self.grid_pose = None # for parsing pose to LLM
        self.is_busy = False
        self.is_replanning = False #  flag, behavior preserved
        self.current_task = None   #  state variable
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

        # Path visualization publisher
        self.path_pub = rospy.Publisher(
            '/planned_path',
            Path,
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
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_ROBOT_STATUS_PUBLISHED, timestamp={rospy.Time.now().to_sec():.3f}, robot_id=robot_1, state=idle, task_id=''")


        rospy.Subscriber(
            '/robot_1/status',
            RobotStatus,
            self.status_cb,
            queue_size=1
        )

        # Services
        self.get_model_state = rospy.ServiceProxy( # No change to name,  was fine
            '/gazebo/get_model_state',
            GetModelState
        )

        # Timer for task decisions
        self.timer = rospy.Timer(
            rospy.Duration(2.0),
            self.decide_cb
        )

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

    def _update_current_pose(self):
        # PERF_LOG for get_model_state call duration
        get_model_state_start_time = time.monotonic()
        current = self.get_model_state(
            model_name='turtlebot3_burger',
            relative_entity_name='map'
        )
        get_model_state_duration = time.monotonic() - get_model_state_start_time
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_GET_MODEL_STATE_CALL, timestamp={rospy.Time.now().to_sec():.3f}, duration_sec={get_model_state_duration:.4f}, success={current.success}")


        if current.success:
            p = current.pose.position
            q = current.pose.orientation
            yaw = self._yaw_from_quat(q)
            grid_x = int(round((p.x - (-10)) / 0.25))
            grid_y = int(round((p.y - (-10)) / 0.25))            
            self.current_pose = {'x': p.x, 'y': p.y, 'theta': yaw}
            self.grid_pose = {'x': grid_x, 'y': grid_y, 'theta': round(np.degrees(yaw))}
        else:
            rospy.logwarn("[Manager] Failed to get model state in _update_current_pose")
            self.current_pose = None

    def status_cb(self, msg):
        log_time = rospy.Time.now().to_sec()
        if msg.robot_id != 'robot_1':
            return

        self.robot_status[msg.robot_id] = msg
        rospy.loginfo(f"[Manager] Status from {msg.robot_id}: {msg.state}, task='{msg.task_id}'") #  log
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_ROBOT_STATUS_RECEIVED, timestamp={log_time:.3f}, robot_id={msg.robot_id}, state={msg.state}, task_id='{msg.task_id}'")


        was_busy = self.is_busy
        self.is_idle = (msg.state == 'idle')
        # self.is_busy = (msg.state == 'busy')
        # self.is_replanning = (msg.state == 'replan')

        #  logic for task completion via status update
        if was_busy and self.is_idle and self.current_task and self.current_task in [t['id'] for t in self.task_list]:
            rospy.loginfo(f"[Manager] Task {self.current_task} completed via status update") #  log
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_TASK_COMPLETED_VIA_STATUS, timestamp={log_time:.3f}, task_id='{self.current_task}', robot_id={msg.robot_id}")
            self.remove_task(self.current_task)
            self.is_busy = False
            self.current_task = None
            if self.task_list:
                rospy.Timer(
                    rospy.Duration(10.0),
                    self.decide_cb,
                    oneshot=True
                )

    def remove_task(self, task_id):
        log_time = rospy.Time.now().to_sec()
        initial_task_count = len(self.task_list)
        self.completed_tasks.add(task_id)
        self.task_list = [
            t for t in self.task_list
            if t['id'] not in self.completed_tasks
        ]
        if len(self.task_list) < initial_task_count:
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_TASK_REMOVED, timestamp={log_time:.3f}, task_id='{task_id}', remaining_tasks={len(self.task_list)}")
        else:
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_TASK_REMOVE_ATTEMPT_NO_CHANGE, timestamp={log_time:.3f}, task_id='{task_id}', remaining_tasks={len(self.task_list)}")


    def decide_cb(self, _):        
        self.is_replanning = (self.robot_status["robot_1"].state == 'replan')
        self._update_current_pose()
        if not self.is_replanning:
            decide_start_time = rospy.Time.now().to_sec()
            rospy.loginfo("[Manager] Deciding next task...") #  log
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_DECIDE_CB_START, timestamp={decide_start_time:.3f}")


            if not self.current_pose:
                #  code did not log here but would return
                rospy.loginfo(f"[PERF_LOG] event=MANAGER_DECIDE_CB_ABORT_NO_POSE, timestamp={rospy.Time.now().to_sec():.3f}")
                return
            if self.is_busy:
                #  code did not log here but would return
                rospy.loginfo(f"[PERF_LOG] event=MANAGER_DECIDE_CB_DEFER_ROBOT_BUSY, timestamp={rospy.Time.now().to_sec():.3f}")
                return
            if not self.task_list:
                #  code did not log here but would return
                rospy.loginfo(f"[PERF_LOG] event=MANAGER_DECIDE_CB_NO_TASKS, timestamp={rospy.Time.now().to_sec():.3f}")
                return
            
            # prompt_payload = {
            #     'robot_pose': self.current_pose,
            #     'pending_tasks': self.task_list,
            #     'map_config': self.map_cfg,
            #     'instruction': (
            #         "Given your current pose, the pending tasks, and the map configuration "
            #         "(cell size, origins, named locations), choose the next best task. "
            #         "Reply STRICTLY with JSON: {\"next_task\": \"<task_id>\"}."
            #     )
            # }
            rospy.logerr(f"[Manager] {self.task_list}")
            prompt_payload = {
                'robot_pose': self.grid_pose,
                'pending_tasks': self.task_list,
                'instruction': (
                    "Given your current pose and the pending tasks, choose the next best task. "
                    "Reply STRICTLY with JSON: {\"next_task\": \"<task_id>\"}. "
                    "Do not write any additional text or code. "
                )
            }
            
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_LLM_REQUEST_SENT, timestamp={rospy.Time.now().to_sec():.3f}, robot_pose={format_pose_for_log(self.current_pose)}, num_pending_tasks={len(self.task_list)}")
            llm_call_start_time = time.monotonic()
            text_response = self.client.generate(
                model=self.model,
                prompt=json.dumps(prompt_payload)
            ).response.strip()
            llm_call_duration = time.monotonic() - llm_call_start_time
            llm_response_time = rospy.Time.now().to_sec()
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_LLM_RESPONSE_RECEIVED, timestamp={llm_response_time:.3f}, duration_sec={llm_call_duration:.3f}, response_summary='{text_response[:100]}...'")


            next_task_id_from_llm = None # Renamed to avoid confusion with self.current_task
            try:
                next_task_id_from_llm = json.loads(text_response)['next_task']
            except Exception:
                m = re.search(r'"next_task"\s*:\s*"([^"]+)"', text_response)
                if not m:
                    rospy.logerr(f"[Manager] Could not parse LLM response:\n>>{text_response}") #  log
                    rospy.loginfo(f"[PERF_LOG] event=MANAGER_LLM_PARSE_FAIL, timestamp={rospy.Time.now().to_sec():.3f}")
                    return
                next_task_id_from_llm = m.group(1)

            rospy.loginfo(f"[PERF_LOG] event=MANAGER_LLM_SUGGESTED_TASK, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{next_task_id_from_llm}'")

            assign_task_success = self._assign_task(next_task_id_from_llm)
            if assign_task_success:
                self._plan_and_execute(next_task_id_from_llm)
            else:
                #  code did not have an else here for _assign_task failure from decide_cb
                rospy.loginfo(f"[PERF_LOG] event=MANAGER_ASSIGN_TASK_FAILED_FROM_DECIDE_CB, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{next_task_id_from_llm}'")

            rospy.loginfo(f"[PERF_LOG] event=MANAGER_DECIDE_CB_END, timestamp={rospy.Time.now().to_sec():.3f}, total_duration_sec={rospy.Time.now().to_sec() - decide_start_time:.3f}")
        else: 
            self._plan_and_execute(self.robot_status["robot_1"].task_id) #  uses robot_status to get task_id


    def _assign_task(self, task_id_to_assign):
        assign_time = rospy.Time.now().to_sec()
        status = self.robot_status.get('robot_1')
        if not status or status.state != 'idle':
            rospy.logwarn("[Manager] Robot not available for task assignment") #  log
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_ASSIGN_TASK_FAIL_ROBOT_NOT_IDLE, timestamp={assign_time:.3f}, task_id='{task_id_to_assign}', robot_state={status.state if status else 'UNKNOWN'}")
            return False

        rospy.loginfo(f"[Manager] Assigning task {task_id_to_assign}") #  log
        self.is_busy = True
        self.current_task = task_id_to_assign # Using  self.current_task
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_TASK_ASSIGNED, timestamp={assign_time:.3f}, task_id='{self.current_task}'")
        return True

    def _plan_and_execute(self, task_id_being_processed): # Renamed param to avoid clash with self.current_task
        plan_exec_start_time = rospy.Time.now().to_sec()
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_PLAN_EXEC_START, timestamp={plan_exec_start_time:.3f}, task_id='{task_id_being_processed}'")

        if not self.current_pose:
            rospy.logerr(f"[Manager] Cannot plan and execute for task {task_id_being_processed}: current_pose is None.")
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_PLAN_EXEC_FAIL_NO_POSE, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}'")
            #  logic for resetting state on failure was typically conditional on self.current_task
            # To preserve  logic, if no_pose, it just returns. State changes would happen if current_task was set.
            if self.current_task == task_id_being_processed: # If this was the task we just set...
                 self.is_busy = False
                 self.current_task = None
            return

        rospy.loginfo(f"[Manager] Checking path existence with PAT for task {task_id_being_processed}") #  log
        pat_wait_start_time = time.monotonic()
        rospy.wait_for_service('/validator_agent/validate_path_with_pat') # , blocks indefinitely
        pat_wait_duration = time.monotonic() - pat_wait_start_time
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_PAT_SERVICE_WAIT_END, timestamp={rospy.Time.now().to_sec():.3f}, duration_sec={pat_wait_duration:.3f}")
        rospy.loginfo("[Manager] Waiting for validate_path_with_pat service") #  log (now redundant due to wait_for_service log)

        validate_path_srv = rospy.ServiceProxy('/validator_agent/validate_path_with_pat', ValidatePathWithPat)

        start_pose_msg = Pose()
        start_pose_msg.position.x = self.current_pose['x']
        start_pose_msg.position.y = self.current_pose['y']
        start_pose_msg.position.z = 0.0
        start_pose_msg.orientation = self._quat_from_yaw(self.current_pose['theta'])

        goal_location = next((t for t in self.task_list if t['id'] == task_id_being_processed), None)
        if not goal_location:
            rospy.logerr(f"[Manager] Task {task_id_being_processed} not found in task list") #  log
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_PLAN_EXEC_FAIL_TASK_NOT_FOUND, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}'")
            #  behavior for this specific error case: simply returns.
            # State reset (is_busy, current_task) would only occur if self.current_task matched task_id_being_processed.
             # Initial idle status
            idle = RobotStatus(
                robot_id='robot_1',
                state='idle',
                task_id=''
            )
            self.status_pub.publish(idle)
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_ROBOT_STATUS_PUBLISHED, timestamp={rospy.Time.now().to_sec():.3f}, robot_id=robot_1, state=idle, task_id=''")
            if self.current_task == task_id_being_processed:
                 self.is_busy = False
                 self.current_task = None
            return

        goal_pose_msg = Pose()
        goal_pose_msg.position.x = goal_location['x'] * 0.25 + (-10) #  calculation
        goal_pose_msg.position.y = goal_location['y'] * 0.25 + (-10) #  calculation
        goal_pose_msg.position.z = 0.0
        goal_pose_msg.orientation = self._quat_from_yaw(0.0)

        start_stamped = PoseStamped(header=rospy.Header(frame_id="map", stamp=rospy.Time.now()), pose=start_pose_msg)
        goal_stamped = PoseStamped(header=rospy.Header(frame_id="map", stamp=rospy.Time.now()), pose=goal_pose_msg)

        rospy.loginfo(f"[PERF_LOG] event=MANAGER_PAT_REQUEST_SENT, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}', start_pose={format_pose_for_log(start_pose_msg)}, goal_pose={format_pose_for_log(goal_pose_msg)}")
        pat_call_start_time = time.monotonic()
        pat_resp = None
        try:
            pat_resp = validate_path_srv(start_pose=start_stamped, goal_pose=goal_stamped)
        except rospy.ServiceException as e:
            pat_call_duration = time.monotonic() - pat_call_start_time
            rospy.logerr(f"[Manager] PAT service call failed: {e}") # Not in , but good for context
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_PAT_SERVICE_CALL_FAIL, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}', duration_sec={pat_call_duration:.3f}, error='{e}'")
            if self.current_task == task_id_being_processed: #  style check
                self.is_busy = False
                self.current_task = None
            return

        pat_call_duration = time.monotonic() - pat_call_start_time
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_PAT_RESPONSE_RECEIVED, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}', duration_sec={pat_call_duration:.3f}, exists={pat_resp.exists if pat_resp else 'N/A'}, message='{pat_resp.message if pat_resp else 'N/A'}'")


        if not pat_resp.exists:
            rospy.logerr(f"[Manager] PAT verification shows no path exists for task {task_id_being_processed}: {pat_resp.message}") #  log
            self.remove_task(task_id_being_processed)
             # Initial idle status
            idle = RobotStatus(
                robot_id='robot_1',
                state='idle',
                task_id=''
            )
            self.status_pub.publish(idle)
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_ROBOT_STATUS_PUBLISHED, timestamp={rospy.Time.now().to_sec():.3f}, robot_id=robot_1, state=idle, task_id=''")
            if self.current_task: #  check
                self.is_busy = False
                self.is_replanning = False #  reset replanning flag
                self.current_task = None
            return

        rospy.loginfo("[Manager] PAT verification confirms path exists, proceeding with planning") #  log

        planner_wait_start_time = time.monotonic()
        rospy.wait_for_service('/get_path') # , blocks indefinitely
        planner_wait_duration = time.monotonic() - planner_wait_start_time
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_PLANNER_SERVICE_WAIT_END, timestamp={rospy.Time.now().to_sec():.3f}, duration_sec={planner_wait_duration:.3f}")

        get_path_srv = rospy.ServiceProxy('/get_path', GetPath)
        rospy.loginfo(f"[Manager] Planning path for task {task_id_being_processed}") #  log
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_PLANNER_REQUEST_SENT, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}', start_pose={format_pose_for_log(start_pose_msg)}, goal_pose={format_pose_for_log(goal_pose_msg)}")
        planner_call_start_time = time.monotonic()
        path_resp = None
        try:
            path_resp = get_path_srv(start_pose=start_pose_msg, goal_pose=goal_pose_msg)
        except rospy.ServiceException as e:
            planner_call_duration = time.monotonic() - planner_call_start_time
            rospy.logerr(f"[Manager] Planner service call failed: {e}") # Not in , but good for context
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_PLANNER_SERVICE_CALL_FAIL, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}', duration_sec={planner_call_duration:.3f}, error='{e}'")
            if not self.is_replanning: #  check
                if self.current_task == task_id_being_processed:
                    self.is_busy = False
                    self.current_task = None
            return

        planner_call_duration = time.monotonic() - planner_call_start_time
        num_waypoints = len(path_resp.waypoints) if path_resp and path_resp.waypoints else 0
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_PLANNER_RESPONSE_RECEIVED, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}', duration_sec={planner_call_duration:.3f}, success={num_waypoints > 0}, num_waypoints={num_waypoints}")


        if not path_resp.waypoints:
            rospy.logerr(f"[Manager] Planner failed to find path for {task_id_being_processed}") #  log
            if not self.is_replanning: #  logic
                if self.current_task:
                    self.is_busy = False
                    self.current_task = None
            return

        rospy.loginfo(f"[Manager] Planner found {len(path_resp.waypoints)} waypoints for task {task_id_being_processed}") #  log

        # Publish the planned path for visualization in RViz
        if hasattr(path_resp, 'path'):
            path_msg = path_resp.path
            if not path_msg.header.frame_id:
                path_msg.header.frame_id = "map"
            path_msg.header.stamp = rospy.Time.now()
            self.path_pub.publish(path_msg)
            rospy.loginfo("[Manager] Published planned path for visualization")

        if len(path_resp.waypoints) == 1 and path_resp.descriptions and path_resp.descriptions[0] == "Already at goal":
            rospy.loginfo(f"[Manager] Already at goal location for task {task_id_being_processed}") #  log
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_TASK_ALREADY_AT_GOAL, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}'")
            self.remove_task(self.current_task) #  uses self.current_task
            self.is_busy = False
            self.current_task = None
            return

        rospy.loginfo(f"[Manager] Executing path for task {task_id_being_processed}") #  log
        executor_wait_start_time = time.monotonic()
        rospy.wait_for_service('/execute_path') # , blocks indefinitely
        executor_wait_duration = time.monotonic() - executor_wait_start_time
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_EXECUTOR_SERVICE_WAIT_END, timestamp={rospy.Time.now().to_sec():.3f}, duration_sec={executor_wait_duration:.3f}")
        rospy.loginfo("[Manager] Waiting for execute_path service") #  log

        exec_path_srv = rospy.ServiceProxy('/execute_path', ExecutePath)
        rospy.loginfo("[Manager] Got execute_path service") #  log
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_EXECUTOR_REQUEST_SENT, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}', num_waypoints={len(path_resp.waypoints)}")
        executor_call_start_time = time.monotonic()
        exec_resp = None
        try:
            exec_resp = exec_path_srv(
                task_id=task_id_being_processed,
                waypoints=path_resp.waypoints,
                suggested_actions=path_resp.suggested_actions,
                descriptions=path_resp.descriptions
            )
        except rospy.ServiceException as e:
            executor_call_duration = time.monotonic() - executor_call_start_time
            rospy.logerr(f"[Manager] Executor service call failed: {e}") # Not in 
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_EXECUTOR_SERVICE_CALL_FAIL, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}', duration_sec={executor_call_duration:.3f}, error='{e}'")
            if self.current_task == task_id_being_processed: #  style check
                self.is_busy = False
                self.current_task = None
            return

        executor_call_duration = time.monotonic() - executor_call_start_time
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_EXECUTOR_RESPONSE_RECEIVED, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}', duration_sec={executor_call_duration:.3f}, success={exec_resp.success if exec_resp else 'N/A'}")
        rospy.loginfo(f"[Manager] Execution result for {task_id_being_processed}: {exec_resp.success if exec_resp else 'N/A'}") #  log (added None check)


        if exec_resp.success: #  logic branch
            status = self.robot_status.get('robot_1')
            if status and status.state == 'replan':
                rospy.loginfo("[Manager] Execution returned success but task not complete, retrying with new path") #  log
                rospy.loginfo(f"[PERF_LOG] event=MANAGER_EXEC_SUCCESS_AWAIT_STATUS_OR_REPLAN, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}', robot_state={status.state if status else 'UNKNOWN'}")
                # self.is_replanning = True #  set flag
                # rospy.Timer(
                #     rospy.Duration(1.0),
                #     lambda _: self._plan_and_execute(task_id_being_processed), #  passes task_id_being_processed
                #     oneshot=True
                # )
                return
            
            if task_id_being_processed not in [t['id'] for t in self.task_list]:
                rospy.loginfo(f"[Manager] Task {task_id_being_processed} was completed during execution") #  log
                rospy.loginfo(f"[PERF_LOG] event=MANAGER_TASK_ALREADY_REMOVED_POST_EXEC, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}'")
                if self.current_task == task_id_being_processed:
                    self.current_task = None
                return

            if status and status.state == 'idle':
                rospy.loginfo(f"[Manager] Task {task_id_being_processed} completed normally") #  log
                rospy.loginfo(f"[PERF_LOG] event=MANAGER_TASK_COMPLETED_POST_EXEC_SUCCESS, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}'")
                self.remove_task(task_id_being_processed) #  used task_id param
                self.current_task = None
                return

        else: #  logic branch for exec_resp.success == False
            rospy.logerr(f"[Manager] Execution failed for {task_id_being_processed}") #  log
            rospy.loginfo(f"[PERF_LOG] event=MANAGER_TASK_EXECUTION_FAILED, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}'")
            if self.current_task: #  check
                self.is_busy = False
                self.current_task = None
            return
        #  code did not have a finally block here for the entire _plan_and_execute
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_PLAN_EXEC_END, timestamp={rospy.Time.now().to_sec():.3f}, task_id='{task_id_being_processed}', total_duration_sec={rospy.Time.now().to_sec() - plan_exec_start_time:.3f}")

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        ManagerAgent().spin()
    except rospy.ROSInterruptException:
        #  code did not have a log here for ROSInterruptException
        rospy.loginfo(f"[PERF_LOG] event=MANAGER_ROS_INTERRUPT, timestamp={rospy.Time.now().to_sec():.3f}")
    #  code did not have a finally block here
    rospy.loginfo(f"[PERF_LOG] event=MANAGER_SHUTDOWN, timestamp={rospy.Time.now().to_sec():.3f}")