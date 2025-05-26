#!/usr/bin/env python3
import rospy, json, re
from warehouse_robot.srv import AssignTask, AssignTaskRequest
from warehouse_robot.msg import RobotStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from warehouse_robot.srv import GetPath, ExecutePath
import ollama

class OllamaAgent:
    def __init__(self):
        rospy.init_node('ollama_agent')
        # 1) load map config
        self.world_name = rospy.get_param('~world_name')
        self.map_cfg    = rospy.get_param(self.world_name)

        # 2) build and track tasks
        self.task_list       = [
            {'id': n, 'x': c[0], 'y': c[1]}
            for n,c in self.map_cfg['locations'].items()
        ]
        self.completed_tasks = set()

        # 3) LLM client
        self.client = ollama.Client()
        self.model  = rospy.get_param('~model','llama3.2')

        # 4) initial status publisher (idle)
        self.status_pub = rospy.Publisher(
            '/robot_1/status', RobotStatus, queue_size=1, latch=True
        )
        idle = RobotStatus(robot_id='robot_1',
                           state='idle',
                           task_id='',
                           battery_pct=100)
        self.status_pub.publish(idle)

        # 5) state + subscriptions
        self.current_pose = None
        self.is_busy      = False
        self.current_task = None
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped,
                         self.pose_cb, queue_size=1)
        rospy.Subscriber('/robot_1/status', RobotStatus,
                         self.status_cb, queue_size=1)

        # 6) AssignTask proxy
        rospy.wait_for_service('/assign_task')
        self.assign_srv = rospy.ServiceProxy('/assign_task', AssignTask)

        # 7) poll LLM every 2 s
        self.timer = rospy.Timer(rospy.Duration(2.0),
                                 self.decide_cb)

        rospy.loginfo(f"[OLLAMA] Ready in world '{self.world_name}'")

    def pose_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = self._yaw_from_quat(q)
        self.current_pose = {'x':p.x,'y':p.y,'theta':yaw}

    def remove_task(self, task_id):
        self.completed_tasks.add(task_id)
        self.task_list = [t for t in self.task_list
                          if t['id'] not in self.completed_tasks]
        rospy.loginfo(f"[OLLAMA] Removed {task_id}. {len(self.task_list)} left")

    def decide_cb(self, _):
        # guard: need pose + not busy + tasks remain
        if not self.current_pose or self.is_busy or not self.task_list:
            return

        # build prompt (your original detailed instruction)
        prompt_payload = {
            'robot_pose':    self.current_pose,
            'pending_tasks': self.task_list,
            'map_config':    self.map_cfg,
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

        # parse JSON
        try:
            next_id = json.loads(text)['next_task']
        except Exception:
            m = re.search(r'"next_task"\s*:\s*"([^"]+)"', text)
            if not m:
                rospy.logerr(f"[OLLAMA] Could not parse LLM response:\n>>{text}")
                return
            next_id = m.group(1)

        # assign
        req = AssignTaskRequest(robot_id="robot_1",
                                task_id=next_id,
                                location_name=next_id)
        res = self.assign_srv(req)
        if not res.accepted:
            rospy.logwarn(f"[OLLAMA] Assignment failed: {res.message}")
            return

        rospy.loginfo(f"[OLLAMA] Assigned {next_id}")
        self.is_busy      = True
        self.current_task = next_id

        # plan
        rospy.wait_for_service('/get_path')
        get_path = rospy.ServiceProxy('/get_path', GetPath)
        path_resp = get_path(goal_name=next_id)

        # execute
        rospy.wait_for_service('/execute_path')
        exec_path = rospy.ServiceProxy('/execute_path', ExecutePath)
        ok = exec_path(
            task_id=self.current_task,
            waypoints=path_resp.waypoints,
            suggested_actions=path_resp.suggested_actions,
            descriptions=path_resp.descriptions
        ).success

        if ok:
            rospy.loginfo(f"[OLLAMA] Execution started for {next_id}")
        else:
            rospy.logerr(f"[OLLAMA] Execution failed for {next_id}")
            # clear busy so we can retry
            self.is_busy      = False
            self.current_task = None

    def status_cb(self, msg: RobotStatus):
        if msg.robot_id != 'robot_1':
            return
        was_busy     = self.is_busy
        self.is_busy = (msg.state == 'busy')
        # on busy→idle, complete the task
        if was_busy and not self.is_busy and self.current_task:
            rospy.loginfo(f"[OLLAMA] Completed {self.current_task}")
            self.remove_task(self.current_task)
            self.current_task = None
            # trigger a one-off re-decision after 5s
            if self.task_list:
                rospy.Timer(rospy.Duration(5.0),
                            self.decide_cb,
                            oneshot=True)

    @staticmethod
    def _yaw_from_quat(q):
        import numpy as np
        return np.arctan2(2*(q.w*q.z + q.x*q.y),
                          1 - 2*(q.y*q.y + q.z*q.z))

if __name__ == '__main__':
    try:
        OllamaAgent()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
