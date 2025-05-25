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

        #  params (including locations dict)
        self.world_name = rospy.get_param('~world_name')
        self.map_cfg    = rospy.get_param(self.world_name)

        # task list once
        self.task_list = [
            {'id': name, 'x': coords[0], 'y': coords[1]}
            for name, coords in self.map_cfg['locations'].items()
        ]

        # Ollama client & model choice
        self.client = ollama.Client()
        self.model  = rospy.get_param('~model', 'llama3.2')

        # Publish an initial idle status so ManagerAgent will accept
        self.status_pub = rospy.Publisher('/robot_1/status',
                                          RobotStatus,
                                          queue_size=1,
                                          latch=True)
        idle = RobotStatus(robot_id='robot_1',
                           state='idle',
                           task_id='',
                           battery_pct=100)
        self.status_pub.publish(idle)

        # State holder & subscriptions
        self.current_pose = None
        rospy.Subscriber('/amcl_pose',
                         PoseWithCovarianceStamped,
                         self.pose_cb,
                         queue_size=1)

        #AssignTask service proxy
        rospy.wait_for_service('/assign_task')
        self.assign_srv = rospy.ServiceProxy('/assign_task', AssignTask)

        #Poll the LLM every 100 seconds
        self.timer = rospy.Timer(rospy.Duration(2.0), self.decide_cb)

        rospy.loginfo(f"[OLLAMA] Ready in world '{self.world_name}'")

    def pose_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = self._yaw_from_quat(q)
        self.current_pose = {'x': p.x, 'y': p.y, 'theta': yaw}

    def decide_cb(self, _):
        # guard: we need a pose to ask the LLM
        if self.current_pose is None:
            return

        # Prompt the LLM as before…
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

        # Extract next_id via JSON or regex
        try:
            data    = json.loads(text)
            next_id = data['next_task']
        except Exception:
            m = re.search(r'"next_task"\s*:\s*"([^"]+)"', text)
            if not m:
                rospy.logerr(f"[OLLAMA] Could not parse:\n>> {text}")
                return
            next_id = m.group(1)

        # Assign the task
        req = AssignTaskRequest(robot_id="robot_1",
                                task_id=next_id,
                                location_name=next_id)
        res = self.assign_srv(req)
        if not res.accepted:
            rospy.logwarn(f"[OLLAMA] Assignment failed: {res.message}")
            return

        rospy.loginfo(f"[OLLAMA] Assigned {next_id}: {res.message}")
        self.busy = True

        # Plan via GetPath
        rospy.wait_for_service('/get_path')
        get_path = rospy.ServiceProxy('/get_path', GetPath)
        path_resp = get_path(goal_name=next_id)

        # Execute via ExecutePath
        rospy.wait_for_service('/execute_path')
        exec_path = rospy.ServiceProxy('/execute_path', ExecutePath)
        exec_resp = exec_path(
            waypoints=path_resp.waypoints,
            suggested_actions=path_resp.suggested_actions,
            descriptions=path_resp.descriptions
        )

        if exec_resp.success:
            rospy.loginfo(f"[OLLAMA] Started execution of {next_id}")
            # —— TEST PAUSE: let the robot move for 25 now ——
            rospy.loginfo("[OLLAMA] Will re-query the LLM in 100 s…")
            rospy.Timer(rospy.Duration(25.0),
                        self.decide_cb,
                        oneshot=True)
        else:
            rospy.logerr(f"[OLLAMA] Execution failed: {exec_resp}")
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
