# manager_agent.py
#!/usr/bin/env python3
import rospy
from warehouse_robot.srv import AssignTask, AssignTaskResponse
from warehouse_robot.msg import RobotStatus

class ManagerAgent:
    def __init__(self):
        rospy.init_node('manager_agent')
        # Subscribe to robot status updates
        self.status_sub = rospy.Subscriber(
            '/robot_1/status', RobotStatus, self.cb_status
        )
        # Service to assign tasks to robot_1
        self.assign_srv = rospy.Service(
            'assign_task', AssignTask, self.handle_assign
        )
        self.robot_status = {}
        rospy.loginfo("[Manager] Ready.")

    def cb_status(self, msg):
        # Store or update robot status in internal database
        self.robot_status[msg.robot_id] = msg
        rospy.loginfo(f"[Manager] Status from {msg.robot_id}: {msg.state}, task={msg.task_id}")

    def handle_assign(self, req):
        # Decide whether to accept the task
        status = self.robot_status.get(req.robot_id)
        if not status or status.state != 'idle':
            return AssignTaskResponse(False, "Robot not available")
        # Accept the task
        rospy.loginfo(f"[Manager] Assigning {req.task_id}@{req.location_name} to {req.robot_id}")
        return AssignTaskResponse(True, "Task accepted")

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    ManagerAgent().spin()
