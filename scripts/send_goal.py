#!/usr/bin/env python3
import rospy
import sys
from warehouse_robot.srv import GetPath, ExecutePath

def print_usage():
    print("Usage: rosrun warehouse_robot send_goal.py <location>")
    print("Example locations: storage_area, charging_station, etc.")
    print("If no location is provided, will use the goal_name parameter (default: storage_area)")

if __name__=='__main__':
    rospy.init_node('send_goal')
    
    # Get goal from command line argument or parameter
    if len(sys.argv) > 1:
        goal = sys.argv[1]
        rospy.loginfo(f"Using command line goal: {goal}")
    else:
        goal = rospy.get_param('~goal_name', 'storage_area')
        rospy.loginfo(f"Using parameter goal: {goal}")

    rospy.wait_for_service('/get_path')
    get_path = rospy.ServiceProxy('/get_path', GetPath)
    path_resp = get_path(goal_name=goal)

    rospy.loginfo(f"\nPath planning complete. Received {len(path_resp.waypoints)} waypoints:")
    for i, waypoint in enumerate(path_resp.waypoints):
        x = waypoint.pose.position.x
        y = waypoint.pose.position.y
        desc = path_resp.descriptions[i] if i < len(path_resp.descriptions) else ''
        rospy.loginfo(f"Waypoint {i+1}: ({x:.2f}, {y:.2f}) - {desc}")
    
    rospy.wait_for_service('/execute_path')
    exec_path = rospy.ServiceProxy('/execute_path', ExecutePath)
    result = exec_path(
        waypoints=path_resp.waypoints,
        suggested_actions=path_resp.suggested_actions,
        descriptions=path_resp.descriptions
    )
    
    if result.success:
        rospy.loginfo("Goal reached successfully!")
    else:
        rospy.logerr("Failed to reach goal")

