#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid

class GridOverlayPublisher:
    def __init__(self):
        rospy.init_node('grid_overlay_publisher')
        self.pub = rospy.Publisher('/grid_overlay', MarkerArray, queue_size=1)

        # parameters
        self.world_name = rospy.get_param("~world_name")
        self.config = rospy.get_param(self.world_name)
        self.cell_size = self.config["cell_size"]
        
        self.grid_origin_x = self.config["grid_origin_x"]
        self.grid_origin_y = self.config["grid_origin_y"]
        self.visible_width = self.config["grid_width"]
        self.visible_height = self.config["grid_height"]

        # default origin until map arrives
        self.origin_x = 0.0
        self.origin_y = 0.0

        # subscribe to map to get origin
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)

    def map_callback(self, msg: OccupancyGrid):
        map_info = msg.info
        self.origin_x = map_info.origin.position.x
        self.origin_y = map_info.origin.position.y
        
        # number of grid cells across = total meters / cell_size
        self.grid_width  = int(round(map_info.width  * map_info.resolution / self.cell_size))
        self.grid_height = int(round(map_info.height * map_info.resolution / self.cell_size))
        
        rospy.loginfo(f"[GridOverlay] map origin=({self.origin_x:.2f},{self.origin_y:.2f}), grid size={self.grid_width}×{self.grid_height}")

    def make_label(self, text, i, j):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = "grid_labels"
        m.id = i * 1000 + j
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.scale.z = 0.1
        m.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        # position each label at cell center, offset by map origin
        m.pose.position.x = self.origin_x + i * self.cell_size + self.cell_size / 2.0
        m.pose.position.y = self.origin_y + j * self.cell_size + self.cell_size / 2.0
        m.pose.position.z = 0.1
        m.text = text
        m.lifetime = rospy.Duration(0)
        return m

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if not hasattr(self, 'grid_width') or not hasattr(self, 'grid_height'):
                rospy.logdebug("[GridOverlay] waiting for map...")
                rate.sleep()
                continue

            # Compute starting indices based on origin offset
            start_i = int((self.grid_origin_x - self.origin_x) / self.cell_size)
            start_j = int((self.grid_origin_y - self.origin_y) / self.cell_size)

            markers = MarkerArray()
            for i in range(int(round(self.visible_width/self.cell_size))):
                for j in range(int(round(self.visible_height/self.cell_size))):
                    global_i = start_i + i
                    global_j = start_j + j
                    label = self.make_label(f"({global_i},{global_j})", global_i, global_j)
                    markers.markers.append(label)

            self.pub.publish(markers)
            rate.sleep()


if __name__ == "__main__":
    GridOverlayPublisher().run()

