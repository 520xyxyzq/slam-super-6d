#!/usr/bin/env python2.7

# NB (Odin): Shamelessly stolen from https://answers.ros.org/question/389557/how-to-visualize-transformstamped-in-rviz/, 2/4/22

import rospy
from geometry_msgs.msg import TransformStamped
from vicon_bridge.msg import Markers
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from tf2_msgs.msg import TFMessage

class TFconverter:
    def __init__(self):
        self.pub_tf = rospy.Publisher("/tf", TFMessage, queue_size=1)
        self.sub_msg = rospy.Subscriber("/vicon/ZED2_ZQ/ZED2_ZQ", TransformStamped, self.my_callback)
        self.sub_msg2 = rospy.Subscriber("/vicon/markers", Markers, self.my_callback2)
        self.pub_markers = rospy.Publisher("/vicon/markers_world", Marker, queue_size=1)

    def my_callback(self, data):
        # self.pub_tf.publish(TFMessage([data]))  # works but no 'nicely readable'

        # clearer
        tf_msg = TFMessage()
        data.header.frame_id = "world" # Hack to make it visualizable
        tf_msg.transforms.append(data)
        self.pub_tf.publish(tf_msg)

    def my_callback2(self, data):
        # self.pub_tf.publish(TFMessage([data]))  # works but no 'nicely readable'

        # clearer
        # data.header.frame_id = "world" # Hack to make it visualizable
        marker_msg = Marker()
        marker_msg.header = data.header
        marker_msg.header.frame_id = "world"
        marker_msg.ns = "vicon_markers"
        marker_msg.pose.orientation.w = 1.0 # Valid quaternion

        marker_msg.points = []
        for marker in data.markers:
            point = marker.translation
            # The points are in mm, needs them in m
            point.x /= 1000
            point.y /= 1000
            point.z /= 1000
            marker_msg.points.append(point)

        marker_msg.type = Marker.SPHERE_LIST
        color = ColorRGBA()
        color.r = 1.0
        color.g = 0.2
        color.b = 0.0
        color.a = 1.0
        marker_msg.color = color
        sphere_diameter = 15e-3 # m
        sphere_radius = sphere_diameter/2
        marker_msg.scale.x = sphere_radius
        marker_msg.scale.y = sphere_radius
        marker_msg.scale.z = sphere_radius

        self.pub_markers.publish(marker_msg)

if __name__ == '__main__':
    rospy.init_node('transform_to_TF_node')
    tfb = TFconverter()
    rospy.spin()