#!/usr/bin/env python2.7

import rospy
import rospkg
from geometry_msgs.msg import PoseStamped

import numpy as np


class Logger(object):
    def __init__(self):
        rospy.Subscriber("/svo/pose_cam/0", PoseStamped, self.pose_cb)
        self.data = []

    def pose_cb(self, pose_msg):
        timestamp = pose_msg.header.stamp.secs

        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        z = pose_msg.pose.position.z
    
        qw = pose_msg.pose.orientation.w
        qx = pose_msg.pose.orientation.x
        qy = pose_msg.pose.orientation.y
        qz = pose_msg.pose.orientation.z

        # TUM format: timestamp x y z q_x q_y q_z q_w 
        self.data += [timestamp, x, y, z, qx, qy, qz, qw]

    def save_to_file(self, path):
        data = np.array(self.data).reshape((-1, 1 + 3 + 4))
        np.savetxt(path + "/odom_raw.txt", data)

if __name__ == '__main__':
    rospy.init_node('pose_logger')
    logger = Logger()
    try:
        while not rospy.is_shutdown():
            pass
    except rospy.ROSException:
        pass

    jackal_path = rospkg.RosPack().get_path('jackal_ros')
    rospy.loginfo("Logging to " + jackal_path + "/../log")
    logger.save_to_file(jackal_path + "/../log")
