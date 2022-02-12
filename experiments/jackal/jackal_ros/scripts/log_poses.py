#!/usr/bin/env python2.7

import rospy
import rospkg
from geometry_msgs.msg import PoseStamped

import numpy as np

def get_save_path(jackal_path):
    if not rospy.has_param("ycb_item") or not rospy.has_param("number"):
        rospy.logwarn("Could not find param for YCB item or number, saving directly to {}".format(jackal_path))
        return jackal_path

    ycb_item = rospy.get_param("ycb_item")
    number = rospy.get_param("number")

    if ycb_item == "cracker":
        image_folder = "003_cracker_box_16k"
    elif ycb_item == "sugar":
        image_folder = "004_sugar_box_16k"
    elif ycb_item == "spam":
        image_folder = "010_potted_meat_can_16k"

    save_path = "{}/../{}/00{}".format(jackal_path, image_folder, number)

    return save_path
    

class Logger(object):
    def __init__(self):
        rospy.Subscriber("/svo/pose_cam/0", PoseStamped, self.pose_cb)
        self.data = []

    def pose_cb(self, pose_msg):
        timestamp = pose_msg.header.stamp.to_sec()

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
    save_path = get_save_path(jackal_path)
    rospy.loginfo("Saving to " + save_path)
    logger.save_to_file(save_path)
