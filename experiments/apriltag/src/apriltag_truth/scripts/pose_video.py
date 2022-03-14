#!/usr/bin/env python2
# Visualize poses from output of truth.py
# Ethan Yang ethany@mit.edu

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import CameraInfo, Image


class PoseVideo(object):

    def __init__(self):
        rospy.Subscriber("/zed2/zed_node/left/uncompressed",
                         Image, self.image_cb)
        rospy.Subscriber("/zed2/zed_node/left/camera_info",
                         CameraInfo, self.info_cb)

        self.bridge = CvBridge()
        self.K = None
        file_name = rospy.get_param('bag_name', None) + '_gt.txt'
        self.input_file = open(rospy.get_param(
            'filtered_output_path')+file_name, "r")
        rospy.loginfo('Using file: ' + file_name)
        self.next_line = self.input_file.readline()

        # Uncomment for writing video to PATH
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # self.vid_out = cv2.VideoWriter('PATH', fourcc, 15, (672, 376))

        self.line_num = 1
        self.image_num = 1

    def show_image(self, img):
        # Show image in OpenCV window
        # self.vid_out.write(img)
        cv2.imshow("Image Window", img)
        cv2.waitKey(3)

    def info_cb(self, msg):
        # get camera matrix
        self.K = np.array(msg.K).reshape(3, 3)

    def image_cb(self, img_msg):
        # Convert the ROS Image message to a CV2 Image
        cur_time = img_msg.header.stamp.to_sec()
        cv_image = self.bridge.imgmsg_to_cv2(img_msg,  desired_encoding='bgr8')
        rospy.loginfo('On image_num: ' + str(self.image_num))
        self.image_num += 1
        while True:
            tokens = [float(i) for i in self.next_line.split()]
            timestamp = tokens[0]
            rospy.loginfo('On line: ' + str(self.line_num))
            if abs(timestamp - cur_time) >= 0.00001:
                rospy.loginfo('Mismatched timestamps')
                rospy.loginfo('Image time: ' + str(cur_time))
                rospy.loginfo('File time: ' + str(timestamp))
            if abs(timestamp - cur_time) < 0.00001:
                trans = tokens[1:4]
                rot = tokens[4:]
                if rot != [0., 0., 0., 0.]:
                    R = Rot.from_quat(rot)
                    t = np.array(trans)
                    assert self.K is not None
                    self.draw_axis(cv_image, R, t, self.K)
                self.line_num += 1
                self.next_line = self.input_file.readline()
                break
            elif timestamp < cur_time:
                # File has an extra timestamp
                rospy.loginfo('skipping an file time')
                self.line_num += 1
                self.next_line = self.input_file.readline()
                continue
            else:
                # File is missing a timestamp
                rospy.loginfo('skipping an image time')
                break
        # Show the converted image
        self.show_image(cv_image)

    def draw_axis(self, img, R, t, K):
        """
        Draw a set of coordinate axes on openCV img.
        Axes are drawn in BGR order.
        Input:
            img (openCV image) : image to be modified
            R (scipy quat) : rotation quaternion
            t (3-array) : translation
            K (3x3-array) : camera matrix
        Output:
            Modified input img
        """
        rotV = R.as_rotvec()
        points = np.float32([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
                             [0, 0, 0]]).reshape(-1, 3)
        axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[0].ravel()), (255, 0, 0), 10)
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[1].ravel()), (0, 255, 0), 3)
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[2].ravel()), (0, 0, 255), 3)
        return img

    def main(self):
        while not rospy.is_shutdown():
            continue


if __name__ == "__main__":
    rospy.init_node("pose_video_node", anonymous=True)
    node = PoseVideo()
    try:
        node.main()
    except rospy.ROSInterruptException:
        pass
