#!/usr/bin/env python2.7

from __future__ import print_function
import rospy
import rospkg
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from apriltag_ros.msg import AprilTagDetectionArray
import sys

VALID_YCB_ITEMS = ["cracker", "sugar", "spam"]
VALID_NUMBERS = [1, 2, 3, 4]
VALID_APRILTAG_IDS = {
    "cracker": [ 8,  9, 10, 11, 12, 13],
    "sugar":   [14, 15, 16, 17, 18, 19],
    "spam":    [ 0,  1,  2,  3,  4,  5]
}

TAG_WIDTH = 50e-3 # m
TAG_PAPER_EDGE_GAP = 10e-3 # m
PAPER_A4_HEIGHT = 297e-3 # m
PAPER_A4_WIDTH = 210e-3 # m


def Rt2T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, -1] = t

    return T

def pose_msg2T(pose_msg):
    t = np.array([
        pose_msg.pose.position.x,
        pose_msg.pose.position.y,
        pose_msg.pose.position.z
    ])

    R = Rot.from_quat([
        pose_msg.pose.orientation.x,
        pose_msg.pose.orientation.y,
        pose_msg.pose.orientation.z,
        pose_msg.pose.orientation.w
    ]).as_dcm()

    T = Rt2T(R, t)

    return T

class ObjectPoseEstimator(object):
    def __init__(self, ycb_item):
        if not rospy.has_param("ycb_item"):
            rospy.logerr("Needs YCB item to estimate object pose!")
            sys.exit(-1)

        ycb_item = rospy.get_param('ycb_item')
        if not ycb_item["name"] in VALID_YCB_ITEMS:
            rospy.logerr("Invalid YCB item, must be one of {}, received ''".format(VALID_YCB_ITEMS, ycb_item))
            sys.exit(-1)

        self.ycb_item = ycb_item
        self.apriltag_detections_sub = rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.apriltag_detections_cb, queue_size=1000)

        self.min_id = self.get_min_tag_id(ycb_item)
        self.T_tos = self.compute_T_tos()

        self.all_computed_transforms = []

        rospy.on_shutdown(self.save_to_file)

    def average_transforms(self, transforms):
        t_avg = transforms[:, :3, -1].mean(axis=0)

        R_avg = transforms[:, :3, :3].mean(axis=0)
        # Project onto SO(3)
        U, _, VT = np.linalg.svd(R_avg)
        D = np.eye(3)
        D[-1, -1] = np.linalg.det(U)*np.linalg.det(VT)
        R_avg = U@D@VT

        T_avg = Rt2T(R_avg, t_avg)

        return T_avg

    def T2idposquat(image_id, T):
        quat = Rot.from_dcm(T[:3, :3]).as_quat()
        pos = T[:3, -1]
        return np.hstack((image_id, pos, quat))

    def apriltag_detections_cb(self, apriltag_detections_msg):
        image_id = apriltag_detections_msg.header.seq

        transforms = np.empty((len(apriltag_detections_msg.detections), 4, 4))

        for k, apriltag_detection in enumerate(apriltag_detections_msg.detections):
            tag_id = apriltag_detection.size[0]
            if not tag_id in VALID_APRILTAG_IDS[self.ycb_item["name"]]:
                rospy.logwarn("Detected tag id {} which should not be possible for YCB item '{}'".format(tag_id, self.ycb_item["name"]))
                continue

            # Map tag_id to {0, ..., 5} for convenience
            tag_id = tag_id % self.min_id

            # T_ct = transform of tag (t) relative camera (c)
            T_ct = pose_msg2T(apriltag_detection.pose) # Needs to peel of one "pose" layer as original message is PoseWithCovarianceStamped
            # Object pose in camera frame
            T_co = T_ct.dot(self.T_tos[tag_id])
            transforms[k] = T_co

        T_co = self.average_transforms(transforms)

        self.all_computed_transforms.append(self.T2idposquat(image_id, T_co))


    def get_min_tag_id(self, ycb_item):
        if ycb_item["name"] == "cracker":
            return 8
        elif ycb_item["name"] == "sugar":
            return 14
        elif ycb_item["name"] == "spam":
            return 0

    def compute_T_tos(self):
        """
        Compute transform of object frame (o) relative tag frames (t) 
        """

        Rz = Rot.from_euler('z', -np.pi/2).as_dcm()
        Trz = np.eye(4)
        Trz[:3, :3] = Rz

        Rx = Rot.from_euler('x', np.pi/2).as_dcm()
        Trx = np.eye(4)
        Trx[:3, :3] = Rx
        
        Tt = np.eye(4)
        T_tos = np.empty((6, 4, 4))

        dz = -self.ycb_item["y"]/2.0 # y is height axis in object frame in negative z direction in tag frame

        for tag_id in range(6):
            if tag_id == 0:
                dx = PAPER_A4_WIDTH/2.0 - TAG_WIDTH/2.0 - TAG_PAPER_EDGE_GAP # in positive x direction in tag frame
                dy = -(PAPER_A4_HEIGHT/2.0 - TAG_WIDTH/2.0 - TAG_PAPER_EDGE_GAP) # in negative y direction in tag frame
            elif tag_id == 1:
                dx = -(PAPER_A4_WIDTH/2.0 - TAG_WIDTH/2.0 - TAG_PAPER_EDGE_GAP) # in negative x direction in tag frame
                dy = -(PAPER_A4_HEIGHT/2.0 - TAG_WIDTH/2.0 - TAG_PAPER_EDGE_GAP) # in negative y direction in tag frame
            elif tag_id == 2:
                dx = (PAPER_A4_WIDTH/2.0 - TAG_WIDTH/2.0 - TAG_PAPER_EDGE_GAP) # in positive x direction in tag frame
                dy = 0.0
            elif tag_id == 3:
                dx = -(PAPER_A4_WIDTH/2.0 - TAG_WIDTH/2.0 - TAG_PAPER_EDGE_GAP) # in negative x direction in tag frame
                dy = 0.0
            elif tag_id == 4:
                dx = (PAPER_A4_WIDTH/2.0 - TAG_WIDTH/2.0 - TAG_PAPER_EDGE_GAP) # in positive x direction in tag frame
                dy = (PAPER_A4_HEIGHT/2.0 - TAG_WIDTH/2.0 - TAG_PAPER_EDGE_GAP) # in posiive y direction in tag frame
            elif tag_id == 5:
                dx = -(PAPER_A4_WIDTH/2.0 - TAG_WIDTH/2.0 - TAG_PAPER_EDGE_GAP) # in negative x direction in tag frame
                dy = (PAPER_A4_HEIGHT/2.0 - TAG_WIDTH/2.0 - TAG_PAPER_EDGE_GAP) # in posiive y direction in tag frame

            t = np.array([dx, dy, dz])
            Tt[:3, -1] = t

            T_tos[tag_id] = Tt.dot(Trz).dot(Trx)

        return T_tos

    def save_to_file(self):
        mat = np.array(self.all_computed_transforms)
        path = rospkg.RosPack().get_path("jackal_ros") + "/../logs"
        np.savetxt(path + "./object_frame_image_id_raw.txt", mat)

if __name__ == "__main__":
    pass