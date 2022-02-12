#!/usr/bin/env python2.7

from __future__ import print_function
import rospy
import rospkg
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from apriltag_ros.msg import AprilTagDetectionArray
from sensor_msgs.msg import Image
import sys
import cv2
import cv_bridge


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


CAMERA_INTRINSICS = [
    276.32251, # fx
    276.32251, # fy
    353.70087, # cx
    179.08852  # cy
]


def Rt2T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, -1] = t

    return T

def T2Rt(T):
    R = T[:3, :3]
    t = T[:3, -1]

    return R, t

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

def project(K, X):
    """
    Computes the pinhole projection of a (3 or 4)xN array X using
    the camera intrinsic matrix K. Returns the pixel coordinates
    as an array of size 2xN.
    """
    if len(X.shape) == 3:
        uvw = K[None].dot(X[:,:3,:])
        uvw /= uvw[:,None,2,:]
        uv = uvw[:,:2,:]
        uv = np.vstack((uv[:,0,:].ravel(), uv[:,1,:].ravel()))
    else:
        uvw = K.dot(X[:3,:])
        uvw /= uvw[2,:]
        uv = uvw[:2,:]
    return uv.astype(int)


def draw_frame(img, K, R, t, scale=1):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.
    Control the length of the axes by specifying the scale argument.
    """
    T = Rt2T(R, t)
    X = T.dot(np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]]))
    u,v = project(K, X)

    print("u: {}\nv: {}".format(u, v))

    img = cv2.line(img, (int(u[0]), int(v[0])), (int(u[1]), int(v[1])), (255,0,0), 5)
    img = cv2.line(img, (int(u[0]), int(v[0])), (int(u[2]), int(v[2])), (0,255,0), 5)
    img = cv2.line(img, (int(u[0]), int(v[0])), (int(u[3]), int(v[3])), (0,0,255), 5)

    return img

class ObjectPoseEstimator(object):
    def __init__(self):
        if not rospy.has_param("ycb_item"):
            rospy.logerr("Needs YCB item to estimate object pose!")
            sys.exit(-1)

        ycb_item = rospy.get_param('ycb_item')
        if not ycb_item["name"] in VALID_YCB_ITEMS:
            rospy.logerr("Invalid YCB item, must be one of {}, received ''".format(VALID_YCB_ITEMS, ycb_item))
            sys.exit(-1)

        self.ycb_item = ycb_item
        self.apriltag_detections_sub = rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.apriltag_detections_cb, queue_size=1000)
        self.object_pose_pub = rospy.Publisher("/object_pose_estimates", Image, queue_size=1000)

        self.min_id = self.get_min_tag_id(ycb_item)
        self.T_tos = self.compute_T_tos()

        self.all_computed_transforms = []

        fx, fy, cx, cy = CAMERA_INTRINSICS
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])

        rospy.on_shutdown(self.save_to_file)

    # Hardcoding, but just for visualization
    def draw_transform_to_img(self, image_id, T):
        img = cv2.imread("/home/odinase/mrg_ws/src/slam-super-6d/experiments/jackal/003_cracker_box_16k/001/left/{}.png".format(str(image_id).zfill(5)), cv2.IMREAD_COLOR)
        R, t = T2Rt(T)
        img = draw_frame(img, self.K, R, t)
        return img

    def average_transforms(self, transforms):
        t_avg = transforms[:, :3, -1].mean(axis=0)

        R_avg = transforms[:, :3, :3].mean(axis=0)
        # Project onto SO(3)
        U, _, VT = np.linalg.svd(R_avg)
        D = np.eye(3)
        D[-1, -1] = np.linalg.det(U)*np.linalg.det(VT)
        R_avg = U.dot(D).dot(VT)

        T_avg = Rt2T(R_avg, t_avg)

        return T_avg

    def T2idposquat(self, image_id, T):
        quat = Rot.from_dcm(T[:3, :3]).as_quat()
        pos = T[:3, -1]
        return np.hstack((image_id, pos, quat))

    def apriltag_detections_cb(self, apriltag_detections_msg):
        if len(apriltag_detections_msg.detections) == 0:
            return
        image_id = apriltag_detections_msg.header.seq

        transforms = np.empty((len(apriltag_detections_msg.detections), 4, 4))

        for k, apriltag_detection in enumerate(apriltag_detections_msg.detections):
            tag_id = apriltag_detection.id[0]
            if not tag_id in VALID_APRILTAG_IDS[self.ycb_item["name"]]:
                rospy.logwarn("Detected tag id {} which should not be possible for YCB item '{}'".format(tag_id, self.ycb_item["name"]))
                continue

            # Map tag_id to {0, ..., 5} for convenience
            tag_id = tag_id % self.min_id

            # T_ct = transform of tag (t) relative camera (c)
            T_ct = pose_msg2T(apriltag_detection.pose.pose) # Needs to peel of one "pose" layer as original message is PoseWithCovarianceStamped
            # Object pose in camera frame
            T_co = T_ct.dot(self.T_tos[tag_id])
            transforms[k] = T_co

        T_co = self.average_transforms(transforms)

        img = self.draw_transform_to_img(image_id, T_co)
        bridge = cv_bridge.CvBridge()
        image_message = bridge.cv2_to_imgmsg(img, encoding="passthrough")

        self.object_pose_pub.publish(image_message)

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
    rospy.init_node("object_pose_estimator")

    object_estimator = ObjectPoseEstimator()

    rospy.spin()