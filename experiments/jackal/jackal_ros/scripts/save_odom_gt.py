#!/usr/bin/env python2.7

from __future__ import print_function
from multiprocessing.sharedctypes import Value
import rospy
import rosbag
import sys
import os
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as Rot
import numpy as np


def parse_inputs(argv):
    if not len(sys.argv) == 3:
        print("Needs to pass in arg for YCB object and dataset number, received now {} args".format(len(sys.argv)))
        sys.exit(-1)

    real_path = os.path.realpath(__file__)
    dirname = os.path.dirname(real_path)
    ycb_item = sys.argv[1]

    if not ycb_item in ["cracker", "sugar", "spam"]:
        print("YCB item needs to be one of 'cracker', 'sugar' or 'spam', is now {}".format(ycb_item)) 
        sys.exit(-1)

    try:
        number = int(argv[2])
    except ValueError:
        print("number passed in as arg 2 is not valid int, passed in {}".format(argv[2]))
        sys.exit(-1)

    if not number in [1, 2, 3, 4]:
        print("number must be one of 1, 2, 3, 4, is now {}".format(number))
        sys.exit(-1)

    print("Doing YCB item '{}', dataset number {}, in directory {}".format(ycb_item, number, dirname))

    return ycb_item, number, dirname


def invT(T):
    R = T[:3, :3]
    t = T[:3, -1]

    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, -1] = -R.T.dot(t)

    return Tinv

def compute_odom2cam():
    translation = np.eye(4)
    translation[:3,-1] = np.array([0.0, 60.0e-3, 0.0]) # Left cam is 60 mm to the left of the center

    Tt = translation
    Ry = Rot.from_euler('y', np.pi/2).as_dcm()
    Ty = np.eye(4)
    Ty[:3, :3] = Ry

    Rz = Rot.from_euler('z', -np.pi/2).as_dcm()
    Tz = np.eye(4)
    Tz[:3, :3] = Rz

    T_oc = Tt.dot(Ty).dot(Tz) # Transform from camera to odometry

    T_co = invT(T_oc)

    return T_co

def compute_vicon2cam():
    """
    Note that the transform computed here is approximate!
    """
    translation = np.eye(4)
    translation[:3,-1] = np.array([0.0, 60.0e-3, 0.0]) # Left cam is 60 mm to the left of the center

    Tt = translation
    Rx = Rot.from_euler('x', np.pi/2).as_dcm()
    Tx = np.eye(4)
    Tx[:3, :3] = Rx

    Rz = Rot.from_euler('z', np.pi).as_dcm()
    Tz = np.eye(4)
    Tz[:3, :3] = Rz

    T_vc = Tt.dot(Tx).dot(Tz) # Transform from camera to odometry

    T_cv = invT(T_vc)

    return T_cv

def pos_quat2T(pos, quat):
    t = np.array([pos.x, pos.y, pos.z])
    R = Rot.from_quat([quat.x, quat.y, quat.z, quat.w]).as_dcm()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, -1] = t

    return T

def odom_msg2T(odom_msg):
    pos = odom_msg.pose.pose.position
    quat = odom_msg.pose.pose.orientation

    return pos_quat2T(pos, quat)

def transorm_msg2T(trans_msg):
    pos = trans_msg.translation
    quat = trans_msg.rotation

    return pos_quat2T(pos, quat)

def parse_odom(odom_bag_path):
    odom_bag = rosbag.Bag(odom_bag_path)
    t0 = 0.0
    odoms = []
    ts = [0]
    prev_T = np.eye(4)
    T_co = compute_odom2cam()
    T_co_inv = invT(T_co)

    for k, (_, msg, t) in enumerate(odom_bag.read_messages(topics=['/zed2/zed_node/odom'])):
        t = t.to_sec()
        if k == 0:
            t0 = t
            # Initialize previous transform to compute relative transforms
            prev_T = odom_msg2T(msg)
            # Define first transform as identity
            odoms.append(np.eye(4))
            continue
        
        T = odom_msg2T(msg)
        rel_T_odom = invT(prev_T).dot(T)
        prev_T = T
        rel_T_cam = T_co.dot(rel_T_odom).dot(T_co_inv)

        # Take previous transform and append relative transform to get new "absolute transform" relative camera frame
        new_T = odoms[k-1].dot(rel_T_cam)

        odoms.append(new_T)
        ts.append(t - t0)

    odom_bag.close()

    return np.array(odoms), np.array(ts), t0

def parse_gt(gt_bag_path):
    gt_bag = rosbag.Bag(gt_bag_path)

    ts = []
    gts = []

    # T_cv = compute_vicon2cam()
    # T_cv_inv = invT(T_cv)

    # for _, msg, t in gt_bag.read_messages(topics=['/vicon/ZED2_ZQ/ZED2_ZQ']):
    #     t = t.to_sec()
    #     T = 
    #     pass

    #     ts.append(t - t0)

    # gt_bag.close()

    # geometry_msgs/TransformStamped
    for k, (_, msg, t) in enumerate(gt_bag.read_messages(topics=['/vicon/ZED2_ZQ/ZED2_ZQ'])):
        t = t.to_sec()
        # if k == 0:
        #     # Initialize previous transform to compute relative transforms
        #     prev_T = transorm_msg2T(msg.transform)
        #     # Define first transform as identity
        #     gts.append(prev_T)
        #     continue
        
        T = transorm_msg2T(msg.transform)
        # rel_T_gt = invT(prev_T).dot(T)
        # prev_T = T
        # rel_T_cam = T_cv.dot(rel_T_gt).dot(T_cv_inv)

        # Take previous transform and append relative transform to get new "absolute transform" relative camera frame
        # new_T = T #gts[k-1].dot(rel_T_cam)

        gts.append(T)
        ts.append(t)

    gt_bag.close()

    return np.array(gts), np.array(ts)

def T2pos_quat(T):
    pos = T[:3, -1]
    quat = Rot.from_dcm(T[:3, :3]).as_quat() # [x, y, z, w]
    return pos, quat

def write_Ts_ts_to_TUM(Ts, ts, filename="poses.txt"):
    tum_lines = []
    for T, t in zip(Ts, ts):
        (x, y, z), (qx, qy, qz, qw) = T2pos_quat(T)
        tum_lines.append([t, x, y, z, qx, qy, qz, qw])

    np.savetxt(filename, tum_lines)

if __name__ == "__main__":
    ycb_item, number, dirname = parse_inputs(sys.argv)

    # odom_bag_path = "{}/../../bags/ziqi/{}_{}.bag".format(dirname, ycb_item, number)
    gt_bag_path = "{}/../../bags/odin/{}{}.bag".format(dirname, ycb_item, number)

    # print("looking at bag {}".format(odom_bag_path))
    # odoms, ts, t0 = parse_odom(odom_bag_path) 
    t0 = 0

    print("looking at bag {}".format(gt_bag_path))
    gts, ts_gt = parse_gt(gt_bag_path)

    # write_Ts_ts_to_TUM(odoms, ts, "odom.txt")
    write_Ts_ts_to_TUM(gts, ts_gt, "cam_gt_raw.txt")