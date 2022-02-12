#!/usr/bin/env python

from cv2 import distanceTransform
from scipy.spatial.transform import Rotation as Rot

from evo.core import trajectory, sync
from evo.tools import plot, file_interface, log

import numpy as np
import os
import sys


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


def get_save_path(dirname, ycb_item, number):
    if ycb_item == "cracker":
        image_folder = "003_cracker_box_16k"
    elif ycb_item == "sugar":
        image_folder = "004_sugar_box_16k"
    elif ycb_item == "spam":
        image_folder = "010_potted_meat_can_16k"

    save_path = "{}/../../{}/00{}".format(dirname, image_folder, number)

    return save_path


def invT(T):
    R = T[:3, :3]
    t = T[:3, -1]

    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, -1] = -R.T.dot(t)

    return Tinv

def pos_quat2T(pos, quat):
    t = np.array([pos.x, pos.y, pos.z])
    R = Rot.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()

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

def T2pos_quat(T):
    pos = T[:3, -1]
    quat = Rot.from_matrix(T[:3, :3]).as_quat() # [x, y, z, w]
    return pos, quat

def write_Ts_ts_to_TUM(Ts, ts, filename="poses.txt"):
    tum_lines = []
    for T, t in zip(Ts, ts):
        (x, y, z), (qx, qy, qz, qw) = T2pos_quat(T)
        tum_lines.append([t, x, y, z, qx, qy, qz, qw])

    np.savetxt(filename, tum_lines)

if __name__ == "__main__":
    ycb_item, number, dirname = parse_inputs(sys.argv)
    save_path = get_save_path(dirname, ycb_item, number)
    odom_data = save_path + "/odom_raw.txt" # SVO odom
    gt_data = save_path + "/cam_gt_raw.txt"

    odom_traj = file_interface.read_tum_trajectory_file(odom_data)
    gt_traj = file_interface.read_tum_trajectory_file(gt_data)

    t_max_diff = 0.01

    # We align Vicon to odom since odom is already in camera frame
    odom_traj, gt_traj = sync.associate_trajectories(
        odom_traj, gt_traj, max_diff=t_max_diff,
        first_name="odom", snd_name="ref")

    t0 = odom_traj.timestamps[0]
    odom_traj.timestamps = odom_traj.timestamps - t0
    gt_traj.timestamps = gt_traj.timestamps - t0
    print(len(odom_traj.timestamps))
    print(odom_traj.timestamps[:5])
    print(gt_traj.timestamps[:5])

    R, t, scale = gt_traj.align(odom_traj, n=-1, correct_scale=True)

    rot = Rot.from_matrix(R).as_euler('XYZ')

    print("Rotation: {}\n translation: {}\n scale: {}".format(rot*180/np.pi, t, scale))

    file_interface.write_tum_trajectory_file()

    # write_Ts_ts_to_TUM(odoms, ts, "odom.txt")
    # write_Ts_ts_to_TUM(gts, ts_gt, "cam_gt.txt")