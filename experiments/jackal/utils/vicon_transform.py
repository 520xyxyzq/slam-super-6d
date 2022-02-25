#!/usr/bin/env python3
# Convert vicon poses to ground truth camera pose
# Ziqi Lu ziqilu@mit.edu
# TODO: This file needs more tests

import argparse
import os

import gtsam
import numpy as np


def main(src, dst, trans, quat):
    """
    Convert vicon to camera ground truth poses
    @param src: [str] Source txt with vicon poses
    @param dst: [str] Destination txt file name for camera ground truth poses
    @param trans: [3-list] Translation part for camToVicon pose
    @param quat: [4-list] Quaternion (x, y, z, w) part for camToVicon pose
    """
    quat = np.hstack((quat[-1], quat[:3]))
    cam2Vicon = gtsam.Pose3(gtsam.Rot3.Quaternion(*quat), gtsam.Point3(*trans))

    assert(os.path.isfile(src)), "Error: %s not a file" % src

    vicon_poses = np.loadtxt(src)
    gt = np.zeros_like(vicon_poses)
    for ii in range(vicon_poses.shape[0]):
        # copy paste stamps
        gt[ii, 0] = vicon_poses[ii, 0]
        quat_v = vicon_poses[ii, 4:]
        quat_v = np.hstack((quat_v[-1], quat_v[:3]))
        trans_v = vicon_poses[ii, 1:4]
        pose_v = gtsam.Pose3(
            gtsam.Rot3.Quaternion(*quat_v),
            gtsam.Point3(*trans_v)
        )
        if ii == 0:
            # Initial camera pose is identity
            gt[ii, -1] = 1
            prev_pose = pose_v
            accum_pose = gtsam.Pose3()
            continue
        # Compute relative transform
        delta_transform = prev_pose.inverse().compose(pose_v)
        delta_transform = cam2Vicon * delta_transform * cam2Vicon.inverse()
        # Accumulate camera poses
        accum_pose = accum_pose.compose(delta_transform)

        quat_new = accum_pose.rotation().toQuaternion().coeffs()
        trans_new = accum_pose.translation()
        gt[ii, 1:4] = trans_new
        gt[ii, 4:] = quat_new
        prev_pose = pose_v

    np.savetxt(dst, gt, fmt="%.18f")


if __name__ == "__main__":
    # Read command line args
    parser = argparse.ArgumentParser()
    jackal_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument(
        "--src", "-s", type=str, help="source txt",
        default=jackal_dir + "/010_potted_meat_can_16k/004/cam_gt_raw.txt"
    )
    parser.add_argument(
        "--dst", "-d", type=str, help="destination txt",
        default="/home/ziqi/Desktop/cam_gt.txt"
    )
    parser.add_argument(
        "--trans", "-t", type=float, nargs=3,
        help="Translation part for camToVicon pose",
        default=[0.06307904, -0.01538599, -0.00929433]
    )
    parser.add_argument(
        "--quat", "-q", type=float, nargs=4,
        help="Quaternion (x, y, z, w) part for camToVicon pose",
        default=[0.21452065,  0.92716258, -0.28143703,  0.12305947]
    )
    args = parser.parse_args()

    main(args.src, args.dst, args.trans, args.quat)
