#!/usr/bin/env python3
# Evaluate pseudo labels (keypoints) wrt the ground truth
# Ziqi Lu, ziqilu@mit.edu
# Copyright 2022 The Ambitious Folks of the MRG

import argparse

import numpy as np
from transforms3d.quaternions import quat2mat


class LabelEval:
    def __init__(self, dim):
        """
        @param dim: [3 array/list] Dimension (x,y,z) of the object
        """
        pass

    def add_cuboid(self, trans, quat, dim):
        """
        Compute cuboid of an object from its pose and dimensions
        @param trans: [3 array] Object center location wrt camera frame [m]
        @param quat: [4 array] Obj quaternion_xyzw wrt camera frame
        (obj coord: x left y down z out)
        @param dim: [3 array] Dimension (x,y,z) of the object
        @return cuboid: [8x3 array] object cuboid vertex coords wrt cam [m]
        """
        # In case they are not np arrays
        # And change location unit to [cm]
        trans, quat, dim = np.array(trans) * 100, np.array(quat), np.array(dim)
        # Vertex order for the NEW training script (centroid included)
        vert = np.array(
            [
                [1, -1, 1],
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, 1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, -1],
                [0, 0, 0]
            ]
        )
        # Vector from center to one vertex (id 3)
        vector = dim.reshape(1, 3) / 2
        # Rotation matrix from quaternion (quat2mat follows qw qx qy qz order)
        rot_mat = quat2mat([quat[-1], quat[0], quat[1], quat[2]])
        # Transform vertex coords to world frame
        cuboid = rot_mat.dot((vert * vector).T) + trans.reshape(3, 1)
        return (cuboid / 100.0).T

    def project_cuboid(self, cuboid, intrinsics,
                       cam_trans=[0, 0, 0], cam_quat=[0, 0, 0, 1]):
        """
        Project cuboid (vertices) onto image
        @param cuboid: [Nx3 array] Object cuboid vertex coordinates wrt cam [m]
        @param intrinsics: [5 array] [fx, fy, cx, cy, s]
        @param cam_trans: [3 array] Camera translation (wrt world frame)
        (This should always be [0, 0, 0] for DOPE)
        @param cam_quat: [4 array] Camera quaternion (wrt world frame)
        (This should always be [0, 0, 0, 1] for DOPE)
        @return cuboid_proj: [Nx2 array] projected cuboid (pixel) coordinates
        """
        # In case they are not np arrays
        cam_trans, cam_quat = np.array(cam_trans), np.array(cam_quat)
        # Assemble intrinsic matrix
        fx, fy, cx, cy, s = intrinsics
        K = np.eye(3)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1] = fx, fy, cx, cy, s
        # Assemble extrinsic matrix
        cam_pose = np.eye(4)
        cam_pose[:3, 3:] = cam_trans.reshape(3, 1)
        cam_rot = quat2mat(
            [cam_quat[-1], cam_quat[0], cam_quat[1], cam_quat[2]])
        cam_pose[:3, :3] = cam_rot
        # Extrinsic matrix is inverse of camera world pose
        Rt = (np.linalg.inv(cam_pose))[:3, :]
        # Project
        cuboid_homo = np.hstack((cuboid, np.ones((cuboid.shape[0], 1))))
        cuboid_proj_homo = K.dot(Rt.dot(cuboid_homo.T))
        cuboid_proj = (cuboid_proj_homo[:2, :] / cuboid_proj_homo[2, :]).T
        return cuboid_proj


if __name__ == "__main__":
    # Read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dim", "-d", type=float, nargs=3, help="Object dimension (x y z)",
        default=[10.1647, 8.3543, 5.7601]
    )
    parser.add_argument(
        "--gt", type=str,
        help="Ground truth object (relative) poses",
        default="/home/ziqi/Desktop/0000.txt"
    )
    parser.add_argument(
        "--intrinsics", "-in", type=float, nargs=5,
        help="Camera intrinsics: fx, fy, cx, cy, s",
        default=[1066.778, 1067.487, 312.9869, 241.3109, 0]
    )
    args = parser.parse_args()

    label_eval = LabelEval(args.dim)
