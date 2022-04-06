#!/usr/bin/env python3
# Evaluate pseudo labels (keypoints) wrt the ground truth
# Ziqi Lu, ziqilu@mit.edu
# Copyright 2022 The Ambitious Folks of the MRG

import argparse
import os

import numpy as np
from transforms3d.quaternions import quat2mat
from utils import gtsamPose32Tum, readTum


class LabelEval:
    def __init__(self, dets, gt, dim, intrinsics):
        """
        Load recomputed and ground truth object poses
        @param dets: [string or {stamp:gtsam.Pose3}] Recomputed object poses
        @param gt: [string] Path to the ground truth object pose detections
        @param dim: [3 array/list] Dimension (x,y,z) of the object
        @param intrinsics: [5-array/list] Camera intrinsics
        """
        # Read re-computed object pose detections
        if type(dets) == str:
            self._dets_ = readTum(dets)
        elif type(dets) == dict:
            self._dets_ = dets
        else:
            assert(False), \
                "Error: Unsupported detection format! Must be dict or str."
        assert(len(self._dets_) > 0), "Error: recomputed detections empty!"
        # Read ground truth object pose detections
        self._gt_ = readTum(gt)

        self._dim_ = dim
        self._intrinsics_ = intrinsics

    def error(self):
        """
        Compute errors (statistics) in pseudo labels
        @return mean: [float] Mean error [pixels]
        @return median: [float] Median error [pixels]
        @return std: [float] Standard deviation for errors [pixels]
        """
        error = []
        for stamp, pose in self._gt_.items():
            # Skip if no recomputed obj pose at this stamp
            if stamp not in self._dets_:
                continue
            # Convert pose to tum format
            pose_gt = gtsamPose32Tum(pose)
            pose_det = gtsamPose32Tum(self._dets_[stamp])
            # Make and project cuboid to image
            cuboid_gt = self.add_cuboid(pose_gt[:3], pose_gt[3:], self._dim_)
            cuboid_gt_proj = self.project_cuboid(cuboid_gt, self._intrinsics_)
            cuboid_det = self.add_cuboid(
                pose_det[:3], pose_det[3:], self._dim_
            )
            cuboid_det_proj = self.project_cuboid(
                cuboid_det, self._intrinsics_
            )
            # Compute keypoint location errors in pixels
            errors = np.linalg.norm(cuboid_det_proj - cuboid_gt_proj, axis=1)
            error += errors.tolist()
        return np.mean(error), np.median(error), np.std(error)

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
            [cam_quat[-1], cam_quat[0], cam_quat[1], cam_quat[2]]
        )
        cam_pose[:3, :3] = cam_rot
        # Extrinsic matrix is inverse of camera world pose
        Rt = (np.linalg.inv(cam_pose))[:3, :]
        # Project
        cuboid_homo = np.hstack((cuboid, np.ones((cuboid.shape[0], 1))))
        cuboid_proj_homo = K.dot(Rt.dot(cuboid_homo.T))
        cuboid_proj = (cuboid_proj_homo[:2, :] / cuboid_proj_homo[2, :]).T
        return cuboid_proj


if __name__ == "__main__":
    # Package root directory
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # Read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dim", "-dim", type=float, nargs=3, help="Object dimension [x y z]",
        default=[16.4036, 21.3437, 7.1800]
    )
    parser.add_argument(
        "--det", "-d", type=str, help="Recomputed object poses",
        default=root + "/experiments/ycbv/pseudo_labels/003_cracker_box_16k" +
        "/Inlier_1/0007_obj0.txt"
    )
    parser.add_argument(
        "--gt", "-g", type=str, help="Ground truth object (relative) poses",
        default=root + "/experiments/ycbv/inference/" +
        "/003_cracker_box_16k/ground_truth/0007_ycb_gt.txt"
    )
    parser.add_argument(
        "--intrinsics", "-in", type=float, nargs=5,
        help="Camera intrinsics: fx, fy, cx, cy, s",
        default=[1066.778, 1067.487, 312.9869, 241.3109, 0]
    )
    args = parser.parse_args()

    label_eval = LabelEval(args.det, args.gt, args.dim, args.intrinsics)
    print(label_eval.error())
