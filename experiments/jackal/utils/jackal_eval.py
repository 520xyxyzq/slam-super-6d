#!/usr/bin/env python3
# Evaluate pose estimations in the jackal experiments
# Ziqi Lu ziqilu@mit.edu

import argparse
import os

import gtsam
import numpy as np
from transforms3d.quaternions import qisunit, quat2mat


def tum2GtsamPose3(tum_pose):
    """
    Convert tum format pose to GTSAM Pose3
    @param tum_pose: [7-array] x,y,z,qx,qy,qz,qw
    @return pose3: [gtsam.pose3] GTSAM Pose3
    """
    assert(len(tum_pose) == 7), "Error: Tum format pose must have 7 entrices"
    tum_pose = np.array(tum_pose)
    # gtsam quaternion order wxyz
    qx, qy, qz = tum_pose[3:-1]
    qw = tum_pose[-1]
    pose3 = gtsam.Pose3(
        gtsam.Rot3.Quaternion(qw, qx, qy, qz),
        gtsam.Point3(tum_pose[:3])
    )
    return pose3


def gtsamPose32Tum(pose3):
    """
    Convert tum format pose to GTSAM Pose3
    @param pose3: [gtsam.pose3] GTSAM Pose3
    @return tum_pose: [7-array] x,y,z,qx,qy,qz,qw
    """
    tum_pose = np.ones((7,))
    tum_pose[0:3] = pose3.translation()
    quat = pose3.rotation().quaternion()
    # From gtsam wxyz to tum xyzw
    tum_pose[3:6] = quat[1:]
    tum_pose[6] = quat[0]
    return tum_pose


def readTum(txt):
    """
    Read poses from txt file (tum format) into dict of GTSAM poses
    @param txt: [string] Path to the txt file containing poses
    @return poses: [dict] {stamp: gtsam.Pose3}
    """
    # TODO: assert tum format here
    assert(os.path.isfile(txt)), "Error: %s not a file" % (txt)
    rel_poses = np.loadtxt(txt)
    # Read poses into dict of GTSAM poses
    poses = {}
    for ii in range(rel_poses.shape[0]):
        # Skip lines with invalid quaternions
        if (qisunit(rel_poses[ii, 4:])):
            poses[rel_poses[ii, 0]] = tum2GtsamPose3(rel_poses[ii, 1:])
    return poses


class JackalEval:

    def __init__(self, dets, gt, dim, intrinsics):
        """
        Load recomputed and ground truth object poses
        @param dets: [list of strs] Detected object pose files
        @param gt: [string] Path to the ground truth object pose detections
        @param dim: [3 array/list] Dimension (x,y,z) of the object
        @param intrinsics: [5-array/list] Camera intrinsics
        """
        # Read re-computed object pose detections
        self._dets_ = []
        for ii, det in enumerate(dets):
            assert(os.path.isfile(det)), "Error: %s is not a file" % dets
            self._dets_.append(readTum(det))
            assert(len(self._dets_[ii]) > 0), \
                "Error: recomputed detections empty!"

        assert(os.path.isfile(gt)), "Error: %s is not a file" % gt
        # Read ground truth object pose detections
        self._gt_ = readTum(gt)
        assert(len(self._gt_) > 0), "Error: Ground truth poses empty!"

        self._dim_ = dim
        self._intrinsics_ = intrinsics

    def curve(self, thresh=30):
        """
        Make accuracy-threshold curves
        @param thresh: [float] Cutoff value for the plot
        @return curve: [NxM array] the curves
        """
        # Range of the threshold axes
        xrange = np.arange(0, thresh + 0.1, 0.1)
        curve = np.zeros((len(xrange), 1 + len(self._dets_)))
        curve[:, 0] = xrange
        print("AUCs: ")
        for ii, det in enumerate(self._dets_):
            errors = self.error(ii)
            height = []
            for e in xrange:
                height.append(np.sum(errors <= e))
            curve[:, ii+1] = height
            curve[:, ii+1] /= len(self._gt_)
            auc = np.trapz(curve[:, ii+1], dx=0.1)
            print(auc)
        return curve

    def error(self, ind):
        """
        Compute errors in pseudo labels
        @param ind: [int] Index of det file to compute errors
        @return error: [list] List of pseudo label errors [pixels]
        """
        assert(ind < len(self._dets_)), \
            "Error: index %d out of range %d" % (ind, len(self._dets_))
        error = []
        for stamp, pose in self._gt_.items():
            # Skip if no recomputed obj pose at this stamp
            if stamp not in self._dets_[ind]:
                continue
            # Convert pose to tum format
            pose_gt = gtsamPose32Tum(pose)
            pose_det = gtsamPose32Tum(self._dets_[ind][stamp])
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
            avg_error = np.mean(errors)
            error.append(avg_error)
        return error

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
    # Read command line args
    parser = argparse.ArgumentParser()
    jackal_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument(
        "--gt", "-g", type=str, help="Ground truth object poses (.txt)",
        default="/home/ziqi/Desktop/test/cracker_4_0001.txt"
    )
    parser.add_argument(
        "--det", "-d", nargs="+", type=str, help="detection files (.txt)",
        default=[
            jackal_dir + "/003_cracker_box_16k/004/infer_before.txt",
            jackal_dir + "/003_cracker_box_16k/004/infer_after.txt"
        ]
    )
    parser.add_argument(
        "--dim", "-dim", type=float, nargs=3, help="Object dimension [x y z]",
        default=[16.4036, 21.3437, 7.1800]
    )
    parser.add_argument(
        "--intrinsics", "-in", type=float, nargs=5,
        help="Camera intrinsics: fx, fy, cx, cy, s",
        default=[276.3225, 276.3225, 353.70087, 179.0885, 0]
    )
    args = parser.parse_args()

    jackal_eval = JackalEval(args.det, args.gt, args.dim, args.intrinsics)
    curve = jackal_eval.curve()
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': "Times New Roman", 'font.size': "20"})
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(curve[:, 0], curve[:, 1], "k", linewidth=3)
    plt.plot(curve[:, 0], curve[:, 2], "r", linewidth=3)
    plt.legend(["Before", "After"], fontsize=20)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Reprojection error (pixel)")
    plt.xlim([0, 30])
    plt.ylim([0, 1])
    ax.set_aspect(25)
    plt.tight_layout()
    plt.title("003_cracker_box")
    plt.show()
