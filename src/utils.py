#!/usr/bin/env python3
# Utility functions
# Ziqi Lu, ziqilu@mit.edu
# Copyright 2022 The Ambitious Folks of the MRG

import os

import gtsam
import numpy as np
from transforms3d.quaternions import qisunit


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


def readNoiseModel(noise):
    """
    Read noise as GTSAM noise model
    @param noise: [1 or 6 array/list or gtsam.noiseModel] Noise model
    @return noise_model: [gtsam.noiseModel] GTSAM noise model
    """
    # Read prior noise model
    # TODO(ZQ): maybe allow for non-diagonal terms in noise model?
    if type(noise) in [gtsam.noiseModel.Isotropic, gtsam.noiseModel.Diagonal,
                       gtsam.noiseModel.Unit, gtsam.noiseModel.Gaussian]:
        return noise
    if len(noise) == 1:
        noise_model = \
            gtsam.noiseModel.Isotropic.Sigma(6, noise[0])
    elif len(noise) == 6:
        noise_model = \
            gtsam.noiseModel.Diagonal.Sigmas(np.array(noise))
    else:
        assert(False), "Error: Unexpected noise model type!"
    return noise_model
