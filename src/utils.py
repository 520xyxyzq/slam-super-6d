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
    Convert GTSAM Pose3 to tum format pose array
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


def pose3DictToTum(data_dict):
    """
    Convert dict of GTSAM poses to (tum format) array
    @param data_dict: [{stamp:gtsam.Pose3}] Stamp-Pose3 dict
    @return data_tum: [Nx8 array] stamp,x,y,z,qx,qy,qz,qw
    """
    assert(len(data_dict) > 0), "Error: Data dictionary empty"
    # Make sure stamps are in sorted order
    stamps = sorted(data_dict.keys())
    # Save as tum format array
    data_tum = np.zeros((len(stamps), 8))
    for ii, stamp in enumerate(stamps):
        data_tum[ii, 0] = stamp
        data_tum[ii, 1:] = gtsamPose32Tum(data_dict[stamp])
    return data_tum


def gtsamValuesToTum(values, stamps):
    """
    Convert GTSAM Values + stamps to (tum format) camera trajectory array
    NOTE: Camera pose variables must follow 'x' + 'int' format
    @param values: [gtsam.Values] GTSAM Values
    @param stamps: [N-list] Time stamps
    @return tum: [Nx8 array] stamp,x,y,z,qx,qy,qz,qw
    """
    # Access all variable keys
    keys = values.keys()
    assert(len(keys) != 0), "Error: No variable in GTSAM Values"
    # Sort keys in terms of symbol index
    keys = sorted(keys, key=lambda x: gtsam.Symbol(x).index())
    # Push camera poses ('x' + 'int') into list
    cam_poses = list()
    for key in keys:
        # check if a key belongs to a camera pose variable
        if gtsam.Symbol(key).chr() == ord('x'):
            cam_poses.append(values.atPose3(key))
    assert(len(cam_poses) != 0), \
        "Error: No camera pose variable ('x'+'int') in GTSAM values!"
    # Sort stamps
    stamps = sorted(stamps)
    assert(len(stamps) != 0), "Error: Stamps list empty!"
    assert(len(stamps) == len(cam_poses)), "Error: #pose != #stamps!"
    # Save stamp-poses to tum format array
    tum = np.zeros((len(stamps), 8))
    for ii, stamp in enumerate(stamps):
        tum[ii, 0] = stamp
        tum[ii, 1:] = gtsamPose32Tum(cam_poses[ii])
    return tum


def readNoiseModel(noise):
    """
    Read noise as GTSAM noise model
    @param noise: [1 or 6 array/list or gtsam.noiseModel] Noise model
    @return noise_model: [gtsam.noiseModel] GTSAM noise model
    """
    # Read prior noise model
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


def printWarn(msg):
    """
    Print warning msg in yellow
    @param msg: [str] msg to print
    """
    print('\033[93m' + msg + '\033[0m')
