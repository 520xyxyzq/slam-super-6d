#!/usr/bin/env python3
# Read ground truth camera trajectory from the ycb-video dataset
# Ziqi Lu ziqilu@mit.edu
# Copyright 2022 The Ambitious Folks of the MRG

import argparse
import glob
import os

import numpy as np
import scipy.io
from transforms3d.quaternions import mat2quat

# Read command line args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--ycb", "-y", type=str,
    help="Directory to YCB-V data folder",
    default="/media/ziqi/LENOVO_USB_HDD/data/YCB-V/data/"
)
parser.add_argument(
    "--out", "-o", type=str,
    help="Directory to save the GT camera trajectory txts",
    default=os.path.dirname(os.path.dirname(os.path.realpath(__file__))) +
    "/odom/ground_truth/"
)
parser.add_argument(
    "--fps", "-f", type=float, help="YCB sequence FPS", default=10.0
)
parser.add_argument(
    "--seq", "-s", type=int, nargs="+", default=0,
    help="YCB sequence ID (0~91)"
)
args = parser.parse_args()

ycb_folder = args.ycb if args.ycb[-1] == "/" else args.ycb + "/"
ycb_seqs = [ycb_folder + str(seq).rjust(4, "0") + "/" for seq in args.seq]
# Save files here
target_folder = args.out if args.out[-1] == "/" else args.out + "/"
if args.fps != 10.0:
    print("Warn: If change FPS, need to change it in ycb.yaml for comparison")
fps = args.fps

for ii, seq in enumerate(ycb_seqs):
    # Initialize time stamp
    stamp = 0.0
    # Get sorted .mat file names in ycb folder
    mat_fnames = sorted(glob.glob(seq + '*-meta.mat'))
    # Initialize cam pose array
    poses = np.zeros((len(mat_fnames), 8))

    for tt, mat_fname in enumerate(mat_fnames):
        mat = scipy.io.loadmat(mat_fname)
        Rt = mat['rotation_translation_matrix']
        Rt = np.vstack((Rt, np.array([0.0, 0.0, 0.0, 1.0])))
        Rt = np.linalg.inv(Rt)
        R, t = Rt[:3, :3], Rt[:3, 3]
        poses[tt, 0] = stamp
        poses[tt, 1:4] = t
        # mat2quat quat order w x y z
        quat = mat2quat(R)
        poses[tt, 4:7] = quat[1:4]
        poses[tt, 7] = quat[0]
        # Increment the stamp
        stamp += 1.0 / fps

    # Save to file
    np.savetxt(
        target_folder + str(args.seq[ii]).rjust(4, "0") + ".txt", poses,
        fmt=["%.1f"] + ["%.12f"] * 7
    )
