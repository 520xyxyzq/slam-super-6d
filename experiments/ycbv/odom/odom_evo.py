#!/usr/bin/env python3
# Evaluate the ORB-SLAM3 estimated odom on YCB videos 
# Author: Ziqi Lu ziqilu@mit.edu
# Copyright 2021 The Ambitious Folks of the MRG

import os
import glob
import scipy.io
from transforms3d.quaternions import quat2mat, mat2quat
import argparse
import numpy as np
from evo.core import trajectory, sync, metrics
from evo.tools import file_interface, plot

# Read command line args
parser = argparse.ArgumentParser()
parser.add_argument("--ycb", type=str, \
                    help = "Directory to YCB-V data folder", \
                    default = "/media/ziqi/LENOVO_USB_HDD/data/YCB-V/data/")
parser.add_argument("--seq", type=str, \
                    help = "YCB sequence id", \
                    default = "0000")
parser.add_argument("--orbslam", type=str, \
                    help = "Directory in which ORB-SLAM results are stored", \
                    default = '/home/ziqi/Desktop/')
parser.add_argument("--out", type=str, \
                    help = "Directory to save the error file", \
                    default = '/home/ziqi/Desktop/test/results/')

args = parser.parse_args()
ycb_folder = args.ycb if args.ycb[-1] == '/' else args.ycb + '/'
ycb_data = ycb_folder + args.seq + '/'
result_folder = args.orbslam if args.orbslam[-1] == '/' else args.orbslam + '/'
result_fname = result_folder + args.seq + '.txt'
# Folder that contains the current py file
file_path = os.path.dirname(os.path.abspath(__file__))

# Get sorted .mat file names in target folder
mat_fnames = sorted(glob.glob(ycb_data + '*-meta.mat'))
# Create the ground truth camera pose file and save it beside this .py file
posef = open(file_path + "/cam_poses.txt", "w")

t_start = 0.0
# NOTE: If change this, also change fps in ycb.yaml and ycb2orbslam.py
fps = 10.0

# Initiate time stamp
stamp = t_start

for mat_fname in mat_fnames:
    mat = scipy.io.loadmat(mat_fname)
    Rt = mat['rotation_translation_matrix']
    Rt = np.vstack((Rt, np.array([0.0, 0.0, 0.0, 1.0])))
    Rt = np.linalg.inv(Rt)
    R, t = Rt[:3,:3], Rt[:3,3]
    # NOTE: mat2quat assumes quat in the order of w x y z
    quat = mat2quat(R)
    posef.write(str("{:.6f}".format(stamp)) + " ")
    posef.write(str(t[0]) + " " + str(t[1]) + " " + str(t[2]) + " ")
    posef.write(str(quat[1]) + " " +\
                str(quat[2]) + " " +\
                str(quat[3]) + " " +\
                str(quat[0]) + "\n")
    # Increment the stamp
    stamp += 1.0/fps

posef.close()

# Read Trajs from file
traj_ref =file_interface.read_tum_trajectory_file(file_path + "/cam_poses.txt")
traj = file_interface.read_tum_trajectory_file(result_fname)
# Synchronizes two trajectories by matching their timestamps.
traj_ref, traj = sync.associate_trajectories(traj_ref, traj)
# Align to reference traj using Umeyama alignment
traj.align(traj_ref, correct_scale = False)
# Compute average traj error
data = (traj_ref, traj)
ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
ape_metric.process_data(data)
ape_statistics = ape_metric.get_all_statistics()
# Fetch statistics
mean = ape_statistics["mean"]
median = ape_statistics["median"]
rmse = ape_statistics["rmse"]
sse = ape_statistics["sse"]
std = ape_statistics["std"]

# Write error statistics into file
error_folder = args.out if args.out[-1] == '/' else args.out + '/'
error_txt = error_folder + "error.txt"
# If no error.txt or it's empty, write header
if not os.path.isfile(error_txt) or os.stat(error_txt).st_size == 0:
    with open(error_txt, "w") as error_file:
        error_file.write("seq mean median rmse sse std\n")    
with open(error_txt, "a") as error_file:
    error_file.write(args.seq + " " + str(mean) + " " + str(median) + " " + \
                   str(rmse) + " " + str(sse) + " " + str(std) + "\n")

# Remove the ground truth file
os.remove(file_path + "/cam_poses.txt")
