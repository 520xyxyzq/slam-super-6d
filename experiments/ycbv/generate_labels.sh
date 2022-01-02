#!/bin/bash
# Generate pseudo labels from cam odom and obj pose detections for NVIDIA DOPE
# Author: Ziqi Lu ziqilu@mit.edu
# Copyright 2022 The Ambitious Folks of the MRG

# Usage: generate_labels $1 object_name $2 out_folder
obj=${1:-004_sugar_box_16k}
# Where you save the pseudo labels
out_folder=${2:-~/Desktop/test}

# TODO(ZQ): check whether out_folder is empty before execution
# TODO(zq): fix path endings i.e. "/"

# Root folder of the package (slam-super-6d)
root=$(dirname $(dirname $(dirname $(realpath $0))))
# Python script
pseudo_labeler=$root/src/pseudo_labeler.py
# Where you store the DOPE detection txt files
det_folder=$(dirname $(realpath $0))/dets/results/$obj
# Where you store the odometry txt files
odom_folder=$(dirname $(realpath $0))/odom/results
# Where you store the ground truth camera trajectory txt files
gt_cam_folder=$(dirname $(realpath $0))/odom/ground_truth
# Where you store the ground truth object poses txt files
gt_det_folder=$(dirname $(realpath $0))/dets/ground_truth/$obj

for det in $det_folder/*.txt
do
    # Get detection txt name from the real path
    det_fname=$(basename $det)
    # Get seq id from file name
    seq=${det_fname:0:4}
    # Get odometry txt real path
    odom=$odom_folder/$seq.txt
    # Ground truth object poses
    gt_obj=$gt_det_folder/${seq}_ycb_gt.txt
    # Ground truth camera trajectory
    gt_cam=$gt_cam_folder/$seq.txt
    # Optimize!
    python3 $pseudo_labeler --odom $odom --dets $det --out $out_folder \
    --gt_cam $gt_cam --gt_obj $gt_obj --kernel 2 --optim 1 -dn 0.08 \
    --save --verbose
done
