#!/bin/bash
# Generate pseudo labels from cam odom and obj pose detections for NVIDIA DOPE
# Author: Ziqi Lu ziqilu@mit.edu
# Copyright 2022 The Ambitious Folks of the MRG

# Usage: generate_labels $1 object_name $2 out_folder
obj=${1:-010_potted_meat_can_16k}
# Which estimator you used to infer object poses
# i.e. folder names under root/experiments/ycbv/inference/results/
training_data=${2:-Initial}
# Labeling mode: SLAM(0); Inlier(1); Hybrid(2); PoseEval(3)
labeling_mode=${3:-2}
# Where you save the pseudo labels
out_folder=${4:-~/Desktop/test}
# Where you store the YCB-V data folder
ycb_folder=${5:-/media/ziqi/LENOVO_USB_HDD/data/YCB-V/data}

# TODO(ZQ): check whether out_folder is empty before execution
# TODO(zq): fix path endings i.e. "/"

# Root folder of the package (slam-super-6d)
root=$(dirname $(dirname $(dirname $(realpath $0))))
# Python script
pseudo_labeler=$root/src/pseudo_labeler.py
# Where you store the DOPE predictions txt files
det_folder=$(dirname $(realpath $0))/inference/$obj/$training_data
# Where you store the odometry txt files
odom_folder=$(dirname $(realpath $0))/odom/results
# Where you store the ground truth camera trajectory txt files
gt_cam_folder=$(dirname $(realpath $0))/odom/ground_truth
# Where you store the ground truth object poses txt files
gt_det_folder=$(dirname $(realpath $0))/inference/$obj/ground_truth

for det in $det_folder/*.txt
do
    # Get pose predictions txt name from the real path
    det_fname=$(basename $det)
    # Get seq id from file name
    seq=${det_fname:0:4}
    echo "Generating seq $seq"
    # Get odometry txt real path
    odom=$odom_folder/$seq.txt
    # Ground truth object poses
    gt_obj=$gt_det_folder/${seq}_ycb_gt.txt
    # Ground truth camera trajectory
    gt_cam=$gt_cam_folder/$seq.txt
    # Optimize!
    python3 $pseudo_labeler --odom $odom --dets $det --out $out_folder \
    --obj ${obj:0:-4} --imgs "$ycb_folder/$seq/*-color.png" \
    --gt_cam $gt_cam --gt_obj $gt_obj --kernel 0 --optim 1 -j -l 10 \
    --mode $labeling_mode --save
done

# Concatenate all the error files and rm them
cat $out_folder/*_error.txt > $out_folder/error.txt
rm $out_folder/*_error.txt
