#!/bin/bash
# Bash script to run ORBSLAM on YCB videos
# Author: Ziqi Lu ziqilu@mit.edu
# Copyright 2021 The Ambitious Folks of the MRG

# Directory of the test folder
# (where you store the python files, images, results, etc.)
# Make sure the ORBSLAM yaml file is in the folder
test_folder=${1:-~/Desktop/test}
# Where you stored the YCB-V data
ycb_folder=${2:-/media/ziqi/LENOVO_USB_HDD/data/YCB-V/data}
# ORBSLAM3 root folder
orbslam_folder=${3:-~/Packages/ORB_SLAM3}

# CD into the test folder
cd $test_folder

# Remove all the image files (if exist)
if ls -1qA $test_folder/rgb/ | grep -q .
    then rm $test_folder/rgb/*-color.png
fi

if ls -1qA $test_folder/depth/ | grep -q .
    then rm $test_folder/depth/*-depth.png
fi

# Run ORBSLAM on YCB video dataset and save traj and keyframes in results folder
for n in `seq -f "%04g" 0 91`
do
    # Copy imgs into folders
    cp $ycb_folder/"$n"/*-color.png $test_folder/rgb/
    cp $ycb_folder/"$n"/*-depth.png $test_folder/depth/

    # Make the depth, rgb and association files
    python3 $test_folder/ycb2orbslam.py \
    --ycb $ycb_folder --seq "$n" --out "$test_folder"

    # YCB uses different camera models before and after seq 60
    if (($n<60))
        then config_file=$test_folder/YCB.yaml
    else
        config_file=$test_folder/YCB2.yaml
    fi

    # Run ORB-SLAM
    $orbslam_folder/Examples/RGB-D/rgbd_tum \
    $orbslam_folder/Vocabulary/ORBvoc.txt \
    $config_file \
    $test_folder $test_folder/association.txt

    # Rename the camera traj as seq number
    mv KeyFrameTrajectory.txt $test_folder/results/"$n"_keyframe.txt
    mv CameraTrajectory.txt $test_folder/results/$n.txt

    # Remove all the image files
    rm $test_folder/rgb/*-color.png
    rm $test_folder/depth/*-depth.png
done

# Perform error analysis on odom estimates
echo "Performing error analysis..."
for n in `seq -f "%04g" 0 91`
do
    # Run odom error analysis
    # We use world relative object poses to evalute the odom drifts
    python3 $test_folder/odom_evo.py \
    --ycb $ycb_folder --seq "$n" --orbslam $test_folder/results \
    --out $test_folder/results
done
