#!/usr/bin/env bash
# Bash script to run Nvidia DOPE on YCB videos
# Author: Ziqi Lu ziqilu@mit.edu
# Copyright 2021 The Ambitious Folks of the MRG

# Directory of folder that contains the current file
# TODO: make param for the file's folder
# TODO: Automate the process of getting the seq ids
test_folder=${1:-~/Desktop/test}
cd ~/vnav_ws

for n in "20" "25" "29" "33" "36" "37" "43" "49" "51" "54" "55" "60" "74" "77" "85" "89"
do
    roslaunch dope dope.launch &
    PID=$!
    sleep 10s
    rosrun lab_8 ycb_img_pub.py _seq:=$n
    sleep 2s
    kill $PID
done

