#!/bin/bash
# Bash script to generate DOPE training data
# Author: Ziqi Lu ziqilu@mit.edu
# Copyright 2021 The Ambitious Folks of the MRG

# Object name used in training
obj=${1:-"010"}
# Where you store the pseudo-labels (txt files)
label_folder=${2:-/home/ziqi/Desktop/test}
# Partial file name used to distinguish pseudo label txt files from other files
file_format=${3:-*obj0.txt}
# Do you want to duplicate hard examples?
duplicate_hard=${4:-false}
# Where you store the ycb-video data folder
ycb_folder=${5:-/media/ziqi/LENOVO_USB_HDD/data/YCB-V/data}
# Where you save the training data
out_folder=${6:-/home/ziqi/Desktop/train}

# Generate DOPE training data
# TODO(ZQ): delete all the files from training folder if not empty?
for det in $label_folder/$file_format
do
    # Get detection txt name from the real path
    det_fname=$(basename $det)
    # Get seq id from file name
    seq=${det_fname:0:4}
    echo "Generating seq $seq"
    # Create the sub directory in the training folder if it doesn't exist
    mkdir -p -- ${out_folder}/${seq}
    # Start data generation
    if [ "$duplicate_hard" = true ] ; then
        python3 $(dirname $(realpath $0))/dope_export.py --obj $obj \
        --seq ${seq} --txt ${det} --out ${out_folder}/${seq} \
        --hard ${det:0:-4}_hard.txt --new
    else
        python3 $(dirname $(realpath $0))/dope_export.py --obj $obj \
        --seq ${seq} --txt ${det} --out ${out_folder}/${seq} --new
    fi
done
