#!/usr/bin/env python3
# Convert a ycb sequence folder to a TUM-format folder that is processable by ORB-SLAM3
# NOTE: Pre-Configure the folder
# Create a "target_folder/rgb" directory
# And run "cp ycb_folder/*-color.png target_folder/rgb" in terminal
# Ziqi Lu ziqilu@mit.edu

import glob
import argparse

# Read command line args
parser = argparse.ArgumentParser()
parser.add_argument("--ycb", type=str, \
                    help = "Directory to YCB-V data folder", \
                    default = "/media/ziqi/LENOVO_USB_HDD/data/YCB-V/data/")
parser.add_argument("--seq", type=str, \
                    help = "YCB sequence id", \
                    default = "0000")
parser.add_argument("--out", type=str, \
                    help = "Directory to save the imgs and files", \
                    default = '/home/ziqi/Desktop/test')

args = parser.parse_args()
ycb_folder = args.ycb if args.ycb[-1] == '/' else args.ycb + '/'
ycb_folder = ycb_folder + args.seq + '/'
target_folder = args.out if args.out[-1] == '/' else args.out + '/'


t_start = 0.0
fps = 10.0
# Create the stamp-filename files
rgb_file = open(target_folder + "rgb.txt", "w")
depth_file = open(target_folder + "depth.txt", "w")
ass_file = open(target_folder + "association.txt", "w")

# Write file headers
rgb_file.write("# YCB sequence " + ycb_folder[-5:-1] + "\n")
rgb_file.write("# timestamp filename\n")
depth_file.write("# YCB sequence " + ycb_folder[-5:-1] + "\n")
depth_file.write("# timestamp filename\n")

# Get sorted image file names in target folder
img_fnames = sorted(glob.glob(target_folder + 'rgb/*-color.png'))
depth_fnames = sorted(glob.glob(target_folder + 'depth/*-depth.png'))
assert(len(img_fnames) == len(depth_fnames))
# Initiate time stamp
stamp = t_start

for ii in range(len(img_fnames)):
    # Write time stamp & filename
    # Get local file name by slicing the string
    rgb_file.write(str("{:.6f}".format(stamp)) + " rgb/" + 
		 img_fnames[ii][-16:] +"\n")
    depth_file.write(str("{:.6f}".format(stamp)) + " depth/" + 
		 depth_fnames[ii][-16:] + "\n")
    ass_file.write(str("{:.6f}".format(stamp)) + " rgb/" + 
		 img_fnames[ii][-16:] +" ")
    ass_file.write(str("{:.6f}".format(stamp)) + " depth/" + 
		 depth_fnames[ii][-16:] + "\n")
    # Increment the stamp
    stamp += 1.0/fps

rgb_file.close()
depth_file.close()
ass_file.close()
