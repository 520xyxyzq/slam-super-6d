#!/usr/bin/env python3
# Read ground truth object pose measurements from YCB-V dataset
# Ziqi Lu ziqilu@mit.edu

import argparse
import glob
import json
import os

import numpy as np
import scipy.io
from transforms3d.quaternions import mat2quat


def main(obj, ycb, seqs, out, ycb_json, fps=10.0):
    '''
    Extract ground truth data for object pose detections from YCB-V dataset
    @param obj: [str] object name
    @param ycb: [str] path to ycb data folder
    @param seqs: [list] ycb seqs (names) to read ground truth from
    @param out: [str] target folder to save the training data
    @param ycb_json: [str] json file containing all ycb objects' data
    '''
    # Open _ycb_original.json to load models' static transformations
    with open(ycb_json) as yj:
        transforms = json.load(yj)
    # Names of all YCB objects
    class_names = transforms["exported_object_classes"]
    # Find index of the current object
    obj_id = class_names.index(obj)
    # Load fixed transform for the obj (YCB to DOPE defined coordinate system)
    # NOTE: this is the transform of the original frame wrt the new frame
    obj_transf = \
        transforms["exported_objects"][obj_id]["fixed_model_transform"]
    obj_transf = np.array(obj_transf).T
    obj_transf[:3, :] /= 100
    # NOTE: we are using non-translation transformation
    obj_transf[:3, 3] = [0, 0, 0]
    print("Fetching %s ground truth pose measurements" % obj)

    # Check whether the out folder exists, if not, create it.
    if not os.path.exists(out):
        os.makedirs(out)
    for ii in seqs:
        # reformat the sequence name string
        seq_id = ii.rjust(4, "0")
        # Create the camera pose file
        gtf = open(out + seq_id + "_ycb_gt.txt", "w")
        print("seq " + seq_id)
        ycb_data = ycb + seq_id + '/'
        # Get sorted .mat file names in target folder
        mat_fnames = sorted(glob.glob(ycb_data + '*-meta.mat'))
        # All the obj indices in this seq
        indices = scipy.io.loadmat(mat_fnames[0])['cls_indexes'].ravel()
        # Extract the index of the object (1-indexed)
        ind = np.where(indices == obj_id + 1)[0][0]
        for jj in range(len(mat_fnames)):
            # Read data
            mat = scipy.io.loadmat(mat_fnames[jj])
            data = mat['poses']
            Rt = data[:, :, ind]
            Rt = np.vstack((Rt, np.array([0.0, 0.0, 0.0, 1.0])))
            Rt = Rt.dot(np.linalg.inv(obj_transf))
            R, t = Rt[:3, :3], Rt[:3, 3]
            quat = mat2quat(R)
            gtf.write(str("{:.6f}".format(jj / fps)) + " ")
            gtf.write(str(t[0]) + " " + str(t[1]) + " " + str(t[2]) + " ")
            gtf.write(str(quat[1]) + " " +
                      str(quat[2]) + " " +
                      str(quat[3]) + " " +
                      str(quat[0]) + "\n")
        gtf.close()


def seqsHavingObj(obj, ycb_json):
    '''
    Read ids of sequences that have "obj" in it
    @param obj: [string] object name
    @param ycb_json: [string] Path to the _ycb_original.json file
    @return seqs: [list] List of seqs (as strings) having obj
    '''
    with open(ycb_json) as yj:
        transforms = json.load(yj)
    class_names = transforms["exported_object_classes"]
    # Find index of the current object
    obj_id = class_names.index(obj)
    return transforms["exported_objects"][obj_id]["seqs"]


if __name__ == "__main__":
    # Read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj", type=str, help="Object name", default="004_sugar_box_16k"
    )
    parser.add_argument(
        "--ycb",
        type=str,
        help="Directory to YCB-V data folder",
        default="/media/ziqi/LENOVO_USB_HDD/data/YCB-V/data/",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Directory to save the GT detection txts, no need for obj name!",
        default=os.path.dirname(os.path.dirname(os.path.realpath(__file__))) +
        "/dets/ground_truth/"
    )
    # NOTE: There should be no need to modify the following params
    parser.add_argument(
        "--ycb_json",
        type=str,
        help="Path to the _ycb_original.json file",
        default=os.path.dirname(os.path.dirname(os.path.realpath(__file__))) +
        "/_ycb_original.json"
    )
    parser.add_argument(
        "--fps", type=float, help="Sequence FPS", default=10.0
    )
    parser.add_argument("--seqs", type=str, nargs="+",
                        help="Seqs to generate GT from, make sure have obj",
                        default=None)
    args = parser.parse_args()

    target_folder = args.out if args.out[-1] == "/" else args.out + "/"
    target_folder += args.obj + "/"
    ycb_folder = args.ycb if args.ycb[-1] == "/" else args.ycb + "/"
    # If seqs not specified, read all the seqs in which obj shows up
    if args.seqs is None:
        seqs = seqsHavingObj(args.obj, args.ycb_json)
    else:
        seqs = args.seqs

    main(args.obj, ycb_folder, seqs,
         target_folder, args.ycb_json, args.fps)
