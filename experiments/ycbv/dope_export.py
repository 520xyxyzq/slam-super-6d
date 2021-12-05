#!/usr/bin/env python3
# Generate and save DOPE training data from computed detections
# Ziqi Lu ziqilu@mit.edu

import argparse
import glob
import json
import shutil

import numpy as np
from transforms3d.quaternions import qisunit, quat2mat


def add_cuboid(trans, quat, dim):
    """
    Compute cuboid of an object from its pose and dimensions
    @param trans: [3 array] Object center location wrt camera frame [m]
    @param quat: [4 array] Obj quaternion_xyzw wrt camera frame
    (obj coord: x left y down z out)
    @param dim: [3 array] Dimension (x,y,z) of the object
    @return cuboid: [8x3 array] object cuboid vertex coordinates wrt cam [m]
    """
    # In case they are not np arrays
    # And change location unit to [cm]
    trans, quat, dim = np.array(trans * 100), np.array(quat), np.array(dim)
    # Helper vector to get vector from obj center to vertices
    # Vertex order here:
    # research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/
    vert = np.array(
        [
            [1, -1, 1],
            [-1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
            [1, -1, -1],
            [-1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
        ]
    )
    # Vector from center to one vertex (id 3)
    vector = dim.reshape(1, 3) / 2
    # Rotation matrix from quaternion (quat2mat follows qw qx qy qz order)
    rot_mat = quat2mat([quat[-1], quat[0], quat[1], quat[2]])
    # Transform vertex coords to world frame
    cuboid = rot_mat.dot((vert * vector).T) + trans.reshape(3, 1)
    return (cuboid / 100.0).T


def project_cuboid(cuboid, intrinsics,
                   cam_trans=[0, 0, 0], cam_quat=[0, 0, 0, 1]):
    """
    Project cuboid (vertices) onto image
    @param cuboid: [Nx3 array] Object cuboid vertex coordinates wrt cam [m]
    @param intrinsics: [5 array] [fx, fy, cx, cy, s]
    @param cam_trans: [3 array] Camera translation (wrt world frame)
    (This should always be [0, 0, 0] for DOPE)
    @param cam_quat: [4 array] Camera quaternion (wrt world frame)
    (This should always be [0, 0, 0, 1] for DOPE)
    @return cuboid_proj: [Nx2 array] projected cuboid (pixel) coordinates
    """
    # In case they are not np arrays
    cam_trans, cam_quat = np.array(cam_trans), np.array(cam_quat)
    # Assemble intrinsic matrix
    fx, fy, cx, cy, s = intrinsics
    K = np.eye(3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1] = fx, fy, cx, cy, s
    # Assemble extrinsic matrix
    cam_pose = np.eye(4)
    cam_pose[:3, 3:] = cam_trans.reshape(3, 1)
    cam_rot = quat2mat([cam_quat[-1], cam_quat[0], cam_quat[1], cam_quat[2]])
    cam_pose[:3, :3] = cam_rot
    # Extrinsic matrix is inverse of camera world pose
    Rt = (np.linalg.inv(cam_pose))[:3, :]
    # Project
    cuboid_homo = np.hstack((cuboid, np.ones((cuboid.shape[0], 1))))
    cuboid_proj_homo = K.dot(Rt.dot(cuboid_homo.T))
    cuboid_proj = (cuboid_proj_homo[:2, :] / cuboid_proj_homo[2, :]).T
    return cuboid_proj


def read_poses(txt):
    """
    Read txt file w/ relative poses (tum format) into np array
    @param txt: [string] Path to the txt file containing relative obj poses
    @return stamp: [N array] stamp
    @return rel_trans: [Nx3 array] x,y,z
    @return rel_quat: [Nx4 array] qx,qy,qz,qw
    """
    rel_poses = np.loadtxt(txt)
    # Skip lines with invalid quaternions
    ind = []
    for ii in range(rel_poses.shape[0]):
        if (qisunit(rel_poses[ii, 4:])):
            ind.append(ii)
    return rel_poses[ind, 0], rel_poses[ind, 1:4], rel_poses[ind, 4:]


def copy_img(src, dest):
    """
    Copy img from one path to another
    @param src: Source file name
    @param dest: Destination file name
    """
    shutil.copy2(src, dest)


def data2dict(obj, trans, quat, centroid, centroid_proj, cuboid, cuboid_proj):
    """
    Orgnize object data into a dictionary to be saved in json file
    @param trans: [3 array] object relative translation [m]
    @param quat: [4 array] Obj relative quaternion_xyzw
    @param centroid: [3 array] object cuboid centroid to cam frame [m]
    @param centroid_proj: [2 array] obj projected cuboid centroid
    @param cuboid: [8x3 array] object cuboid to cam frame
    @param cuboid_proj: [8x2 array] obj projected cuboid [m]
    @return data_dict: [dict] object data dictionary
    """
    # We should not need to use camera data in training
    # NOTE: All translations must be saved in [cm]
    # TODO(ziqi): Make this applicable to multi-objects
    data_dict = {
        "camera_data": {
            "location_worldframe": [0, 0, 0],
            "quaternion_xyzw_worldframe": [0, 0, 0, 1],
        },
        "objects": [
            {
                "class": obj,
                "location": (100 * trans).tolist(),
                "quaternion_xyzw": quat.tolist(),
                "bounding_box": {  # Loaded in train but not used
                    "top_left": [0, 0],
                    "bottom_right": [0, 0]
                },
                "cuboid_centroid": (100 * centroid).tolist(),
                "projected_cuboid_centroid": centroid_proj.tolist(),
                "cuboid": (100 * cuboid).tolist(),
                "projected_cuboid": cuboid_proj.tolist(),
            }
        ],
    }
    return data_dict


def main(obj, txt, ycb, ycb_json, out, fps=10.0,
         intrinsics=[1066.778, 1067.487, 312.9869, 241.3109270, 0]):
    """
    Save DOPE training data to target folder
    @param dim: [str] Object name
    @param txt: [str] path to .txt file (tum format) with object poses
    @param ycb: [str] path to ycb img folder
    @param ycb_json: [str] json file containing ycb object data
    @param out: [str] target folder to save the training data
    @param fps: [float] Sequence fps
    """
    # Get names of all the imgs in ycb folder
    img_fnames = sorted(glob.glob(ycb + "*-color.png"))
    # Get all the relative object poses
    indices, rel_trans, rel_quat = read_poses(txt)
    # Sanity check
    assert (
        len(img_fnames) >= rel_trans.shape[0]
    ), "Error: #img < #poses should never happen, check folder names"
    # Read object dimensions from ycb_original.json
    with open(ycb_json) as ycb_json:
        obj_data = json.load(ycb_json)
    # Names of all YCB objects
    class_names = obj_data["exported_object_classes"]
    # Find index of the current object
    obj_id = class_names.index(obj)
    # Load fixed dimensions for the object
    dim = obj_data["exported_objects"][obj_id]["cuboid_dimensions"]
    for ii in range(rel_trans.shape[0]):
        # continue
        # Index for image
        ind = int(indices[ii] * fps)
        # Copy img to target folder and rename by index
        copy_img(img_fnames[ind], out + "{:06}".format(ind + 1) + ".png")
        # compute cuboid of object
        cuboid = add_cuboid(rel_trans[ii, :], rel_quat[ii, :], dim)
        # Intrinsics hard coded for YCB sequence
        # TODO(ziqi): Make this a param
        cuboid_proj = project_cuboid(cuboid, intrinsics)
        # NOTE: Centroid is always object center for YCB objects
        # But may need to change this for other objects
        centroid = rel_trans[ii, :]
        # Project centroid to center using the same function
        centroid_proj = project_cuboid(centroid.reshape(1, 3), intrinsics)
        # Throw all the data into a dictionary
        data_dict = data2dict(
            obj, rel_trans[ii, :], rel_quat[ii, :],
            centroid, centroid_proj,
            cuboid, cuboid_proj
        )
        # Save dictionary to json file
        json_file = out + "{:06}".format(ind + 1) + ".json"
        with open(json_file, "w+") as fp:
            json.dump(data_dict, fp, indent=4, sort_keys=False)
    # TODO(ziqi): Save camera data and object data to json
    # Another sanity check
    assert (
        len(glob.glob(out + "*.png")) == rel_trans.shape[0]
    ), "Error: #imgs must always == #poses, check target folder"
    print("Data Generation Finished!")


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
    parser.add_argument("--seq", type=str,
                        help="YCB sequence id", default="0000")
    parser.add_argument(
        "--txt",
        type=str,
        help="Directory to txt (tum format) containing" +
        "relative object poses",
        default="/home/ziqi/Desktop/0000.txt",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Directory to save the imgs and labels",
        default="/home/ziqi/Desktop/test",
    )
    # There should be no need to change the following params
    parser.add_argument(
        "--ycb_json",
        type=str,
        help="Path to the _ycb_original.json file",
        default="/home/ziqi/Desktop/slam-super-6d/experiments/" +
        "ycbv/_ycb_original.json",
    )
    parser.add_argument(
        "--img",
        type=str,
        help="Training image name (with extension)",
        default="*-color.png",
    )
    parser.add_argument(
        "--fps", type=float, help="Sequence FPS", default=10.0
    )

    args = parser.parse_args()
    ycb_folder = args.ycb if args.ycb[-1] == "/" else args.ycb + "/"
    ycb_folder = ycb_folder + args.seq + "/"
    target_folder = args.out if args.out[-1] == "/" else args.out + "/"
    main(
        args.obj, args.txt, ycb_folder, args.ycb_json, target_folder, args.fps
    )
