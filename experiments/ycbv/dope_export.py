#!/usr/bin/env python3
# Generate and save DOPE training data from computed detections
# Ziqi Lu ziqilu@mit.edu

import argparse
import glob
import json
import os
import shutil

import numpy as np
from transforms3d.quaternions import qisunit, qmult, quat2mat


def add_cuboid(trans, quat, dim, new=False):
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
    trans, quat, dim = np.array(trans) * 100, np.array(quat), np.array(dim)
    if not new:
        # Helper vector to get vector from obj center to vertices
        # Vertex order here:
        # https://research.nvidia.com/publication/2018-06_Falling-Things
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
    else:
        # Vertex order for the new training script (centroid included)
        vert = np.array(
            [
                [1, -1, 1],
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, 1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, -1],
                [0, 0, 0]
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


def matchObjName(obj_sub, ycb_json):
    """
    Use substring to find full object name
    @param obj_sub: [string] Substring of object name
    @param ycb_json: [string] json file containing all ycb objects' data
    @return obj: [string] Object name
    """
    # Read object data from _ycb_original.json
    with open(ycb_json) as yj:
        obj_data = json.load(yj)
    # Names of all YCB objects
    class_names = obj_data["exported_object_classes"]
    # Object names with obj_sub in it
    matches = [s for s in class_names if obj_sub in s]
    assert (len(matches) == 1), "Error: No match or ambiguous obj name"
    return matches[0]


def copy_img(src, dest):
    """
    Copy img from one path to another
    @param src: Source file name
    @param dest: Destination file name
    """
    shutil.copy2(src, dest)


def data2Dict(obj, trans, quat, centroid, centroid_proj, cuboid, cuboid_proj):
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
                "projected_cuboid_centroid": centroid_proj.ravel().tolist(),
                "cuboid": (100 * cuboid).tolist(),
                "projected_cuboid": cuboid_proj.tolist(),
            }
        ],
    }
    return data_dict


def data2DictNew(obj, trans, quat, cuboid, cuboid_proj):
    """
    Orgnize object data into a dictionary to be saved in json file
    This is compatible with new DOPE training script for nvisii generated data
    @param trans: [3 array] object relative translation [m]
    @param quat: [4 array] Obj relative quaternion_xyzw
    @param cuboid: [9x3 array] object cuboid to cam frame
    @param cuboid_proj: [9x2 array] obj projected cuboid + centroid [m]
    @return data_dict: [dict] object data dictionary
    """
    # We should not need to use camera data in training
    # NOTE: All translations are saved in [m]
    # TODO(ziqi): Make this applicable to multi-objects

    # NOTE: The camera coordinate frame in NVISII generated data is following
    #       the blender convention: x right, y up, z in
    trans[1:] = -trans[1:]
    # This quaternion (order: wxyz) means rotating around x for 180 deg
    q_t = np.array([0, 1, 0, 0])
    # Pre-rotate the relative object orientation to get rel pose wrt new frame
    q_new = qmult(q_t, np.hstack((quat[-1], quat[:3])))
    q_new = np.hstack((q_new[1:], q_new[0]))

    data_dict = {
        "camera_data": {
            "location_world": [0, 0, 0],
            "quaternion_world_xyzw": [0, 0, 0, 1],
        },
        "objects": [
            {
                "class": obj,
                "visibility": 1,
                "location": trans.tolist(),
                "quaternion_xyzw": q_new.tolist(),
                "projected_cuboid": cuboid_proj.tolist()
            }
        ],
    }
    return data_dict


def camData2Dict(intrinsics, width, height):
    '''
    Throw camera data into a dictionary
    @param intrinsics: [5 array] Camera intrinsics
    @param width: [int] img width
    @param height: [int] img height
    @return cam_dict: [dict] camera data
    '''
    cam_dict = {"camera_settings": [
        {
            "name": "camera",
            "intrinsic_settings":
            {
                "fx": intrinsics[0],
                "fy": intrinsics[1],
                "cx": intrinsics[2],
                "cy": intrinsics[3],
                "s": intrinsics[4]
            },
            "captured_image_size":
            {
                "width": width,
                "height": height
            }
        }
    ]}
    return cam_dict


def objData2Dict(obj, ycb_json):
    '''
    Throw object data into a dictionary
    @param obj: [string] Object name
    @param ycb_json: [string] json file containing all ycb objects' data
    @return obj_dict: object data dictionary
    '''
    # Read object data from _ycb_original.json
    with open(ycb_json) as yj:
        obj_data = json.load(yj)
    # Names of all YCB objects
    class_names = obj_data["exported_object_classes"]
    # Find index of the current object
    obj_id = class_names.index(obj)
    # Load fixed dimensions for the object
    obj_data = obj_data["exported_objects"][obj_id]
    # Use white (255) bounding box to viz the obj in nvdu
    # And dosen't affect training
    obj_data["segmentation_class_id"] = 255
    # TODO(ziqi): Make this applicable to multi-objects
    obj_dict = {
        "exported_object_classes": [obj],
        "exported_objects": [obj_data]
    }
    return obj_dict

# TODO(ziqi): add a global settings file and make fps a global param


def main(obj, txt, ycb, ycb_json, out, hard=None, new=False,
         intrinsics=[1066.778, 1067.487, 312.9869, 241.3109, 0],
         width=640, height=480, fps=10.0):
    """
    Save DOPE training data to target folder
    @param obj: [str] Object name
    @param txt: [str] path to .txt file (tum format) with object poses
    @param ycb: [str] path to ycb img folder
    @param ycb_json: [str] json file containing all ycb objects' data
    @param out: [str] target folder to save the training data
    @param hard: [str] path to .txt file with hard examples' stamps
    @param new: [bool] Whether generate data for new DOPE training script
    @param intrinsics: [5 array] Camera intrinsics
    @param width: [int] img width
    @param height: [int] img height
    @param fps: [float] Sequence fps
    """
    # Get names of all the imgs in ycb folder
    img_fnames = sorted(glob.glob(ycb + "*-color.png"))
    # Get all the relative object poses
    indices, rel_trans, rel_quat = read_poses(txt)
    # Sanity check
    assert (len(img_fnames) >= rel_trans.shape[0]), \
        "Error: #img < #poses should never happen, check folder names"

    # Read object dimensions from ycb_original.json
    with open(ycb_json) as yj:
        obj_data = json.load(yj)
    # Names of all YCB objects
    class_names = obj_data["exported_object_classes"]
    # Use substring to get full object name
    obj_full = matchObjName(obj, ycb_json)
    # Find index of the current object
    obj_id = class_names.index(obj_full)
    # Load fixed dimensions for the object
    dim = obj_data["exported_objects"][obj_id]["cuboid_dimensions"]

    # Check whether the out folder exists, if not, create it.
    if not os.path.exists(out):
        os.makedirs(out)

    # Generating data
    for ii in range(rel_trans.shape[0]):
        # Index for image
        ind = int(indices[ii] * fps)
        # Copy img to target folder and rename by index
        copy_img(img_fnames[ind], out + "{:06}".format(ind + 1) + ".png")
        # compute cuboid of object
        cuboid = add_cuboid(rel_trans[ii, :], rel_quat[ii, :], dim, new)
        # Intrinsics hard coded for YCB sequence
        cuboid_proj = project_cuboid(cuboid, intrinsics)
        # NOTE: Centroid is always object center for YCB objects
        # But may need to change this for other objects
        centroid = rel_trans[ii, :]
        # Project centroid to center using the same function
        centroid_proj = project_cuboid(centroid.reshape(1, 3), intrinsics)
        # Throw all the data into a dictionary
        if new:
            data_dict = data2DictNew(
                obj, rel_trans[ii, :], rel_quat[ii, :],
                cuboid, cuboid_proj
            )
        else:
            data_dict = data2Dict(
                obj, rel_trans[ii, :], rel_quat[ii, :],
                centroid, centroid_proj,
                cuboid, cuboid_proj
            )
        # Save dictionary to json file
        json_file = out + "{:06}".format(ind + 1) + ".json"
        with open(json_file, "w+") as fp:
            json.dump(data_dict, fp, indent=4, sort_keys=False)

    # Duplicate hard examples
    if hard:
        assert(os.path.isfile(hard)), "Error: %s not a file" % hard
        hard_stamps = np.loadtxt(hard)
        assert(len(hard_stamps.shape) == 1), \
            "Error: %s shape must be (N,)" % hard
        assert(set(hard_stamps).issubset(indices)), \
            "Error: %s contains time stamps not in %s" % (hard, txt)

        for ii, stamp in enumerate(hard_stamps):
            # Get rel obj pose at this stamp
            stamp_ind, = np.where(np.isclose(indices, stamp))
            assert(len(stamp_ind) == 1), "Error: Duplicates in %s" % txt
            trans = rel_trans[stamp_ind[0], :]
            quat = rel_quat[stamp_ind[0], :]
            # Index for image
            ind = int(stamp * fps)
            # Copy img to target folder and rename by index
            copy_img(
                img_fnames[ind], out + "{:06}".format(ind + 1) + "_hard.png"
            )
            # compute cuboid of object
            cuboid = add_cuboid(trans, quat, dim, new)
            # Intrinsics hard coded for YCB sequence
            cuboid_proj = project_cuboid(cuboid, intrinsics)
            # NOTE: Centroid is always object center for YCB objects
            # But may need to change this for other objects
            centroid = trans
            # Project centroid to center using the same function
            centroid_proj = project_cuboid(centroid.reshape(1, 3), intrinsics)
            # Throw all the data into a dictionary
            if new:
                data_dict = data2DictNew(obj, trans, quat, cuboid, cuboid_proj)
            else:
                data_dict = data2Dict(
                    obj, trans, quat, centroid, centroid_proj, cuboid,
                    cuboid_proj
                )
            # Save dictionary to json file
            json_file = out + "{:06}".format(ind + 1) + "_hard.json"
            with open(json_file, "w+") as fp:
                json.dump(data_dict, fp, indent=4, sort_keys=False)

    # Save camera data into _camera_settings.json
    cam_dict = camData2Dict(intrinsics, width, height)
    cam_json_file = out + "_camera_settings.json"
    with open(cam_json_file, "w+") as fp:
        json.dump(cam_dict, fp, indent=4, sort_keys=False)

    # Save object data into _camera_settings.json
    obj_dict = objData2Dict(obj_full, ycb_json)
    obj_json_file = out + "_object_settings.json"
    with open(obj_json_file, "w+") as fp:
        json.dump(obj_dict, fp, indent=4, sort_keys=False)

    # Another sanity check
    if hard:
        assert(
            len(glob.glob(out + "*.png")) == rel_trans.shape[0] +
            hard_stamps.shape[0]
        ), "Error: Error: #imgs must always == #poses, check target folder"
    else:
        assert (len(glob.glob(out + "*.png")) == rel_trans.shape[0]), \
            "Error: #imgs must always == #poses, check target folder"
    print("Data Generation Finished!")


if __name__ == "__main__":
    # Read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj", type=str, help="Object name", default="004_sugar_box_16k"
    )
    parser.add_argument(
        "--ycb", type=str, help="Directory to YCB-V data folder",
        default="/media/ziqi/LENOVO_USB_HDD/data/YCB-V/data/",
    )
    parser.add_argument(
        "--seq", type=str, help="YCB sequence id", default="0000"
    )
    parser.add_argument(
        "--txt", type=str, default="/home/ziqi/Desktop/0000.txt",
        help="Directory to txt (tum format) containing" +
        "relative object poses",
    )
    parser.add_argument(
        "--out", type=str, default="/home/ziqi/Desktop/test",
        help="Directory to save the imgs and labels"
    )
    parser.add_argument(
        "--hard", help="Path to hard example txt file", default=None
    )
    # NOTE: DOPE released a new training script in Dec. 2021
    # It uses slightly different data format
    parser.add_argument(
        "--new", "-n", dest='new', action='store_true',
        help="Is this for new DOPE training script?"
    )
    parser.set_defaults(new=False)
    # NOTE: There should be no need to modify the following params
    parser.add_argument(
        "--ycb_json", type=str, help="Path to the _ycb_original.json file",
        default=os.path.dirname(os.path.realpath(__file__)) +
        "/_ycb_original.json"
    )
    parser.add_argument(
        "--img", type=str, help="Training image name (with extension)",
        default="*-color.png",
    )
    parser.add_argument(
        "--intrinsics", type=float, nargs=5,
        help="Camera intrinsics: fx, fy, cx, cy, s",
        default=[1066.778, 1067.487, 312.9869, 241.3109, 0]
    )
    parser.add_argument(
        "--width", type=int, help="Camera image width", default=640
    )
    parser.add_argument(
        "--height", type=int, help="Camera image height", default=480
    )
    parser.add_argument(
        "--fps", type=float, help="Sequence FPS", default=10.0
    )

    args = parser.parse_args()
    ycb_folder = args.ycb if args.ycb[-1] == "/" else args.ycb + "/"
    ycb_folder = ycb_folder + args.seq + "/"
    target_folder = args.out if args.out[-1] == "/" else args.out + "/"
    # If intrinsics not passed in but ycb seq number < 60, use default
    # If intrinsics not passed in but ycb seq number >= 60, use 2nd default
    if args.intrinsics == [1066.778, 1067.487, 312.9869, 241.3109, 0]:
        if int(args.seq) < 60:
            intrinsics = args.intrinsics
        else:
            # Default camera model is changed for seq 0060 ~ 0091 in YCB-V
            intrinsics = [1077.836, 1078.189, 323.7872, 279.6921, 0]
    else:
        # If intrinsics passed in, use it
        intrinsics = args.intrinsics

    main(
        args.obj, args.txt, ycb_folder, args.ycb_json, target_folder,
        hard=args.hard, new=args.new, intrinsics=intrinsics, width=args.width,
        height=args.height, fps=args.fps
    )
