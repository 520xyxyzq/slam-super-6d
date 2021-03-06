#!/usr/bin/env python3
# Evaluate object pose to handle occlusions and bad pseudo labels
# Ziqi Lu, ziqilu@mit.edu
# Part of the codes are adapted from NVIDIA PoseRBPF
# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Copyright 2022 The Ambitious Folks of the MRG

import argparse
import copy
import glob
import json
import os

import gtsam
import numpy as np
import torch
from networks.aae_models import AAE
from PIL import Image
from scipy.spatial import cKDTree
from torchvision.ops import RoIAlign
from transforms3d.quaternions import mat2quat, qmult


class PoseEval:
    def __init__(self, imgs, obj, ckpt_file, codebook_file, intrinsics,
                 transform):
        """
        Load images, codebook and initialize Auto-Encoder model
        @param imgs: [string] Path to test image folder
        (e.g.: "<ycb_data>/*-color.png" for YCB-V images)
        @param obj: [string] Object name
        @param ckpt_file: [string] Path to object's checkpoint
        @param codebook_file: [string] Path to obj's codebook
        @param intrinsics: [5-list] Camera intrinsics
        @param transform: [4x4 array] Relative quat from DOPE to YCB coordinate
        """
        # Load Auto-Encoder model weights
        assert os.path.isfile(ckpt_file), \
            "Error: Checkpoint file doesn't exist!"
        self._aae_ = AAE([obj], "rgb")
        self._aae_.encoder.eval()
        self._aae_.decoder.eval()
        for param in self._aae_.encoder.parameters():
            param.requires_grad = False
        for param in self._aae_.decoder.parameters():
            param.requires_grad = False
        checkpoint = torch.load(ckpt_file)
        self._aae_.load_ckpt_weights(checkpoint['aae_state_dict'])
        self._encoder_ = copy.deepcopy(self._aae_.encoder)

        # Load codebook and codeposes
        assert os.path.isfile(codebook_file), \
            "Error: Codebook file doesn't exist!"
        self._codepose_ = torch.load(codebook_file)[1].cpu().numpy()
        self._codebook_ = torch.load(codebook_file)[0]
        assert self._codebook_.size(0) > 0, "Error: Codebook is empty!"

        # Set intrinsics
        self._intrinsics_ = intrinsics
        # GTSAM intrinsics
        self._K_ = gtsam.Cal3_S2(
            intrinsics[0], intrinsics[1], intrinsics[4], intrinsics[2],
            intrinsics[3]
        )

        # Load test images in order
        img_fnames = sorted(glob.glob(imgs))
        assert (len(img_fnames) > 0), \
            "Error: No image in folder, check folder name and img extension"
        self._imgs_ = []
        for img_file in img_fnames:
            img = Image.open(img_file)
            # NOTE: Don't forget to normalize the images
            img_np = np.asarray(img).copy() / 255.0
            self._imgs_.append(torch.from_numpy(img_np))

        # Quaternion to convert DOPE to YCB coordinate
        self._quat_trans_ = mat2quat(transform[:3, :3])

    def simScore(self, img_idx, pose, score_thresh=0.5, quat_thresh=0.2):
        """
        Check whether the pose is a good match and return cosine sim score
        @param img_idx: [int] Index for test image in self._imgs_
        @param pose: [gtsam.Pose3] Object pose (wrt camera frame)
        @param score_thresh: [float] Threshold for cosine similarity score
        @param quat_thresh: [float] Threshold for distance between the object's
        quaternion and its nearest neighbor with score > score_threshold
        @return simScore: [None/float] Cosine similarity score if the pose
        matches the image else None
        """
        # Rule out poses with zero z translation, which causes numerical issues
        if np.isclose(pose.z(), 0, atol=1e-2):
            return None

        # Compute the cosine similarity matrix
        assert(img_idx < len(self._imgs_)), "Error: Image index > # Images"
        img = self._imgs_[img_idx]
        roi_center = self.pose2RoICenter(pose)
        obj_center_depth = np.array([[pose.z()]])
        cosSimMat = self.computeCosSimMatrix(img, roi_center, obj_center_depth)

        # Not a match if all scores are below score_thresh
        if np.max(cosSimMat) < score_thresh:
            return None

        # Find all the quaternion above threshold
        assert(cosSimMat.shape[0] == 1), \
            "Error: Cosine similarity matrix wrong shape, should never happen!"
        codeposes = self._codepose_[
            np.argwhere(cosSimMat.ravel() > score_thresh).ravel(), :
        ]

        # NOTE: DOPE object coordinate is difference from PoseRBPF!!!
        # Applying a rotation transformation to align with PoseRBPF
        quat = qmult(pose.rotation().quaternion(), self._quat_trans_)

        # Find nearest neighbor and compute distance
        # NOTE: Need to query both q and -q
        kdtree = cKDTree(codeposes[:, 3:])
        dist1, idx1 = kdtree.query(quat)
        dist2, idx2 = kdtree.query(-quat)

        # Uncomment to debug
        # quat_pos = codeposes[idx1, 3:]
        # quat_neg = codeposes[idx2, 3:]
        # score_above_thresh = cosSimMat[cosSimMat > score_thresh]
        # print("Detected quaternion & nearest neighbor: ", quat, quat_pos)
        # print("- Detected quaternion & nearest neighbor: ", -quat, quat_neg)
        # print("Score: ", score_above_thresh[idx1], "; Distance: ", dist1)
        # print("-q Score: ", score_above_thresh[idx2], "; Distance: ", dist2)
        # print("Highest score: ", np.max(cosSimMat))

        # TODO: Use better distance function (Euclidean for now)
        # Not a match if the nearest neighbor has distance below quat_thresh
        if min(dist1, dist2) > quat_thresh:
            return None
        score_above_thresh = cosSimMat[cosSimMat > score_thresh]
        score = max(score_above_thresh[idx1], score_above_thresh[idx2])
        return score

    def pose2RoICenter(self, pose):
        """
        Convert object pose to object's RoI center in image
        NOTE: Assuming object coord origin is at object center
        @param pose: [gtsam.Pose3] Object pose (x y z qx qy qz qw)
        @return center: [Nx2 array] Centers of the ROIs [[u1, v1],...,[un, vn]]
        """
        # NOTE: Assume camera follow (x right y down z out) convention
        assert(pose.z() > 0), "Error: object (relative) pose must be positive"
        # Project object center to image
        cam = gtsam.PinholeCameraCal3_S2(gtsam.Pose3(), self._K_)
        point = cam.project(pose.translation())
        center = np.array([point])
        return center

    def computeCosSimMatrix(self, image, uvs, zs, target_distance=2.5):
        """
        Compute Cosine similarity matrix for an object ROI in image
        @param image: [Tensor] Source image (height x width x channel)
        @param uvs: [array] Centers of the ROIs [[u1, v1],...,[un, vn]]
        @param zs: [array] Z of object's 3D translation [[z1],...,[zn]]
        @param target_distance: [float] Scale the object in img to make it
        centered at target distance in the 3D space
        @return cosine_distance_matrix: [array] Cosine similarity matrix
        """
        images_roi_cuda, _ = self.get_rois_cuda(
            image.detach(), uvs, zs,
            self._intrinsics_[0],  self._intrinsics_[1],
            target_distance, out_size=128
        )

        # Plot 1st ROI to debug
        # np_img = (images_roi_cuda[0, :, :, :])\
        #     .permute(1, 2, 0).detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.imshow(np_img)
        # plt.show()

        # Forward passing
        n_rois = zs.shape[0]
        class_info = torch.ones((1, 1, 128, 128), dtype=torch.float32)
        class_info_cuda = class_info.cuda().repeat(n_rois, 1, 1, 1)
        images_input_cuda = torch.cat(
            (images_roi_cuda.detach(), class_info_cuda.detach()), dim=1
        )
        codes = self._encoder_.forward(images_input_cuda).\
            view(images_input_cuda.size(0), -1).detach()

        # Compute the similarity between codes and the codebook
        cosine_distance_matrix = \
            self._aae_.compute_distance_matrix(codes, self._codebook_)

        return cosine_distance_matrix.cpu().numpy()

    def get_rois_cuda(self, image, uvs, zs, fu, fv, target_distance=2.5,
                      out_size=128):
        """
        Crop the regions of interest (ROIs) from an image.
        The ROI is zero-padded if it exceeds the boundaries
        @param image: [Tensor] Source image (height x width x channel)
        @param uvs: [array] Centers of the ROIs [[u1, v1, 1],...,[un, vn, 1]]
        @param zs: [array] Z of object's 3D translation [[z1],...,[zn]]
        @param fu: [float] camera focal length fx for the current image
        @param fv: [float] camera focal length fy for the current image
        @param target_distance: [float] Scale the object in img to make it
        centered at target distance in the 3D space
        @param out_size: [int] Out image size
        @return out: [Tensor] Bounding boxes (1 x channel x height x width]
        @return uv_scale: [Tensor] Scales of the bounding boxes
        """
        # camera intrinsics used to compute the codebooks
        fu0, fv0 = 1066.778, 1067.487

        # Convert [height, width, channel] to [1, channel, height, width]
        image = image.permute(2, 0, 1).float().unsqueeze(0).cuda()

        # Compute bounding box width and height
        bbox_u = \
            target_distance * (1 / zs) / fu0 * fu * out_size / image.size(3)
        bbox_u = torch.from_numpy(bbox_u).cuda().float().squeeze(1)
        bbox_v = \
            target_distance * (1 / zs) / fv0 * fv * out_size / image.size(2)
        bbox_v = torch.from_numpy(bbox_v).cuda().float().squeeze(1)

        # Compute bounding box centers
        center_uvs = torch.from_numpy(uvs).cuda().float()
        center_uvs[:, 0] /= image.size(3)
        center_uvs[:, 1] /= image.size(2)

        # Compute bounding box vertices
        boxes = torch.zeros(center_uvs.size(0), 5).cuda()
        boxes[:, 1] = (center_uvs[:, 0] - bbox_u/2) * float(image.size(3))
        boxes[:, 2] = (center_uvs[:, 1] - bbox_v/2) * float(image.size(2))
        boxes[:, 3] = (center_uvs[:, 0] + bbox_u/2) * float(image.size(3))
        boxes[:, 4] = (center_uvs[:, 1] + bbox_v/2) * float(image.size(2))

        # Crop the ROIs from the image
        out = RoIAlign((out_size, out_size), 1.0, 0)(image, boxes)

        uv_scale = target_distance * (1 / zs) / fu0 * fu

        return out, uv_scale


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images", "-i", type=str, help="The folder with test images",
        default="/home/ziqi/Desktop/data/"
    )
    parser.add_argument(
        "--img_ext", "-ext", type=str, default="*-color.png",
        help="Extension used to identify test images files in folder"
    )
    parser.add_argument(
        "--obj", "-o", type=str, help="Object name (No _16k!!)",
        default="010_potted_meat_can"
    )
    parser.add_argument(
        "--ckpt", "-cp", type=str, help="Path to the AAE checkpoint folder",
        default=os.path.dirname(os.path.realpath(__file__)) + "/checkpoints/"
    )
    parser.add_argument(
        "--codebook", "-c", type=str, help="Path to the codebook folder",
        default=os.path.dirname(os.path.realpath(__file__)) + "/codebooks/"
    )
    parser.add_argument(
        "--intrinsics", "-in", type=float, nargs=5,
        help="Camera intrinsics: fx, fy, cx, cy, s",
        default=[1066.778, 1067.487, 312.9869, 241.3109, 0]
    )
    parser.add_argument(
        "--ycb_json", type=str, help="Path to the _ycb_original.json file",
        default=os.path.dirname(os.path.dirname(os.path.realpath(__file__))) +
        "/experiments/ycbv/_ycb_original.json"
    )
    args = parser.parse_args()
    codebook = args.codebook if args.codebook[-1] == "/" \
        else args.codebook + "/"
    ckpt = args.ckpt if args.ckpt[-1] == "/" else args.ckpt + "/"
    img_folder = args.images if args.images[-1] == "/" else args.images + "/"
    codebook += args.obj + ".pth"
    ckpt += args.obj + ".pth"

    # Read static transform from DOPE to YCB coordinate
    # Open _ycb_original.json to load models' static transformations
    with open(args.ycb_json) as yj:
        transforms = json.load(yj)
    # Names of all YCB objects
    class_names = transforms["exported_object_classes"]
    # Find index of the current object
    obj_id = class_names.index(args.obj + "_16k")
    # Load fixed transform for the obj (YCB to DOPE defined coordinate system)
    # NOTE: this is the transform of the original frame wrt the new frame
    obj_transf = \
        transforms["exported_objects"][obj_id]["fixed_model_transform"]
    obj_transf = np.array(obj_transf).T
    obj_transf[:3, :] /= 100

    pe = PoseEval(
        img_folder + args.img_ext, args.obj, ckpt, codebook, args.intrinsics,
        obj_transf
    )
    match_score = pe.simScore(
        0, gtsam.Pose3(gtsam.Rot3.Quaternion(0.8830, 0.2992, -0.3513, -0.0851),
                       gtsam.Point3(-0.0741, 0.0074, 1.0802))
    )
    print(match_score)
