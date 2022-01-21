#!/usr/bin/env python3
# Evaluate object pose to handle occlusions and bad pseudo labels
# Ziqi Lu, ziqilu@mit.edu
# Part of the codes are adapted from NVIDIA PoseRBPF
# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Copyright 2022 The Ambitious Folks of the MRG

import argparse
import copy
import os

# import numpy as np
import torch
from networks.aae_models import AAE
# from PIL import Image
from torchvision.ops import RoIAlign


class PoseEva:
    def __init__(self, img_folder, img_ext, obj, ckpt_file, codebook_file,
                 intrinsics):
        """
        Load images, codebook and initialize Auto-Encoder model
        @param img_folder: [string] Path to test image folder
        @param img_ext: [string] Extension used to identify test image files
        in img_folder (e.g.: "-color.png" for YCB-V images)
        @param obj: [string] Object name
        @param ckpt_file: [string] Path to object's checkpoint
        @param codebook_file: [string] Path to obj's codebook
        @param intrinsics: [5-list] Camera intrinsics
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
        "--obj", "-o", type=str, help="Object name",
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
    args = parser.parse_args()
    codebook = args.codebook if args.codebook[-1] == "/" \
        else args.codebook + "/"
    ckpt = args.ckpt if args.ckpt[-1] == "/" else args.ckpt + "/"
    img_folder = args.images if args.images[-1] == "/" else args.images + "/"
    codebook += args.obj + ".pth"
    ckpt += args.obj + ".pth"

    pe = PoseEva(
        img_folder, args.img_ext, args.obj, ckpt, codebook, args.intrinsics
    )
