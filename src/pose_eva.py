#!/usr/bin/env python3
# Evaluate object pose to handle occlusions and bad pseudo labels
# Ziqi Lu, ziqilu@mit.edu
# Part of the codes are adapted from NVIDIA PoseRBPF
# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full
# Copyright 2022 The Ambitious Folks of the MRG

import argparse
import copy
import os

import torch
from networks.aae_models import AAE


class PoseEva:
    def __init__(self, obj, ckpt_file, codebook_file, intrinsics):
        """
        Load codebook and initialize Auto-Encoder model
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
        self._codepose_ = torch.load(codebook_file)[1].cpu().numpy()
        self._codebook_ = torch.load(codebook_file)[0]
        assert self._codebook_.size(0) > 0, "Error: Codebook is empty!"

        # Set intrinsics
        self._intrinsics_ = intrinsics

    def computeCosSimMat(self, code, codebook, eps=1e-8):
        """
        Compute the cosine similarity matrix btw two code tensors
        @param code: [Tensor] batch of codes from the encoder
        (batch size * code size)
        @param codebook: [Tensor] codebook
        (codebook size * code size)
        @return matrix: [Tensor] cosine similarity matrix
        (batch size * code book size)
        """
        dot_product = torch.mm(code, torch.t(codebook))
        code_norm = torch.norm(code, 2, 1).unsqueeze(1)
        codebook_norm = torch.norm(codebook, 2, 1).unsqueeze(1)
        normalizer = torch.mm(code_norm, torch.t(codebook_norm))

        return dot_product / normalizer.clamp(min=eps)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
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
    codebook += args.obj + ".pth"
    ckpt += args.obj + ".pth"

    pe = PoseEva(args.obj, ckpt, codebook, args.intrinsics)
