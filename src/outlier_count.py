#!/usr/bin/env python3
# Count outliers in object pose predictions
# Ziqi Lu, ziqilu@mit.edu
# Copyright 2022 The Ambitious Folks of the MRG

import argparse
import os

import gtsam
import numpy as np
from scipy.stats.distributions import chi2
from utils import readNoiseModel, readTum


class OutlierCount(object):
    def __init__(self, det, gt_obj):
        """
        Read pose predictions and ground truth object poses
        @param det: [str] Pose predictions (tum format txt)
        @param gt_obj: [str] Ground truth object poses (tum format txt)
        """
        assert(os.path.isfile(det)), "Error: %s not a file" % det
        assert(os.path.isfile(gt_obj)), "Error: %s not a file" % gt_obj
        self._det_ = readTum(det)
        self._gt_obj_ = readTum(gt_obj)

    def chi2test(self, det, gt, noise_model=[0.1], chi2_thresh=0.95):
        """
        Chi^2 test to check whether a pose prediction is an inlier
        @param det: [gtsam.Pose3] Object pose prediction
        @param gt: [gtsam.Pose3] Ground truth object pose wrt camera
        @param noise_model: [np.ndarray or gtsam.noiseModel] Noise model
        @param chi2_thresh: [float] Chi2 distribution confidence level
        @return isInlier: [bool] Whether a pose prediction is an inlier?
        """
        noise_model = readNoiseModel(noise_model)
        # Read pose prediction into a Pose3 btw factor
        factor = gtsam.BetweenFactorPose3(0, 1, det, noise_model)
        # Read ground truth pose into GTSAM values
        values = gtsam.Values()
        values.insert(0, gtsam.Pose3())
        values.insert(1, gt)
        # Compute factor errors
        errors = factor.unwhitenedError(values)
        # Get array of standard deviations from noise model
        stds = noise_model.sigmas()
        # Get critical value of chi2 distribution
        chi2inv = chi2.ppf(chi2_thresh, df=len(stds))
        # Chi2 test
        print(np.sum(errors**2 / stds**2))
        return np.sum(errors**2 / stds**2) < chi2inv

    def count_outliers(self):
        """
        Count the number of outliers, pose predictions and frames
        @return outliers: [int] Number of outliers
        @return num_dets: [int] Number of pose predictions
        @return num_frames: [int] Number of frames
        """
        count = 0
        for stamp, det in self._det_.items():
            assert(stamp in self._gt_obj_), \
                "Error: Stamp mismatch between predictions and ground truth!"
            if self.chi2test(det, self._gt_obj_[stamp]):
                count += 1
        return count, len(self._det_), len(self._gt_obj_)


if __name__ == '__main__':
    # Package root directory
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--det", "-d", type=str,
        help="Object pose predictions (tum format txt)",
        default=root +
        "/experiments/ycbv/dets/results/004_sugar_box_16k/0001.txt"
    )
    parser.add_argument(
        "--gt_obj", "-go", type=str,
        help="Ground truth obj poses (tum format txt)",
        default=root + "/experiments/ycbv/dets/ground_truth/" +
        "004_sugar_box_16k/0001_ycb_gt.txt"
    )
    args = parser.parse_args()

    # Count outliers, detections and total frames
    outlier_count = OutlierCount(args.det, args.gt_obj)
    print(outlier_count.count_outliers())
