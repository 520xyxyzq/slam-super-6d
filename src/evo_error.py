#!/usr/bin/env python3
# Evaluate recomputed object poses using EVO
# Ziqi Lu, ziqilu@mit.edu
# Copyright 2022 The Ambitious Folks of the MRG

import os

from evo.core import metrics, sync
from evo.main_ape import ape as ape_result
from evo.tools import file_interface


class EvoError:
    def __init__(self, dets, gt, out):
        """
        Load recomputed and ground truth object poses
        @param dets: [list of {stamp:gtsam.Pose3}] Recomputed object poses
        @param gt: [list of strings] Path to the ground truth object pose detes
        @param out: [string] Target folder to save data
        """

        # Read re-computed object pose detections
        assert(type(dets) == list and len(dets) > 0), \
            "Error: recomputed detections empty or not a list"
        for det in dets:
            assert(len(det) > 0), "Error: recomputed detections empty!"
        self._dets_ = dets
        # Read ground truth object pose detections
        assert(gt is not None), "Error: Ground truth object poses not passed"
        self._gt_ = gt
        assert(len(self._gt_) == len(self._dets_)), \
            "Error: #Ground truth detection files != #detection files"
        # Read output folder name
        self._out_ = out if out[-1] == "/" else out + "/"

    def error(self, verbose=False, save=False):
        """
        Print and save errors for pseudo labels of object pose detections
        @param gt_dets: [list of strings] ground truth object pose detections
        @param verbose: [bool] Print error stats?
        @param save: [bool] Save errors to file?
        """
        for ii, gt_det in enumerate(self._gt_):
            seq = os.path.basename(self._gt_[0])[:4]
            # TODO: This file name can change, so try to read it
            out_fname = self._out_ + seq + "_obj" + str(ii) + ".txt"
            if not os.path.isfile(out_fname):
                print('\033[93m' +
                      "WARN: Obj %s no recomputed dets, so error not computed"
                      % str(ii) + '\033[0m'
                      )
                continue
            t_error = self.ape(
                gt_det, out_fname, save=save,
                pose_relation=metrics.PoseRelation.translation_part,
                out_name=out_fname[:-4] + "_t_error.zip"
            )
            r_error = self.ape(
                gt_det, out_fname, save=save,
                pose_relation=metrics.PoseRelation.rotation_angle_rad,
                out_name=out_fname[:-4] + "_r_error.zip"
            )
            t_mean, t_median, t_std = \
                t_error["mean"], t_error["median"], t_error["std"]
            r_mean, r_median, r_std = \
                r_error["mean"], r_error["median"], r_error["std"]
            if verbose:
                print("Object %d error: " % (ii))
                print("  Translation part (m): ")
                print("    mean: %.6f; median: %.6f; std:  %.6f" %
                      (t_mean, t_median, t_std))
                print("  Rotation part (rad): ")
                print("    mean: %.6f; median: %.6f; std:  %.6f" %
                      (r_mean, r_median, r_std))

    def ape(self, traj_ref, traj_est, align_origin=False,
            pose_relation=metrics.PoseRelation.translation_part,
            save=False, out_name=None):
        """
        Compute APE btw 2 trajectories or 2 sets of detections
        @param traj_ref: [string] Reference trajectory file (tum format)
        @param traj_est: [string] Estimated trajectory file (tum format)
        @param align_origin: [bool] Align the origin of two trajs?
        @param pose_relation: [string] Metric used to compare poses,
        e.g. "translation part", "rotation angle in radians", etc.
        @param save: [bool] Save the result?
        @param out_name: [str] Absolute file name to save the result
        @return ape_stats: [dict] APE stats for the two trajs
        """
        assert(os.path.isfile(traj_ref)), "Error: %s not a file" % (traj_ref)
        assert(os.path.isfile(traj_est)), "Error: %s not a file" % (traj_est)
        # Read traj
        ref = file_interface.read_tum_trajectory_file(traj_ref)
        est = file_interface.read_tum_trajectory_file(traj_est)
        # Associate trajectories using time stamps
        ref, est = sync.associate_trajectories(ref, est)
        if align_origin:
            est.align_origin(ref)
        data = (ref, est)
        ape_metric = metrics.APE(pose_relation)
        ape_metric.process_data(data)
        if save:
            assert(out_name is not None), \
                "Error: APE result save path unspecified"
            result = ape_result(
                ref, est, pose_relation, align_origin=align_origin,
                ref_name=os.path.basename(traj_ref),
                est_name=os.path.basename(traj_est)
            )
            file_interface.save_res_file(out_name, result)
        return ape_metric.get_all_statistics()
