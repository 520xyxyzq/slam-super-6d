#!/usr/bin/env python3
# Solve pose graph optimization to generate pseudo-labels for 6D obj pose est
# Ziqi Lu, ziqilu@mit.edu
# Copyright 2022 The Ambitious Folks of the MRG

import argparse
import os
from enum import IntEnum

import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
from evo.core import metrics, sync
from evo.tools import file_interface
from transforms3d.quaternions import qisunit

# For GTSAM symbols
L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X


# Robusr kernels to be used in PGO
class Kernel(IntEnum):
    Gauss = 0
    MaxMix = 1
    Cauchy = 2
    GemanMcClure = 3
    Huber = 4
    Tukey = 5
    Welsch = 6


# NLS optimizers
class Optimizer(IntEnum):
    GaussNewton = 0
    LevenbergMarquardt = 1
    # NOTE: GNC by default uses TLS loss type
    # GNC assumes noise models are Gaussian
    # Will replace non-Gaussian robust kernel w/ Gaussian even if provided
    GncGaussNewton = 2
    GncLM = 3
    GncGaussNewtonGM = 4
    GncLMGM = 5


class PseudoLabeler(object):

    def __init__(self, odom_file, det_files):
        '''
        Read camera poses and detections
        @param odom_file: [string] Cam odom file name
        @param det_files: [list of strings] Detection files for target objects
        '''
        # Read camera odometry
        self._odom_ = self.readTum(odom_file)
        assert (len(self._odom_) > 0), \
            "Error: Cam odom file empty or wrong format"

        # Read object detection files
        self._dets_ = []
        # Save also detection file names for output
        self._det_fnames_ = []
        for detf in det_files:
            det = self.readTum(detf)
            self._dets_.append(det)
            det_fname = os.path.basename(detf)
            self._det_fnames_.append(det_fname)
            # TODO(ZIQI): check whether det stamps are subset of odom stamps
            assert (len(det) > 0), \
                "Error: Object det file empty or wrong format"

        # Get all time stamps
        self._stamps_ = sorted(list(self._odom_.keys()))

    def readTum(self, txt):
        """
        Read poses from txt file (tum format) into dict of GTSAM poses
        @param txt: [string] Path to the txt file containing poses
        @return poses: [dict] {stamp: gtsam.Pose3}
        """
        # TODO: assert tum format here
        assert(os.path.isfile(txt)), "Error: %s not a file" % (txt)
        rel_poses = np.loadtxt(txt)
        # Read poses into dict of GTSAM poses
        poses = {}
        for ii in range(rel_poses.shape[0]):
            # Skip lines with invalid quaternions
            if (qisunit(rel_poses[ii, 4:])):
                poses[rel_poses[ii, 0]] = \
                    self.tum2GtsamPose3(rel_poses[ii, 1:])
        return poses

    def tum2GtsamPose3(self, tum_pose):
        """
        Convert tum format pose to GTSAM Pose3
        @param tum_pose: [7-array] x,y,z,qx,qy,qz,qw
        @return pose3: [gtsam.pose3] GTSAM Pose3
        """
        assert(len(tum_pose) == 7), \
            "Error: Tum format pose must have 7 entrices"
        tum_pose = np.array(tum_pose)
        # gtsam quaternion order wxyz
        qx, qy, qz = tum_pose[3:-1]
        qw = tum_pose[-1]
        pose3 = gtsam.Pose3(
            gtsam.Rot3.Quaternion(qw, qx, qy, qz),
            gtsam.Point3(tum_pose[:3])
        )
        return pose3

    def gtsamPose32Tum(self, pose3):
        """
        Convert tum format pose to GTSAM Pose3
        @param pose3: [gtsam.pose3] GTSAM Pose3
        @return tum_pose: [7-array] x,y,z,qx,qy,qz,qw
        """
        tum_pose = np.ones((7,))
        tum_pose[0:3] = pose3.translation()
        quat = pose3.rotation().quaternion()
        # From gtsam wxyz to tum xyzw
        tum_pose[3:6] = quat[1:]
        tum_pose[6] = quat[0]
        return tum_pose

    def buildGraph(self, prior_noise, odom_noise, det_noise, kernel,
                   kernel_param=None, init=None):
        """
        Return odom and dets at next time step
        @param prior_noise: [1 or 6-array] Prior noise model
        @param odom_noise: [1 or 6-array] Camera odom noise model
        @param det_noise: [1 or 6-array or dict] Detection noise model
        @param kernel: [int] robust kernel to use in PGO
        @param kernel_param: [float] robust kernel param (use default if None)
        @param init: [gtsam.Values] User defined initial values for re-init
        """
        self._fg_ = gtsam.NonlinearFactorGraph()
        # TODO(ZQ): check init keys match all fg variables
        if init:
            self._init_vals_ = init
        else:
            self._init_vals_ = gtsam.Values()

        # Read noise models
        prior_noise_model = self.readNoiseModel(prior_noise)
        odom_noise_model = self.readNoiseModel(odom_noise)
        # Allow for different noise models for det factors, i.e. use dict
        # The flag isdict indicates whether noise model is factor dependent
        isdict = type(det_noise) is dict
        det_noise_model = \
            det_noise if isdict else self.readNoiseModel(det_noise)

        # Set robust kernel
        if kernel == Kernel.Gauss:
            pass
        elif kernel == Kernel.MaxMix:
            # TODO: add custom factor
            pass
        elif kernel in [Kernel.Cauchy, Kernel.GemanMcClure, Kernel.Huber,
                        Kernel.Tukey, Kernel.Welsch]:
            if kernel == Kernel.Cauchy:
                robust = gtsam.noiseModel.mEstimator.Cauchy(
                    kernel_param if kernel_param else 0.1
                )
            elif kernel == Kernel.GemanMcClure:
                robust = gtsam.noiseModel.mEstimator.GemanMcClure(
                    kernel_param if kernel_param else 1.0
                )
            elif kernel == Kernel.Huber:
                robust = gtsam.noiseModel.mEstimator.Huber(
                    kernel_param if kernel_param else 1.345
                )
            elif kernel == Kernel.Tukey:
                robust = gtsam.noiseModel.mEstimator.Tukey(
                    kernel_param if kernel_param else 4.6851
                )
            elif kernel == Kernel.Welsch:
                robust = gtsam.noiseModel.mEstimator.Welsch(
                    kernel_param if kernel_param else 2.9846
                )
            if isdict:
                det_noise_model = {k: gtsam.noiseModel.Robust(robust, n)
                                   for (k, n) in det_noise.items()}
            else:
                det_noise_model = \
                    gtsam.noiseModel.Robust(robust, det_noise_model)
        else:
            assert(False), "Error: Unknown robust kernel type"

        # Build graph
        it = 0
        while it != len(self._odom_):
            stamp = self._stamps_[it]
            odom = self._odom_[stamp]
            # Add (prior or) odom factor btw cam poses
            if it == 0:
                self._fg_.add(
                    gtsam.PriorFactorPose3(X(it), odom, prior_noise_model)
                )
            else:
                rel_pose = self.prev_odom.inverse().compose(odom)
                self._fg_.add(
                    gtsam.BetweenFactorPose3(X(it - 1), X(it), rel_pose,
                                             odom_noise_model)
                )
            # Add cam pose initial estimate
            if not init:
                self._init_vals_.insert(X(it), odom)
            # Remember previous odom pose to cpmpute relative cam poses
            self.prev_odom = odom

            # Add detection factor
            for ll, det in enumerate(self._dets_):
                detection = det.get(stamp, False)
                if detection:
                    # If landmark detected first time, add initial estimate
                    # TODO(zq): what if 1st pose det is outlier
                    if not init and not self._init_vals_.exists(L(ll)):
                        self._init_vals_.insert(L(ll), odom.compose(detection))
                    if isdict:
                        det_nm = det_noise_model[(X(it), L(ll))]
                    else:
                        det_nm = det_noise_model
                    self._fg_.add(
                        gtsam.BetweenFactorPose3(X(it), L(ll), detection,
                                                 det_nm)
                    )
            it += 1

    def readNoiseModel(self, noise):
        """
        Read noise as GTSAM noise model
        @param noise: [1 or 6 array] Noise model
        @return noise_model: [gtsam.noiseModel] GTSAM noise model
        """
        # Read prior noise model
        # TODO(ZQ): maybe allow for non-diagonal terms in noise model?
        if len(noise) == 1:
            noise_model = \
                gtsam.noiseModel.Isotropic.Sigma(6, noise[0])
        elif len(noise) == 6:
            noise_model = \
                gtsam.noiseModel.Diagonal.Sigmas(np.array(noise))
        else:
            assert(False), "Error: Noise model must have shape 1 or 6"
        return noise_model

    def solve(self, optimizer, verbose=False):
        """
        Solve robust pose graph optimization
        @param optimizer: [int] NLS optimizer for pose graph optimization
        @param verbose: [bool] Print optimization stats?
        @return result: [gtsam.Values] Optimization result
        """
        # Make sure solve is called after buildGraph()
        assert(hasattr(self, "_fg_")), \
            "Error: No factor graph yet, please build graph before solving it"

        if optimizer == Optimizer.GaussNewton:
            params = gtsam.GaussNewtonParams()
            if verbose:
                params.setVerbosity("ERROR")
            optim = gtsam.GaussNewtonOptimizer(
                self._fg_, self._init_vals_, params
            )
        elif optimizer == Optimizer.LevenbergMarquardt:
            params = gtsam.LevenbergMarquardtParams()
            if verbose:
                params.setVerbosity("ERROR")
            optim = gtsam.LevenbergMarquardtOptimizer(
                self._fg_, self._init_vals_, params
            )
        elif optimizer == Optimizer.GncGaussNewton:
            params = gtsam.GaussNewtonParams()
            params = gtsam.GncGaussNewtonParams(params)
            params.setVerbosityGNC(params.Verbosity.SILENT)
            optim = gtsam.GncGaussNewtonOptimizer(
                self._fg_, self._init_vals_, params
            )
        elif optimizer == Optimizer.GncLM:
            params = gtsam.LevenbergMarquardtParams()
            params = gtsam.GncLMParams(params)
            params.setVerbosityGNC(params.Verbosity.SILENT)
            optim = gtsam.GncLMOptimizer(
                self._fg_, self._init_vals_, params
            )
        elif optimizer in [Optimizer.GncGaussNewtonGM, Optimizer.GncLMGM]:
            # TODO(any): keep an eye on this
            assert(False), \
                "Error: GTSAM Python GNC optim w/ GM loss under development"
        else:
            assert(False), "Error: Unknown optimizer type"

        self._result_ = optim.optimize()

    def solveByIter(self, prior_noise, odom_noise, det_noise, kernel,
                    kernel_param, optimizer, lmd=1, abs_tol=1e-5,
                    rel_tol=1e-2, max_iter=20, verbose=False):
        """
        Jointly optimize SLAM variables and detection noise models by
        alternative minimization
        @param prior_noise: [1 or 6-array] Prior noise model
        @param odom_noise: [1 or 6-array] Camera odom noise model
        @param det_noise: [1 or 6-array] INITIAL detection noise model
        @param kernel: [int] robust kernel to use in PGO
        @param kernel_param: [float] robust kernel param (use default if None
        @param optimizer: [int] NLS optimizer for pose graph optimization
        @param lmd: [float] Regularization coefficient
        @param abs_tol: [float] absolute error threshold
        @param rel_tol: [float] relative error decrease threshold
        @param max_iter: [int] Maxmimum number of iterations
        @param verbose: [bool] Print optimization stats?
        """
        prev_error, error, rel_err, count = 1e20, 1e19, 1e10, 0
        init_vals = None
        while prev_error > error and error > abs_tol and rel_err > rel_tol \
                and count < max_iter:
            self.buildGraph(
                prior_noise, odom_noise, det_noise, kernel, kernel_param,
                init_vals
            )
            self.solve(optimizer, verbose)
            # Update errors
            prev_error = error
            error = self._fg_.error(self._result_)
            rel_err = (prev_error - error) / prev_error
            # Recompute noise models
            factor_errors = self.getFactorErrors(self._fg_, self._result_)
            det_noise = self.recomputeNoiseModel(
                factor_errors, kernel, kernel_param, lmd
            )
            # Reinitialize using estimates from last iteration
            init_vals = self._result_
            count += 1

        # Log stopping reason
        if prev_error < error and verbose:
            print("Stopping iterations because error increased")
        if error <= abs_tol and verbose:
            print("Converged! Absolute error %.6f < %.6f" % (error, abs_tol))
        if prev_error > error and rel_err <= rel_tol and verbose:
            print("Converged! Relative decrease %.6f < %.6f" %
                  (rel_err, rel_tol))
        if count >= max_iter and verbose:
            print("Maximum iteration number reached.")

    def getFactorErrors(self, fg, result):
        """
        Get unwhitened error at all factors
        @param fg: [gtsam.NonlinearFactorGraph] Factor graph
        @param result: [gtsam.Values] PGO results
        @return errors: [dict{tuple:array}] Factor errors indexed by keys;
        Error order rpyxyz
        """
        errors = {}
        for ii in range(fg.size()):
            factor = fg.at(ii)
            keytuple = tuple(factor.keys())
            error = factor.unwhitenedError(result)
            errors[keytuple] = error
        return errors

    def isCamKey(self, key):
        """
        Whether a gtsam Key is a camera pose variable
        @param key: [gtsam.Key] GTSAM variable key
        @return isX: [bool] Key belongs to a camera pose variable?
        """
        return gtsam.Symbol(key).chr() == ord('x')

    def isObjKey(self, key):
        """
        Whether a gtsam Key is an object pose variable
        @param key: [gtsam.Key] GTSAM variable key
        @return isL: [bool] Key belongs to a object pose variable?
        """
        return gtsam.Symbol(key).chr() == ord('l')

    def recomputeNoiseModel(self, errors, kernel, kernel_param, lmd):
        """
        Recompute optimal noise models at all factors
        @param errors: [dict{tuple:array}] Factor unwhitened errors indexed by
        keys; Error order rpyxyz
        @param kernel: [int] robust kernel to use in PGO
        @param kernel_param: [float] robust kernel param (use default if None
        @param lmd: [float] Regularization coefficient
        @return noise_models: [dict{tuple:array}] optimal noise model
        """
        # TODO(zq): Guard against 0 errors
        if kernel == Kernel.Gauss:
            noise_models = \
                {k: gtsam.noiseModel.Diagonal.Sigmas((e**2 / lmd)**(1/4))
                 for (k, e) in errors.items()}
        elif kernel == Kernel.MaxMix:
            # TODO: implement this, use Gaussian reweighting for now
            noise_models = \
                {k: gtsam.noiseModel.Diagonal.Sigmas((e**2 / lmd)**(1/4))
                 for (k, e) in errors.items()}
        else:
            # TODO(ZQ): Can we generalize to optimization w/ robust kernels?
            noise_models = \
                {k: gtsam.noiseModel.Diagonal.Sigmas((e**2 / lmd)**(1/4))
                 for (k, e) in errors.items()}
        return noise_models

    def recomputeDets(self, verbose=False):
        """
        Recompute object pose detections (i.e. Pseudo labels) from PGO results
        @param verbose: [bool] Print object out of image message?
        @return _plabels_: [list of dict] [{stamp: gtsam.Pose3}] Pseudo labels
        """
        self._plabels_ = []
        # For each object
        for ii in range(len(self._dets_)):
            lm_pose = self._result_.atPose3(L(ii))
            obj_dets = {}
            # Recompute relative obj pose at each time step
            for jj, stamp in enumerate(self._stamps_):
                cam_pose = self._result_.atPose3(X(jj))
                rel_obj_pose = cam_pose.inverse().compose(lm_pose)
                # Skip is object center not in image
                if not self.isInImage(rel_obj_pose):
                    if verbose:
                        print("Obj %d not in image at stamp %.1f" %
                              (ii, stamp))
                    continue
                obj_dets[stamp] = rel_obj_pose
            self._plabels_.append(obj_dets)

    def isInImage(self, rel_obj_pose):
        """
        Check if the object (center) is in the image
        TODO(Zq): Handle situations where obj origin is not at center
        @param rel_obj_pose: [gtsam.pose] Relative object pose
        @return isInImage: [bool] Whether object is in image
        """
        # NOTE: Assume camera follow (x right y down z out) convention
        if rel_obj_pose.z() <= 0:
            return False
        # Project object center to
        cam = gtsam.PinholeCameraCal3_S2(gtsam.Pose3(), self._K_)
        point = cam.project(rel_obj_pose.translation())
        if point[0] > self._img_dim_[0] or point[1] > self._img_dim_[1] or \
                point[0] < 0 or point[1] < 0:
            return False
        return True

    def assembleData(self, data_dict):
        """
        Assemble pseudo labels (tum format) from relative object poses
        (GTSAM format)
        @param data_dict: [{stamp:gtsam.Pose3}] Recomputed obj detections
        @param data: [Nx8 array] stamp,x,y,z,qx,qy,qz,qw
        """
        assert(len(data_dict) > 0), "Error: Recomputed detections empty"
        stamps = sorted(data_dict.keys())
        data = np.zeros((len(stamps), 8))
        for ii, stamp in enumerate(stamps):
            data[ii, 0] = stamp
            data[ii, 1:] = self.gtsamPose32Tum(data_dict[stamp])
        return data

    def saveData(self, out, img_dim, intrinsics, verbose=False):
        """
        Save data to target folder
        @param out: [string] Target folder to save results
        @param img_dim: [2-list] Image dimension [width, height]
        @param intrinsics: [5-list] Camera intrinsics (fx, fy, cx, cy, s)
        @param verbose: [bool] Print object not in image msg?
        """
        self._img_dim_ = img_dim
        self._K_ = gtsam.Cal3_S2(
            intrinsics[0], intrinsics[1], intrinsics[4], intrinsics[2],
            intrinsics[3]
        )
        # Make sure saveData is called after self.solve()
        assert(hasattr(self, "_result_")), \
            "Error: No PGO results yet, please solve PGO before saving data"
        # Recompute obj pose detections
        self.recomputeDets(verbose)
        for ii, plabel in enumerate(self._plabels_):
            data = self.assembleData(plabel)
            out_fname = self._det_fnames_[ii]
            np.savetxt(
                out + out_fname[:-4] + "_obj" + str(ii) + out_fname[-4:],
                data, fmt=["%.1f"] + ["%.12f"] * 7
            )

    def error(self, out, gt_dets=None, verbose=False, save=False):
        """
        Error analysis for pseudo labels of object pose detections
        @param out: [string] Target folder to save error.txt
        @param gt_dets: [list of strings] ground truth object pose detections
        @param save: [bool] Save errors to file?
        @return trans_error: [list of dict] Error stats for translation part
        @return rot_error: [list of dict] Error stats for rotation part (rad)
        """
        assert(hasattr(self, "_plabels_")), \
            "Error: No pseudo labels yet, generate data before error analysis"
        if gt_dets is None:
            return
        assert(len(gt_dets) == len(self._dets_)), \
            "Error: #Ground truth detection files != #detection files"
        trans_error, rot_error = [], []
        for ii, gt_det in enumerate(gt_dets):
            out_fname = self._det_fnames_[ii]
            out_fname = out + out_fname[:-4] + \
                "_obj" + str(ii) + out_fname[-4:]
            t_error = self.ape(
                gt_det, out_fname,
                pose_relation=metrics.PoseRelation.translation_part
            )
            r_error = self.ape(
                gt_det, out_fname,
                pose_relation=metrics.PoseRelation.rotation_angle_rad
            )
            trans_error.append(t_error)
            rot_error.append(r_error)
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
        return trans_error, rot_error

    def saveError(self, out, trans_error, rot_error):
        """
        Save error stats to file
        """
        # TODO: implement saveError function
        pass

    def ape(self, traj_ref, traj_est, align_origin=False,
            pose_relation=metrics.PoseRelation.translation_part):
        """
        Compute APE btw 2 trajectories or 2 sets of detections
        @param traj_ref: [string] Reference trajectory file (tum format)
        @param traj_est: [string] Estimated trajectory file (tum format)
        @param align_origin: [bool] Align the origin of two trajs?
        @param pose_relation: [string] Metric used to compare poses,
        e.g. "translation part", "rotation angle in radians", etc.
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
        ape_stats = ape_metric.get_all_statistics()
        return ape_stats

    def plot(self, gt_cam=None):
        """
        Plot estimation results
        @param gt_cam: [str] (Optional) ground truth camera poses
        """
        assert(hasattr(self, "_result_")), \
            "Error: No PGO results yet, please solve PGO before plotting"
        it = 0
        while it != len(self._odom_):
            stamp = self._stamps_[it]
            odom = self._odom_[stamp]
            for det in self._dets_:
                det_ = det.get(stamp, False)
                if det_:
                    lm = odom.compose(det_)
                    fig = gtsam_plot.plot_point3(0, lm.translation(), "g.")
            it += 1

        axes = fig.gca(projection='3d')
        self.plot_traj(axes, self._result_, "b-", 2, "poses")
        self.plot_traj(axes, self._init_vals_, "g--", 2, "odom")
        if gt_cam:
            gt_cam_dict = self.readTum(gt_cam)
            # Align origin
            pose_origin = gt_cam_dict[min(gt_cam_dict)]
            gt_cam_dict = {t: pose_origin.inverse() * p for (t, p)
                           in gt_cam_dict.items()}
            gt_cam_array = self.assembleData(gt_cam_dict)
            axes.plot3D(
                gt_cam_array[:, 1], gt_cam_array[:, 2], gt_cam_array[:, 3],
                "k--", linewidth=2, label="ground truth"
            )
        for ii in range(len(self._dets_)):
            lm_point = self._result_.atPose3(L(ii)).translation()
            gtsam_plot.plot_point3(0, lm_point, "r*")

        axes.view_init(azim=-90, elev=-45)
        axes.legend()
        plt.show()

    def plot_traj(self, ax, result, linespec="k-", linewidth=2, label="poses"):
        """
        Plot camera trajectory
        @param ax: [matplotlib.pyplot.plot.plot] Plot before
        @param result: [gtsam.Values] PGO results (w/ landmark included)
        @param linespec: [string] line color and type
        @param linewidth: [float] linewidth
        @param label: [string] legend for this line
        """
        positions = np.zeros((len(self._stamps_), 3))
        for ii, stamp in enumerate(self._stamps_):
            pose = result.atPose3(X(ii))
            positions[ii, 0] = pose.x()
            positions[ii, 1] = pose.y()
            positions[ii, 2] = pose.z()
        ax.plot3D(
            positions[:, 0], positions[:, 1], positions[:, 2], linespec,
            linewidth=linewidth, label=label
        )


if __name__ == '__main__':

    # Package root directory
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # Read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--odom", "-o", type=str,
        help="Camera odometry file (tum format)",
        default=root + "/experiments/ycbv/odom/results/0001.txt"
    )
    parser.add_argument(
        "--dets", "-d", nargs="+", type=str,
        help="Object detection files (tum format)",
        default=[root + "/experiments/ycbv/dets/results/0001_ycb_poses.txt"]
    )
    parser.add_argument(
        "--prior_noise", "-pn", nargs="+", type=float, default=[0.01],
        help="Prior noise model (std)"
    )
    parser.add_argument(
        "--odom_noise", "-on", nargs="+", type=float, default=[0.01],
        help="Camera odometry noise model (std)"
    )
    parser.add_argument(
        "--det_noise", "-dn", nargs="+", type=float, default=[0.1],
        help="Detection (initial) noise model (std)"
    )
    parser.add_argument(
        "--kernel", "-k", type=int, default=0,
        help="Robust kernel used in pose graph optimization"
    )
    parser.add_argument(
        "--kernel_param", "-kp", type=float, default=None,
        help="Parameter for robust kernel (if None use default)"
    )
    parser.add_argument(
        "--optim", "-op", type=int, default=0,
        help="Optimizer for pose graph optimization"
    )
    parser.add_argument(
        "--out", type=str, help="Target folder to save the pseudo labels",
        default="/home/ziqi/Desktop/test"
    )
    parser.add_argument(
        "--img_dim", "-idim", type=float, nargs=2, default=[640, 480],
        help="Image dimension (width, height)"
    )
    parser.add_argument(
        "--intrinsics", "-in", type=float, nargs=5,
        help="Camera intrinsics: fx, fy, cx, cy, s",
        default=[1066.778, 1067.487, 312.9869, 241.3109, 0]
    )
    parser.add_argument(
        "--gt_cam", "-gc", type=str, default=None,
        help="(Optional) Ground truth camera poses for plot & error analysis"
    )
    parser.add_argument(
        "--gt_obj", "-go", nargs="+", type=str, default=None,
        help="(Optional) Ground truth obj poses for error analysis"
    )
    parser.add_argument(
        "--lmd", "-l", type=float, default=1,
        help="Regularization coefficient for the joint optimization"
    )
    parser.add_argument(
        "--joint", "-j", action="store_true",
        help="Auto-tune detection noise model?"
    )
    parser.add_argument(
        "--plot", "-p", action="store_true", help="Plot results?"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print?"
    )
    args = parser.parse_args()
    target_folder = args.out if args.out[-1] == "/" else args.out + "/"

    pl = PseudoLabeler(args.odom, args.dets)
    if args.joint:
        pl.solveByIter(
            args.prior_noise, args.odom_noise, args.det_noise, args.kernel,
            args.kernel_param, args.optim, lmd=args.lmd, verbose=args.verbose
        )
    else:
        pl.buildGraph(args.prior_noise, args.odom_noise,
                      args.det_noise, args.kernel, args.kernel_param)
        pl.solve(args.optim, verbose=args.verbose)

    pl.saveData(target_folder, args.img_dim, args.intrinsics, args.verbose)
    pl.error(target_folder, args.gt_obj, verbose=args.verbose)
    if args.plot:
        pl.plot(args.gt_cam)
