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
from label_eval import LabelEval
from outlier_count import OutlierCount
from pose_eval import PoseEval
from scipy.spatial.transform import Rotation as R
from scipy.stats.distributions import chi2
from utils import (gtsamValuesToTum, pose3DictToTum, printWarn, readNoiseModel,
                   readTum)

# For GTSAM symbols
L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X


# Robust kernels to be used in PGO
class Kernel(IntEnum):
    L2 = 0
    L1 = 1
    Cauchy = 2
    GemanMcClure = 3
    Huber = 4
    Tukey = 5
    Welsch = 6
    DCE = 7


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


class LabelingMode(IntEnum):
    SLAM = 0
    Inlier = 1
    Hybrid = 2
    PoseEval = 3


class PseudoLabeler(object):

    def __init__(self, odom_file, det_files):
        '''
        Read camera poses and detections
        @param odom_file: [string] Cam odom file name
        @param det_files: [list of strings] Detection files for target objects
        '''
        # Read camera odometry
        self._odom_ = readTum(odom_file)
        assert (len(self._odom_) > 0), \
            "Error: Cam odom file empty or wrong format"

        # Read object detection files
        self._dets_ = []
        # Save also detection file names for output
        self._det_fnames_ = []
        for ll, detf in enumerate(det_files):
            det = readTum(detf)
            if len(det) == 0:
                printWarn("Warning: no detection for object %d" % ll)
                continue
            self._dets_.append(det)
            det_fname = os.path.basename(detf)
            self._det_fnames_.append(det_fname)
            # TODO(ZIQI): check whether det stamps are subset of odom stamps
            assert (len(det) > 0), \
                "Error: Object det file empty or wrong format"

        # Get all time stamps
        self._stamps_ = sorted(list(self._odom_.keys()))

    def buildGraph(self, prior_noise, odom_noise, det_noise, kernel,
                   kernel_param=None, init=None):
        """
        Build pose graph from odom and object pose measurements
        @param prior_noise: [1 or 6-array] Prior noise model
        @param odom_noise: [1 or 6-array] Camera odom noise model
        @param det_noise: [1 or 6-array or dict] Detection noise model
        @param kernel: [int] robust kernel to use in PGO
        @param kernel_param: [float] robust kernel param (use default if None)
        @param init: [gtsam.Values] User defined initial values for re-init
        """
        self._fg_ = gtsam.NonlinearFactorGraph()
        # TODO(ZQ): check init keys match all fg variables
        self._init_vals_ = init if init else gtsam.Values()
        # Use the avg object pose to initialize object pose variables
        if not init:
            for ll, dets in enumerate(self._dets_):
                poses = [self._odom_[tt] * det for tt, det in dets.items()]
                avg_pose = self.avgPoses(poses)
                self._init_vals_.insert(L(ll), avg_pose)

        # Read noise models
        prior_noise_model = readNoiseModel(prior_noise)
        odom_noise_model = readNoiseModel(odom_noise)
        det_noise_model = self.readRobustNoiseModel(
            det_noise, kernel, kernel_param
        )

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
                    if type(det_noise_model) is dict:
                        det_nm = det_noise_model[(X(it), L(ll))]
                    else:
                        det_nm = det_noise_model
                    self._fg_.add(
                        gtsam.BetweenFactorPose3(X(it), L(ll), detection,
                                                 det_nm)
                    )
            it += 1

    def readRobustNoiseModel(self, noise, kernel, kernel_param):
        """
        Read noise model and set robust kernel
        @param noise: [1 or 6-array/list or gtsam.noiseModel or dict] noise
        @param kernel: [int] robust kernel to use in PGO
        @param kernel_param: [float] robust kernel param (use default if None)
        @return robust_noise_model: [gtsam.noiseModel or dict] Robust noise
        model (dict in --> dict out; other in --> gtsam.noiseModel out)
        """
        # noise as dict: noise model is factor dependent
        # else: noise model is constant
        robust_noise_model = \
            noise if type(noise) is dict else readNoiseModel(noise)

        # Set robust kernel
        if kernel == Kernel.L2:
            pass
        elif kernel == Kernel.L1:
            assert(False), "Error: L1 kernel not implemented yet in GTSAM, " +\
                "but our DCCS optimizer supports L1 kernel"
        elif kernel == Kernel.DCE:
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
            # dict in --> dict out; other in --> gtsam.noiseModel out
            if type(noise) is dict:
                robust_noise_model = {k: gtsam.noiseModel.Robust(robust, n)
                                      for (k, n) in noise.items()}
            else:
                robust_noise_model = \
                    gtsam.noiseModel.Robust(robust, robust_noise_model)
        else:
            assert(False), "Error: Unknown robust kernel type"

        return robust_noise_model

    def avgPoses(self, poses):
        """
        Average a list of poses
        @param poses: [list of gtsam.Pose3] Poses to average
        @return avg_pose: [gtsam.Pose3] Average pose
        """
        assert(type(poses) is list and len(poses) > 0), \
            "Error: Pose list is empty"
        t = np.mean(np.array([pose.translation() for pose in poses]), axis=0)
        quats = np.array([pose.rotation().quaternion() for pose in poses])
        # GTSAM -> scipy Rotation quat order wxyz->xyzw
        quats = np.hstack((quats[:, 1:], quats[:, :1]))
        # Rotation Averaging
        quat = R.from_quat(quats).mean().as_quat()
        # scipy -> GTSAM Rotation quat order xyzw->wxyz
        quat = np.hstack((quat[-1], quat[:3]))
        return gtsam.Pose3(gtsam.Rot3.Quaternion(*quat), gtsam.Point3(t))

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
                    kernel_param, optimizer, lmd=1, abs_tol=1e-10,
                    rel_tol=1e-2, max_iter=20, verbose=False):
        """
        Jointly optimize SLAM variables and detection noise models by
        alternating minimization
        @param prior_noise: [1 or 6-array] Prior noise model
        @param odom_noise: [1 or 6-array] Camera odom noise model
        @param det_noise: [1 or 6-array] INITIAL detection noise model
        @param kernel: [int] robust kernel to use in PGO
        @param kernel_param: [float] robust kernel param (use default if None)
        @param optimizer: [int] NLS optimizer for pose graph optimization
        @param lmd: [float] Regularization coefficient
        @param abs_tol: [float] absolute error threshold
        @param rel_tol: [float] relative error decrease threshold
        @param max_iter: [int] Maxmimum number of iterations
        @param verbose: [bool] Print optimization stats?
        """
        prev_error, error, rel_err, count = 1e20, 1e19, 1e10, 0
        init_vals = None
        det_noise_new = det_noise
        while prev_error > error and error > abs_tol and rel_err > rel_tol \
                and count < max_iter:
            self.buildGraph(
                prior_noise, odom_noise, det_noise_new, kernel, kernel_param,
                init_vals
            )
            self.solve(optimizer, verbose)

            # Update errors
            prev_error = error
            # NOTE: The joint loss has an extra term (regularization)
            # TODO: if bootstrap, no self._outlier_ at this step in 1st iter
            if type(det_noise_new) == dict:
                assert(hasattr(self, "_outliers_")), \
                    "Error: Outliers not labeled yet!"
                # Regularization doesn't have outlier terms
                if not kernel == Kernel.DCE:
                    # TODO: may want to use initial noise * outliers as regular
                    regular = (1/lmd)**4 * np.sum([
                        np.linalg.norm(n.sigmas())**2 for (k, n) in
                        det_noise_new.items() if self.isDetFactor(k)
                        and self._stamps_[gtsam.Symbol(k[0]).index()] not in
                        self._outliers_[gtsam.Symbol(k[1]).index()]
                    ])
                else:
                    kernel_param = kernel_param if kernel_param else 0.1
                    regular = np.sum([
                        np.log(np.prod(n.sigmas()) / kernel_param**6)
                        for (k, n) in det_noise_new.items()
                        if self.isDetFactor(k)
                    ])
            else:
                n_edges = sum([len(det) for det in self._dets_])
                if not kernel == Kernel.DCE:
                    # In case of isotropic det noise model
                    noise_norm = np.sqrt(6) * np.linalg.norm(det_noise_new) \
                        if len(det_noise_new) == 1 \
                        else np.linalg.norm(det_noise_new)
                    regular = (1/lmd)**4 * n_edges * noise_norm**2
                else:
                    kernel_param = kernel_param if kernel_param else 0.1
                    # In case of isotropic det noise model
                    if len(det_noise_new) == 1:
                        regular = n_edges * np.log(
                            det_noise_new[0]**6 / kernel_param**6
                        )
                    else:
                        regular = n_edges * np.log(
                            np.prod(det_noise_new) / kernel_param**6
                        )
            error = self._fg_.error(self._result_) + regular
            rel_err = (prev_error - error) / prev_error

            # Recompute noise models
            factor_errors = self.getFactorErrors(self._fg_, self._result_)
            # TODO: Add a warning if det is a dict (for bootstrapping)
            self.labelOutliers(
                factor_errors,
                det_noise if not type(det_noise) == dict else 0.1
            )
            det_noise_new = self.recomputeNoiseModel(
                factor_errors, kernel, kernel_param, lmd
            )
            # Reinitialize using estimates from last iteration
            init_vals = self._result_
            count += 1
            if verbose:
                print("Joint loss at step %d: %.6f" % (count, error))

        # Log stopping reason
        if prev_error < error and verbose:
            printWarn(
                "Warning: Stopping iterations because error increased"
            )
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

    def isDetFactor(self, keytuple):
        """
        Whether a factor is an object pose detection factor from key tuple
        @param keytuple: [tuple] Tuple with factor keys, e.g.(X(10), L(0))
        @return isDetFactor: [bool] Is a detection factor?
        """
        return len(keytuple) == 2 and self.isObjKey(keytuple[1])

    def recomputeNoiseModel(self, errors, kernel, kernel_param, lmd):
        """
        Recompute optimal noise models at all factors
        @param errors: [dict{tuple:array}] Factor unwhitened errors indexed by
        keys; Error order rpyxyz
        @param kernel: [int] robust kernel to use in PGO
        @param kernel_param: [float] robust kernel param (use default if None)
        @param lmd: [float] Regularization coefficient
        @return noise_models: [dict{tuple:array}] optimal noise model
        """
        # TODO(zq): Guard against 0 errors
        assert(len(errors) > 0), "Error: Factor errors empty!"
        assert(hasattr(self, "_outliers_")), "Error: Outliers not labeled!"
        if kernel == Kernel.L2:
            noise_models = {}
            for (k, e) in errors.items():
                if self.isDetFactor(k):
                    stamp = self._stamps_[gtsam.Symbol(k[0]).index()]
                    obj_id = gtsam.Symbol(k[1]).index()
                    # Blow up Outliers' noise models to elim their influence
                    if stamp in self._outliers_[obj_id]:
                        noise_models[k] = gtsam.noiseModel.Isotropic.Sigma(
                            6, 1e10)
                    # Recompute Inlier noise models
                    else:
                        noise_models[k] = gtsam.noiseModel.Diagonal.Sigmas(
                            lmd * (e**2)**(1/4)
                        )
        elif kernel == Kernel.DCE:
            noise_models = {}
            for (k, e) in errors.items():
                if self.isDetFactor(k):
                    stamp = self._stamps_[gtsam.Symbol(k[0]).index()]
                    obj_id = gtsam.Symbol(k[1]).index()
                    # Set the sigma value as error otherwise as sigma_min
                    kernel_param = kernel_param if kernel_param else 0.1
                    sigma_new = abs(e)
                    sigma_new[sigma_new <= kernel_param] = kernel_param
                    noise_models[k] = gtsam.noiseModel.Diagonal.Sigmas(
                        sigma_new
                    )
        else:
            # Can we generalize this to robust kernels?
            # Maybe the reweighting process already robustifies the cost func
            printWarn(
                "Warning: No convergence guarantee if reweight w/ kernels"
            )
            noise_models = {}
            for (k, e) in errors.items():
                if self.isDetFactor(k):
                    stamp = self._stamps_[gtsam.Symbol(k[0]).index()]
                    obj_id = gtsam.Symbol(k[1]).index()
                    # Blow up Outliers' noise models to elim their influence
                    if stamp in self._outliers_[obj_id]:
                        noise_models[k] = gtsam.noiseModel.Isotropic.Sigma(
                            6, 1e10)
                    # Recompute Inlier noise models
                    else:
                        noise_models[k] = gtsam.noiseModel.Diagonal.Sigmas(
                            lmd * (e**2)**(1/4)
                        )
        return noise_models

    def computePseudoPoses(self, mode=0, scoreThreshPGO=0.5,
                           scoreThreshInlier=0.2):
        """
        Compute pseudo ground truth object poses (wrt camera) based on PGO
        results and inlier pose predictions
        @param mode: [int] Pseudo labeling mode: SLAM(0); Inlier(1); Hybrid(2);
        PoseEval(3)
        @param scoreThreshPGO: [float] Cosine similarity score threshold for
        PGO-generated labels
        @param scoreThreshInlier: [float] Cosine similarity score threshold for
        pseudo labels derived from inlier predictions

        NOTE: scoreThreshPGO > scoreThreshInlier since PGO generated labels are
        more likely to misalign with images
        @return _plabels_: [list of dict] [{stamp: gtsam.Pose3}] Pseudo ground
        truth object poses
        """
        self._plabels_ = []
        # Number of hybrid pseudo labels generated from PGO
        self.numHybridPGO = 0
        # For each object
        for ii, dets in enumerate(self._dets_):
            lm_pose = self._result_.atPose3(L(ii))
            obj_dets = {}
            # Recompute relative obj pose at each time step
            for jj, stamp in enumerate(self._stamps_):
                cam_pose = self._result_.atPose3(X(jj))
                rel_obj_pose = cam_pose.inverse().compose(lm_pose)
                # Skip if object center not in image
                # if not self.isInImage(rel_obj_pose):
                #     if verbose:
                #         print("Obj %d not in image at stamp %.1f" %
                #               (ii, stamp))
                #     continue
                if mode == LabelingMode.SLAM:
                    obj_dets[stamp] = rel_obj_pose
                elif mode == LabelingMode.Inlier:
                    if stamp not in dets:
                        continue
                    if stamp not in self._outliers_[ii]:
                        obj_dets[stamp] = dets[stamp]
                elif mode == LabelingMode.Hybrid:
                    # For hard examples, use SLAM result if cos_score > thresh
                    if stamp not in dets or stamp in self._outliers_[ii]:
                        score = self._pose_eval_.simScore(
                            jj, rel_obj_pose, scoreThreshPGO
                        )
                        if score:
                            # if verbose:
                            # print("Hard Example: Obj %d at stamp %.1f !!" %
                            #      (ii, stamp))
                            obj_dets[stamp] = rel_obj_pose
                        continue
                    # For inlier detections, use SLAM result if score higher
                    else:
                        score = self._pose_eval_.simScore(
                            jj, rel_obj_pose, scoreThreshPGO
                        )
                        score_in = self._pose_eval_.simScore(
                            jj, dets[stamp], scoreThreshInlier
                        )
                        # If max(score, score_inlier) < thresh, don't label
                        if not score_in and not score:
                            continue
                        # Use inlier detection as pseudo label if
                        # score < thresh < score_inlier
                        # OR score_inlier > max(thresh, score)
                        if score_in and (not score or score_in > score):
                            obj_dets[stamp] = dets[stamp]
                        else:
                            # Uncomment to print score values
                            # if score_in:
                            #     print(
                            #         "SLAM score %.2f > Inlier score %.2f" %
                            #         (score, score_in)
                            #     )
                            obj_dets[stamp] = rel_obj_pose
                            self.numHybridPGO += 1
                elif mode == LabelingMode.PoseEval:
                    if stamp not in dets:
                        continue
                    # NOTE: 0.5 was used in Deng et al. 2020
                    score = self._pose_eval_.simScore(
                        jj, dets[stamp], 0.5
                    )
                    if score:
                        obj_dets[stamp] = dets[stamp]
                else:
                    assert(False), "Error: Unknown pseudo labeling mode!"
            self._plabels_.append(obj_dets)

    def isInImage(self, rel_obj_pose):
        """
        Check if the object (center) is in the image
        NOTE: Function no longer in use since PoseEval can do the same job
        and inlier dets naturally guarantee objs in image;
        NOTE: Only case we need this is for SLAM labeling mode
        but at least for now we don't do direct PGO-labeling
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

    def labelOutliers(self, errors, det_std, chi2_thresh=0.95):
        """
        Label outlier object pose detections using Chi2 test
        @param errors: [dict{tuple:array}] Factor unwhitened errors indexed by
        keys; Error order rpyxy
        @param det_std: [array or gtsam.noiseModel] Standard deviation for
        detection noise
        @return _outliers_: [list of lists] Outlier stamps for each object
        """
        # NOTE: We should not use the det_noise after joint optimization to
        # label outliers. The outliers are downweighted and thus will pass
        # the chi2 test.
        assert(not type(det_std) == dict), \
            "Error: We should always use a fixed std for chi2 test"
        det_noise = readNoiseModel(det_std) if \
            type(det_std) in (np.ndarray, list) else det_std
        sigmas = det_noise.sigmas()
        chi2inv = chi2.ppf(chi2_thresh, df=len(sigmas))
        outlier_dict = {k: np.sum(e**2 / sigmas**2) > chi2inv
                        for (k, e) in errors.items() if self.isDetFactor(k)}

        # Convert the dict{keytuple: 0 or 1} to outlier stamps
        outliers = [[] for _ in range(len(self._dets_))]
        for (k, isout) in outlier_dict.items():
            if not self.isDetFactor(k) or not isout:
                continue
            x_index = gtsam.Symbol(k[0]).index()
            l_index = gtsam.Symbol(k[1]).index()
            stamp = self._stamps_[x_index]
            outliers[l_index].append(stamp)
        self._outliers_ = outliers

    def saveData(self, out, img_dim, intrinsics, det_std=[0.1], mode=0,
                 pose_eval=None, scoreThreshPGO=0.5, scoreThreshInlier=0.2,
                 detRateThresh=0.05, outlierRateThresh=0.2, verbose=False):
        """
        Save data (pseudo labels & hard examples' stamps) to target folder
        @param out: [string] Target folder to save results
        @param img_dim: [2-list] Image dimension [width, height]
        @param intrinsics: [5-list] Camera intrinsics (fx, fy, cx, cy, s)
        @param det_std: [array, noiseModel] Detection noise stds for chi2 test
        @param mode: [int] Pseudo labeling mode: SLAM(0) Inlier(1) Hybrid(2)
        PoseEval(3)
        @param pose_eval: [PoseEval or None] Pose evaluation module
        @param scoreThreshPGO: [float] Cosine similarity score threshold for
        PGO-generated labels
        @param scoreThreshInlier: [float] Cosine similarity score threshold for
        inlier predicts

        NOTE: scoreThreshPGO > scoreThreshInlier since PGO generated labels are
        more likely to misalign with images
        @param detRateThresh: [float] If %Detected < threshold, don't label seq
        @param outlierRateTresh: [float] %Outliers > threshold, don't label seq
        @param verbose: [bool] Print object not in image msg?
        """
        self._img_dim_ = img_dim
        self._K_ = gtsam.Cal3_S2(
            intrinsics[0], intrinsics[1], intrinsics[4], intrinsics[2],
            intrinsics[3]
        )
        assert(os.path.isdir(out)), "Error: Target folder doesn't exist!"
        # Make sure saveData is called after self.solve()
        assert(hasattr(self, "_result_")), \
            "Error: No PGO results yet, please solve PGO before saving data"
        if mode in [LabelingMode.Hybrid, LabelingMode.PoseEval]:
            assert(pose_eval is not None), \
                "Error: Pose evaluation is None, check labeling mode and input"
            self._pose_eval_ = pose_eval
        # Use chi2 test to find outliers
        errors = self.getFactorErrors(self._fg_, self._result_)
        self.labelOutliers(errors, det_std)
        # Recompute obj pose detections
        self.computePseudoPoses(mode, scoreThreshPGO, scoreThreshInlier)
        for ii, plabel in enumerate(self._plabels_):
            if len(plabel) == 0:
                printWarn(
                    "WARN: Obj %s data not saved, no valid pseudo label"
                    % str(ii)
                )
                continue
            # Save pseudo labels
            data = pose3DictToTum(plabel)
            out_fname = self._det_fnames_[ii]
            # Save hard examples (false positives and false negatives) to files
            fp = [stamp for stamp in self._outliers_[ii] if stamp in plabel]
            fn = [stamp for stamp in data[:, 0]
                  if stamp not in self._dets_[ii] and stamp in plabel]
            hard_egs = sorted(fp + fn)
            hard_egs = np.array([hard_egs]).T

            numFrames = len(self._stamps_)
            numDets = len(self._dets_[ii])
            numOutliers = len(self._outliers_[ii])
            # Save only when #dets > 5% #stamps and #outliers < 20% #dets
            # Or in PoseEval labeling mode
            if numDets > detRateThresh * numFrames \
                    and numOutliers < outlierRateThresh * numDets \
                    or mode == LabelingMode.PoseEval:
                np.savetxt(
                    out + out_fname[:-4] + "_obj" + str(ii) + out_fname[-4:],
                    data, fmt=["%.1f"] + ["%.12f"] * 7
                )
                np.savetxt(
                    out + out_fname[:-4] + "_obj" + str(ii) + "_hard" +
                    out_fname[-4:], hard_egs, fmt="%.1f"
                )
                if mode == LabelingMode.Hybrid:
                    # TODO: what if out_fname[:4] is not integar?
                    np.savetxt(
                        out + out_fname[:-4] + "_obj" + str(ii) + "_stats" +
                        out_fname[-4:],
                        [[int(out_fname[:4]), ii,
                          len(plabel) - hard_egs.shape[0] - self.numHybridPGO,
                          self.numHybridPGO, hard_egs.shape[0]]], fmt="%d"
                    )
                if verbose:
                    print("Data saved to %s!" % out)
            else:
                printWarn(
                    "WARN: Obj %d labels not saved!! #Dets: %d; #Outliers: %d"
                    % (ii, numDets, numOutliers)
                )

    def labelError(self, dim, intrinsics, out, gt_dets=None, verbose=False,
                   save=False):
        """
        Print and save errors in object keypoints (pseudo labels)
        @param dim: [3-list] Object dimension x, y, z
        @param out: [string] Target folder to save data
        @param gt_dets: [list of strings] ground truth object pose detections
        @param verbose: [bool] Print error stats?
        @param save: [bool] Save errors to file?
        """
        assert(hasattr(self, "_plabels_")), \
            "Error: No pseudo labels yet, generate data before error analysis"
        if gt_dets is None:
            printWarn(
                "WARN: Ground truth det files not passed, errors not computed"
            )
            return
        assert(len(gt_dets) == len(self._dets_)), \
            "Error: #Ground truth detection files != #detection files"
        for ii, gt_det in enumerate(gt_dets):
            if len(self._plabels_[ii]) == 0:
                printWarn(
                    "WARN: error not computed since pseudo label empty"
                )
                continue
            label_eval = LabelEval(self._plabels_[ii], gt_det, dim, intrinsics)
            mean, median, std = label_eval.error()
            if verbose:
                print("Object %d keypoint location error (pixel): " % (ii))
                print("    mean: %.2f; median: %.2f; std:  %.2f" %
                      (mean, median, std))
            if save:
                out_fname = self._det_fnames_[ii]
                out_fname = out + out_fname[:-4] + \
                    "_obj" + str(ii) + "_error" + out_fname[-4:]
                seq = os.path.basename(gt_det)[:4]
                np.savetxt(
                    out_fname, [[int(seq), ii, mean, median, std]],
                    fmt=["%d"]*2 + ["%.2f"]*3
                )

    def plot(self, gt_cam=None, gt_obj=None, save=False, out=None):
        """
        Plot estimation results
        @param gt_cam: [str] (Optional) ground truth camera poses
        @param gt_obj: [str] (Optional) ground truth object poses
        @param save: [bool] Save the figure?
        @param out: [string] Where to save the figure
        """
        assert(hasattr(self, "_result_")), \
            "Error: No PGO results yet, please solve PGO before plotting"
        it = 0
        while it != len(self._odom_):
            stamp = self._stamps_[it]
            odom = self._odom_[stamp]
            for ll, det in enumerate(self._dets_):
                det_ = det.get(stamp, False)
                if det_:
                    lm = odom.compose(det_)
                    # TODO(zq): make many plots for different objects
                    color = "y." if stamp in self._outliers_[ll] else "g."
                    fig = gtsam_plot.plot_point3(0, lm.translation(), color)
            it += 1

        axes = fig.gca(projection='3d')
        # Plot optimized camera poses
        cam_poses_opt = gtsamValuesToTum(self._result_, self._stamps_)
        axes.plot3D(
            cam_poses_opt[:, 1], cam_poses_opt[:, 2], cam_poses_opt[:, 3],
            "b-", linewidth=2, label="Traj. after PGO"
        )
        # Convert odom to np array and plot
        odom_poses = np.array(
            [pose.translation() for (stamp, pose) in self._odom_.items()]
        )
        axes.plot3D(
            odom_poses[:, 0], odom_poses[:, 1], odom_poses[:, 2], "g--",
            linewidth=2, label="Odom."
        )
        # Plot ground truth camera trajectory if any
        if gt_cam:
            gt_cam_dict = readTum(gt_cam)
            # Align origin
            pose_origin = gt_cam_dict[min(gt_cam_dict)]
            gt_cam_dict = {
                t: pose_origin.inverse() * p for (t, p) in gt_cam_dict.items()
            }
            gt_cam_array = pose3DictToTum(gt_cam_dict)
            axes.plot3D(
                gt_cam_array[:, 1], gt_cam_array[:, 2], gt_cam_array[:, 3],
                "k--", linewidth=2, label="Ground truth"
            )
        for ii in range(len(self._dets_)):
            lm_point = self._result_.atPose3(L(ii)).translation()
            gtsam_plot.plot_point3(0, lm_point, "r*")

        # Plot ground truth object pose if any
        if gt_obj:
            for kk, go in enumerate(gt_obj):
                obj_poses = readTum(go)
                # Ground truth obj pose is the first relative pose
                gt_obj_pose = obj_poses[min(obj_poses)]
                gtsam_plot.plot_point3(0, gt_obj_pose.translation(), "k*")

        axes.view_init(azim=-90, elev=-45)
        axes.legend()
        if save:
            assert(out is not None), "Error: Figure save path unspecified"
            assert(os.path.isdir(out)), "Error: Target folder doesn't exist!"
            plt.savefig(out + self._det_fnames_[0][:-4] + ".png", dpi=200)
        else:
            plt.show()


if __name__ == '__main__':

    # Package root directory
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # Read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--odom", "-o", type=str,
        help="Camera odometry file (tum format)",
        default=root + "/experiments/ycbv/odom/results/0002.txt"
    )
    parser.add_argument(
        "--dets", "-d", nargs="+", type=str,
        help="Object detection files (tum format)",
        default=[root + "/experiments/ycbv/inference/" +
                 "010_potted_meat_can_16k/Initial/0002.txt"]
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
        "--lmd", "-l", type=float, default=10,
        help="Regularization coefficient for the joint optimization"
    )
    parser.add_argument(
        "--joint", "-j", action="store_true",
        help="Auto-tune detection noise model?"
    )
    parser.add_argument(
        "--mode", "-m", type=int, default=0,
        help="Pseudo labeling mode: SLAM(0); Inlier(1); Hybrid(2); PoseEval(3)"
    )
    parser.add_argument(
        "--imgs", "-i", type=str, help="Path to images (with extensions)",
        default="/media/ziqi/LENOVO_USB_HDD/data/YCB-V/data/0002/*-color.png"
    )
    parser.add_argument(
        "--obj", "-obj", type=str, default="010_potted_meat_can",
        help="Object name (No _16k!!)"
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
        "--spgo", "-sp", type=float, default=0.5,
        help="Cosine similarity score threshold for PGO-generated labels"
    )
    parser.add_argument(
        "--sin", "-si", type=float, default=0.2,
        help="Cosine similarity score threshold for labels from inliers"
    )
    parser.add_argument(
        "--detthresh", "-dt", type=float, default=0.05,
        help="%Detected threshold, below which a seq is not pseudo labeled"
    )
    parser.add_argument(
        "--outthresh", "-ot", type=float, default=0.2,
        help="%Outlier threshold, above which a seq is not labeled"
    )
    parser.add_argument(
        "--ycb_json", type=str, help="Path to the _ycb_original.json file",
        default=root + "/experiments/ycbv/_ycb_original.json"
    )
    parser.add_argument(
        "--plot", "-p", action="store_true", help="Plot results?"
    )
    parser.add_argument(
        "--save", "-s", action="store_true", help="Save error statistics?"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print?"
    )
    args = parser.parse_args()
    target_folder = args.out if args.out[-1] == "/" else args.out + "/"

    # If intrinsics not passed in but ycb seq number < 60, use default
    # If intrinsics not passed in but ycb seq number >= 60, use 2nd default
    seq = os.path.basename(args.odom)[:4]
    if seq.isnumeric() and \
            args.intrinsics == [1066.778, 1067.487, 312.9869, 241.3109, 0]:
        if int(seq) < 60:
            intrinsics = args.intrinsics
        else:
            # Default camera model is changed for seq 0060 ~ 0091 in YCB-V
            intrinsics = [1077.836, 1078.189, 323.7872, 279.6921, 0]
    else:
        # If intrinsics passed in, use it
        intrinsics = args.intrinsics

    pl = PseudoLabeler(args.odom, args.dets)
    assert(len(pl._dets_) > 0), "No Object detection in the sequence"
    if args.joint:
        pl.solveByIter(
            args.prior_noise, args.odom_noise, args.det_noise, args.kernel,
            args.kernel_param, args.optim, lmd=args.lmd, verbose=args.verbose
        )
    else:
        pl.buildGraph(args.prior_noise, args.odom_noise,
                      args.det_noise, args.kernel, args.kernel_param)
        pl.solve(args.optim, verbose=args.verbose)

    # Instantiate pose evaluation module if labeling mode is "Hybrid"
    pose_eval = None
    if args.mode in [LabelingMode.Hybrid, LabelingMode.PoseEval]:
        codebook = args.codebook if args.codebook[-1] == "/" \
            else args.codebook + "/"
        ckpt = args.ckpt if args.ckpt[-1] == "/" else args.ckpt + "/"
        codebook += args.obj + ".pth"
        ckpt += args.obj + ".pth"
        import json

        # Open _ycb_original.json to load models' static transformations
        with open(args.ycb_json) as yj:
            transforms = json.load(yj)
        # Names of all YCB objects
        class_names = transforms["exported_object_classes"]
        # Find index of the current object
        obj_id = class_names.index(args.obj + "_16k")
        # Load fixed transform for the obj (YCB to DOPE defined coordinate)
        # NOTE: this is the transform of the original frame wrt the new frame
        obj_transf = \
            transforms["exported_objects"][obj_id]["fixed_model_transform"]
        obj_transf = np.array(obj_transf).T
        obj_transf[:3, :] /= 100  # [cm] to [m]
        pose_eval = PoseEval(
            args.imgs, args.obj, ckpt, codebook, intrinsics, obj_transf
        )

    pl.saveData(
        target_folder, args.img_dim, intrinsics, det_std=args.det_noise,
        mode=args.mode, pose_eval=pose_eval, scoreThreshPGO=args.spgo,
        scoreThreshInlier=args.sin, detRateThresh=args.detthresh,
        outlierRateThresh=args.outthresh, verbose=args.verbose
    )

    if args.gt_obj:
        import json

        # Open _ycb_original.json to load models' static transformations
        with open(args.ycb_json) as yj:
            transforms = json.load(yj)
        # Names of all YCB objects
        class_names = transforms["exported_object_classes"]
        # Find index of the current object
        obj_id = class_names.index(args.obj + "_16k")
        # Load dimensions of the obj
        obj_dim = transforms["exported_objects"][obj_id]["cuboid_dimensions"]
        pl.labelError(
            obj_dim, intrinsics, target_folder, args.gt_obj,
            verbose=args.verbose, save=args.save
        )
        # Count outliers in pose predictions based on ground truth obj poses
        for ii, go in enumerate(args.gt_obj):
            outlier_count = OutlierCount(args.dets[ii], go)
            num_out, num_det, num_frames = outlier_count.count_outliers()
            if args.verbose:
                print("Obj %d pose prediction stats (based on GT):" % ii)
                print(
                    "    #Outliers: %d; #Predictions: %d; #Frames: %d"
                    % (num_out, num_det, num_frames)
                )
                print(
                    "    %%Outliers: %.1f%%; %%Detected: %.1f%%"
                    % (num_out*100.0/num_det, num_det*100.0/num_frames)
                )

    # Uncomment to use EVO for error analysis (deprecated)
    # if pl._plabels_[0] and args.gt_obj:
    #     from evo_error import EvoError

    #     evo_error = EvoError(pl._plabels_, args.gt_obj, target_folder)
    #     evo_error.error(verbose=args.verbose, save=args.save)

    # Plot or save the traj and landmarks
    if args.plot or args.save:
        pl.plot(args.gt_cam, args.gt_obj, args.save, target_folder)
        if args.plot and args.save:
            print("Info: plot is saved to %s" % target_folder)
