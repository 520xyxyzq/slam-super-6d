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
                   kernel_param=None):
        """
        Return odom and dets at next time step
        @param prior_noise: [1 or 6-array] Prior noise model
        @param odom_noise: [1 or 6-array] Camera odom noise model
        @param det_noise: [1 or 6-array] Detection noise model
        @param kernel: [int] robust kernel to use in PGO
        @param kernel_param: [float] robust kernel param (use default if None)
        """
        self._fg_ = gtsam.NonlinearFactorGraph()
        self._init_vals_ = gtsam.Values()

        prior_noise_model = self.readNoiseModel(prior_noise)
        odom_noise_model = self.readNoiseModel(odom_noise)
        det_noise_model = self.readNoiseModel(det_noise)

        # Set robust kernel
        if kernel == Kernel.Gauss:
            pass
        elif kernel == Kernel.MaxMix:
            # TODO: add custom factor
            pass
        elif kernel == Kernel.Cauchy:
            if kernel_param:
                Cauchy = gtsam.noiseModel.mEstimator.Cauchy(kernel_param)
            else:
                Cauchy = gtsam.noiseModel.mEstimator.Cauchy(0.1)
            det_noise_model = gtsam.noiseModel.Robust(Cauchy, det_noise_model)
        elif kernel == Kernel.GemanMcClure:
            if kernel_param:
                GM = gtsam.noiseModel.mEstimator.GemanMcClure(kernel_param)
            else:
                GM = gtsam.noiseModel.mEstimator.GemanMcClure(1.0)
            det_noise_model = gtsam.noiseModel.Robust(GM, det_noise_model)
        elif kernel == Kernel.Huber:
            if kernel_param:
                Huber = gtsam.noiseModel.mEstimator.Huber(kernel_param)
            else:
                Huber = gtsam.noiseModel.mEstimator.Huber(1.345)
            det_noise_model = gtsam.noiseModel.Robust(Huber, det_noise_model)
        elif kernel == Kernel.Tukey:
            if kernel_param:
                Tukey = gtsam.noiseModel.mEstimator.Tukey(kernel_param)
            else:
                Tukey = gtsam.noiseModel.mEstimator.Tukey(4.6851)
            det_noise_model = gtsam.noiseModel.Robust(Tukey, det_noise_model)
        elif kernel == Kernel.Welsch:
            if kernel_param:
                Welsch = gtsam.noiseModel.mEstimator.Welsch(kernel_param)
            else:
                Welsch = gtsam.noiseModel.mEstimator.Welsch(2.9846)
            det_noise_model = gtsam.noiseModel.Robust(Welsch, det_noise_model)
        else:
            assert(False), "Error: Unknown robust kernel type"

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
            self._init_vals_.insert(X(it), odom)
            # Remember previous odom pose to cpmpute relative cam poses
            self.prev_odom = odom

            # Add detection factor
            for ll, det in enumerate(self._dets_):
                detection = det.get(stamp, False)
                if detection:
                    # If landmark detected first time, add initial estimate
                    if not self._init_vals_.exists(L(ll)):
                        self._init_vals_.insert(L(ll), odom.compose(detection))
                    self._fg_.add(
                        gtsam.BetweenFactorPose3(X(it), L(ll), detection,
                                                 det_noise_model)
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
            assert(False), "Error: Prior noise model must have shape 1 or 6"
        return noise_model

    def solve(self, optimizer):
        """
        Solve robust pose graph optimization
        @param optimizer: [int] NLS optimizer for pose graph optimization
        @return result: [gtsam.Values] Optimization result
        """
        # Make sure solve is called after buildGraph()
        assert(hasattr(self, "_fg_")), \
            "Error: No factor graph yet, please build graph before solving it"

        if optimizer == Optimizer.GaussNewton:
            params = gtsam.GaussNewtonParams()
            params.setVerbosity("ERROR")
            optim = gtsam.GaussNewtonOptimizer(
                self._fg_, self._init_vals_, params
            )
        elif optimizer == Optimizer.LevenbergMarquardt:
            params = gtsam.LevenbergMarquardtParams()
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

    def solveByIter(self, optimizer):
        """
        Jointly optimize SLAM variables and detection noise models by
        alternative minimization
        @param optimizer: [int] NLS optimizer for pose graph optimization
        """
        pass

    def recomputeDets(self):
        """
        Recompute object pose detections (i.e. Pseudo labels) from PGO results
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

    def saveData(self, out, img_dim, intrinsics):
        """
        Save data to target folder
        @param out: [string] Target folder to save results
        @param img_dim: [2-list] Image dimension [width, height]
        @param intrinsics: [5-list] Camera intrinsics (fx, fy, cx, cy, s)
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
        self.recomputeDets()
        for ii, plabel in enumerate(self._plabels_):
            data = self.assembleData(plabel)
            out_fname = self._det_fnames_[ii]
            np.savetxt(
                out + out_fname[:-4] + "_obj" + str(ii) + out_fname[-4:],
                data, fmt=["%.1f"] + ["%.12f"] * 7
            )

    def plot(self):
        """
        Plot estimation results
        """
        it = 0
        while it != len(self._odom_):
            stamp = self._stamps_[it]
            odom = self._odom_[stamp]
            for det in self._dets_:
                det_ = det.get(stamp, False)
                if det_:
                    lm = odom.compose(det_)
                    fig = gtsam_plot.plot_point3(0, lm.translation(), "r.")
            it += 1

        axes = fig.gca(projection='3d')
        self.plot_traj(axes, self._result_, "b-", 2, "poses")
        self.plot_traj(axes, self._init_vals_, "k--", 2, "odom")
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
        "--prior_noise", "-pn", nargs="+", type=float,
        help="Prior noise model (std)", default=[0.01]
    )
    parser.add_argument(
        "--odom_noise", "-on", nargs="+", type=float,
        help="Camera odometry noise model (std)", default=[0.01]
    )
    parser.add_argument(
        "--det_noise", "-dn", nargs="+", type=float,
        help="Detection (initial) noise model (std)", default=[0.1]
    )
    parser.add_argument(
        "--kernel", "-k", type=int,
        help="Robust kernel used in pose graph optimization", default=0
    )
    parser.add_argument(
        "--kernel_param", "-kp", type=float,
        help="Parameter for robust kernel (if None use default)",
        default=None
    )
    parser.add_argument(
        "--optim", "-op", type=int,
        help="Optimizer for pose graph optimization", default=0
    )
    parser.add_argument(
        "--out", type=str, help="Target folder to save the pseudo labels",
        default="/home/ziqi/Desktop/test"
    )
    parser.add_argument(
        "--img_dim", "-idim", type=float, nargs=2,
        help="Image dimension (width, height)", default=[640, 480]
    )
    parser.add_argument(
        "--intrinsics", "-in", type=float, nargs=5,
        help="Camera intrinsics: fx, fy, cx, cy, s",
        default=[1066.778, 1067.487, 312.9869, 241.3109, 0]
    )
    parser.add_argument(
        "--plot", "-p", action="store_true", help="Plot results?"
    )
    args = parser.parse_args()
    target_folder = args.out if args.out[-1] == "/" else args.out + "/"

    pl = PseudoLabeler(args.odom, args.dets)
    pl.buildGraph(args.prior_noise, args.odom_noise,
                  args.det_noise, args.kernel, args.kernel_param)
    pl.solve(args.optim)
    pl.saveData(target_folder, args.img_dim, args.intrinsics)
    if args.plot:
        pl.plot()
