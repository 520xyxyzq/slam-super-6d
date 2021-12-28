#!/usr/bin/env python3
# Solve pose graph optimization to generate pseudo-labels for 6D obj pose est
# Ziqi Lu, ziqilu@mit.edu
# Copyright 2022 The Ambitious Folks of the MRG

import argparse
import os

import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.quaternions import qisunit

# For GTSAM symbols
L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X


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
        for detf in det_files:
            det = self.readTum(detf)
            self._dets_.append(det)
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

    def buildGraph(self, prior_noise, odom_noise, det_noise, kernel, param):
        """
        Return odom and dets at next time step
        @param prior_noise: [1 or 6-array] Prior noise model
        @param odom_noise: [1 or 6-array] Camera odom noise model
        @param det_noise: [1 or 6-array] Detection noise model
        @param kernel: [int] robust kernel to use in PGO
        @param param: [float] robust kernel param
        """
        self._fg_ = gtsam.NonlinearFactorGraph()
        self._init_vals_ = gtsam.Values()

        prior_noise_model = self.readNoiseModel(prior_noise)
        odom_noise_model = self.readNoiseModel(odom_noise)
        det_noise_model = self.readNoiseModel(det_noise)

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
            self.prev_odom = odom
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
        # TODO(ZQ): Add enum class
        if optimizer == 0:
            params = gtsam.GaussNewtonParams()
            params.setVerbosity("ERROR")
            optim = gtsam.GaussNewtonOptimizer(
                self._fg_, self._init_vals_, params
            )
        elif optimizer == 1:
            params = gtsam.LevenbergMarquardtParams()
            params.setVerbosity("ERROR")
            optim = gtsam.LevenbergMarquardtOptimizer(
                self._fg_, self._init_vals_, params
            )
        elif optimizer == 2:
            params = gtsam.GaussNewtonParams()
            params.setVerbosity("ERROR")
            params = gtsam.GncGaussNewtonParams(params)
            optim = gtsam.GncGaussNewtonOptimizer(
                self._fg_, self._init_vals_, params
            )
        elif optimizer == 3:
            params = gtsam.LevenbergMarquardtParams()
            params.setVerbosity("ERROR")
            params = gtsam.GncLMParams()
            optim = gtsam.GncLMOptimizer(
                self._fg_, self._init_vals_, params
            )
        else:
            assert(False), "Error: Unknown optimizer type"

        result = optim.optimize()
        # result.print()
        return result

    def main(self):
        """
        Solve PGO and generate pseudo labels
        """
        while self._t_ != len(self._odom_):
            stamp = self._stamps_[self._t_]
            odom = self._odom_[stamp]
            for det in self._dets_:
                det_ = det.get(stamp, False)
                if det_:
                    lm = odom.compose(det_)
                    gtsam_plot.plot_point3(0, lm.translation(), "r.")
            if self._t_ % 20 == 0:
                fig = gtsam_plot.plot_pose3(0, odom, 0.1)

            self._t_ += 1

        axes = fig.gca(projection='3d')
        axes.view_init(azim=-90, elev=-45)
        axes.legend()
        plt.show()


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
        default=[root+"/experiments/ycbv/dets/results/0001_ycb_poses.txt"]
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
        "--out", type=str, help="Target folder to save the pseudo labels",
        default="/home/ziqi/Desktop/test"
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
        "--plot", "-p", action="store_true", help="Plot results?"
    )
    args = parser.parse_args()
    target_folder = args.out if args.out[-1] == "/" else args.out + "/"

    pl = PseudoLabeler(args.odom, args.dets)
    pl.buildGraph(args.prior_noise, args.odom_noise,
                  args.det_noise, args.kernel, args.kernel_param)
    pl.solve(args.optim)
    pl.main()
