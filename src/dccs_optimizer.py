#!/usr/bin/env python3
# Factor graph optimization by GNC based dynamic covariance COMPONENT scaling
# Ziqi Lu, ziqilu@mit.edu
# Copyright 2022 The Ambitious Folks of the MRG


import argparse

import gtsam


class DCCSOptimizer(object):
    def __init__(
        self, graph, init, kernel, kernel_param, factor2rescale,
        verbose=False
    ):
        """
        Read in the graph and initialize the factor graph.
        @param graph: [gtsam.NonlinearFactorGraph] Factor graph
        @param init: [gtsam.Values] Initial values
        @param kernel: [int] Robust kernel type
        @param kernel_param: [float] Robust kernel parameter
        @param factors2rescale: [list] Whose covariances to rescale
        (Factors are indexed by their key sets)
        """
        # Read as class variables
        self._graph_ = graph
        self._init_ = init
        self._kernel_ = kernel
        self._kernel_param_ = kernel_param
        self._factors2rescale_ = factor2rescale

        # Initial LM algorithm
        params = gtsam.LevenbergMarquardtParams()
        if verbose:
            params.setVerbosity("ERROR")
        self._LM_ = gtsam.LevenbergMarquardtOptimizer(graph, init, params)

    def rebuildGraph(self):
        """
        Rescale the factor covariances in the factor graph
        """
        pass

    def rescaleCovariance(self):
        """
        Rescale the factor covariances
        """
        pass

    def optimize(self):
        """
        Optimize by alternating minimization
        """
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel", type=int, default=0, help="robust kernel type"
    )
    parser.add_argument(
        "--kernel_param", type=float, default=None,  help="robust kernel param"
    )
    args = parser.parse_args()

    # A unit test: a landmark is observed from the 4 vertices of a square
    graph = gtsam.NonlinearFactorGraph()
    init = gtsam.Values()
    factor2rescale = list()

    # Odometry factors
    odom1 = gtsam.BetweenFactorPose2(
        1, 2, gtsam.Pose2(2, 0, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    )
    odom2 = gtsam.BetweenFactorPose2(
        2, 3, gtsam.Pose2(0, 2, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    )
    odom3 = gtsam.BetweenFactorPose2(
        3, 4, gtsam.Pose2(-2, 0, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    )
    odom4 = gtsam.BetweenFactorPose2(
        4, 1, gtsam.Pose2(0, -2, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    )

    # Landmark pose prediction factors
    pred1 = gtsam.BetweenFactorPose2(
        1, 0, gtsam.Pose2(1, 1, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    )
    pred2 = gtsam.BetweenFactorPose2(
        2, 0, gtsam.Pose2(-1, 1, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    )
    pred3 = gtsam.BetweenFactorPose2(
        3, 0, gtsam.Pose2(-1, -1, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    )
    # Outlier
    pred4 = gtsam.BetweenFactorPose2(
        4, 0, gtsam.Pose2(2, 2, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    )

    graph.add(odom1)
    graph.add(odom2)
    graph.add(odom3)
    graph.add(odom4)
    graph.add(pred1)
    graph.add(pred2)
    graph.add(pred3)
    graph.add(pred4)

    # NOTE: only rescale the pose prediction factors' covariances
    factor2rescale.append(set(pred1.keys()))
    factor2rescale.append(set(pred2.keys()))
    factor2rescale.append(set(pred3.keys()))
    factor2rescale.append(set(pred4.keys()))

    init.insert(1, gtsam.Pose2(0.1, 0.1, 0.1))
    init.insert(2, gtsam.Pose2(2, 0.1, 0.1))
    init.insert(3, gtsam.Pose2(2, 2, -0.1))
    init.insert(4, gtsam.Pose2(0.1, 2, -0.1))
    init.insert(0, gtsam.Pose2(0.9, 1.1, 0))

    dccs_optimizer = DCCSOptimizer(
        graph, init, args.kernel, args.kernel_param, factor2rescale,
        verbose=True
    )
    dccs_optimizer.optimize()
