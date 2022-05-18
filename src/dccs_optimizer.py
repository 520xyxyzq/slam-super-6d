#!/usr/bin/env python3
# Factor graph optimization by GNC based dynamic covariance COMPONENT scaling
# Ziqi Lu, ziqilu@mit.edu
# Copyright 2022 The Ambitious Folks of the MRG

# We optimize the following loss:
# Loss = \sum_{k}\sum_{j} r^2_{kj} / \sigma^2_{kj} / s_{kj} + \Phi(1/s_{kj})
# by alternating minimization

import argparse
from enum import IntEnum

import gtsam
import numpy as np
from scipy.stats.distributions import chi2
from utils import printWarn


class Kernel(IntEnum):
    L2 = 0
    L1 = 1  # Truncated L1 (Lu et al. 2022)
    Cauchy = 2
    GemanMcClure = 3
    Huber = 4
    Tukey = 5
    Welsch = 6
    DCE = 7  # Pfeifer et al. (2017)


class DCCSOptimizer(object):
    def __init__(
        self, graph, init, kernel, kernel_param, factor2rescale,
        abs_tol=1e-10, rel_tol=2e-2, max_iter=20, verbose=False
    ):
        """
        Read in the graph and initialize the factor graph.
        @param graph: [gtsam.NonlinearFactorGraph] Factor graph
        @param init: [gtsam.Values] Initial values
        @param kernel: [int] Robust kernel type
        @param kernel_param: [float] Robust kernel parameter
        @param factors2rescale: [list] Whose covariances to rescale
        (Factors are indexed by the order they are added to the graph)
        @param abs_tol: [float] absolute error threshold
        @param rel_tol: [float] relative error decrease threshold
        @param max_iter: [int] Max number of iterations
        @param verbose: [bool] Print optimization stats?
        """
        # Make sure the factor2rescale list was properly populated
        assert(type(factor2rescale) == list), \
            "Error: factor2rescale must be a list"
        if len(factor2rescale) == 0:
            printWarn(
                "No covariances will be re-scaled, optimizer reduces to LM"
            )
        else:
            assert(all(type(elem) == int for elem in factor2rescale)), \
                "Error: factor indices must be integers"
            assert(len(set(factor2rescale)) <= graph.size()), \
                "Error: #Factor indices more than #factors in graph"
            assert(max(factor2rescale) < graph.size()), \
                "Error: Factor index out of range: " + \
                f"{max(factor2rescale)}/{graph.size()}"
            assert(min(factor2rescale) >= 0), \
                "Error: Factor index has to be non-negative"

        # Make sure the specified kernel is known
        kernels = set(item.value for item in Kernel)
        assert(kernel in kernels), f"Error: kernel id {kernel} " + \
            f"out of range 0-{len(kernels)-1}"

        # Read as class variables
        self._graph_ = graph
        self._result_ = init
        self._kernel_ = kernel
        self._kernel_param_ = self.readKernelParam(kernel, kernel_param)
        self._factors2rescale_ = factor2rescale
        self._abs_tol_ = abs_tol
        self._rel_tol_ = rel_tol
        self._max_iter_ = max_iter
        self._verbose_ = verbose

        # Make a chi2inv lookup table, calling the function in loop is slow
        self._chi2inv_ = dict()
        for fac in self._factors2rescale_:
            dim = graph.at(fac).dim()
            if dim not in self._chi2inv_:
                self._chi2inv_[dim] = chi2.ppf(0.95, dim)

        # Remember factors' initial noise models (std here)
        # TODO: dessemble the robust noise model or add an assertion
        self._sigmas_ = {
            fac: graph.at(fac).noiseModel().sigmas()
            for fac in self._factors2rescale_
        }

        # Intialize the covariance scaling constants as ones
        self._scale_ = {
            fac: np.ones_like(graph.at(fac).noiseModel().sigmas())
            for fac in self._factors2rescale_
        }

        # TODO: No need to freeze if we know the correct penalty
        # Freeze the penalty term for outlier measurements
        # to avoid their penalty values exaggerating the joint loss
        # only for truncated L1 and L2 loss
        self._freeze_ = {fac: False for fac in self._factors2rescale_}
        # Penalty values for factors to rescale
        self.updatePenalty()

        # Initial JOINT loss
        self._loss_ = \
            self._graph_.error(init) + sum(self._penalty_.values())
        # Initial previous JOINT loss (loss*1.2 to pass the convergence test)
        self._prev_loss_ = self._loss_ * 1.2

        # Number of iterations
        self._iter_ = 0

    def readKernelParam(self, kernel, kernel_param):
        """
        Read kernel parameters;
        Use default kernel parameters if not specified
        @param kernel: [Kernel] Robust kernel type
        @param kernel_param: [float or None] Robust kernel parameter
        @return kernel_param_read: [float or None] Updated kernel parameter
        """
        default_kernel_params = {
            Kernel.L2: None,
            Kernel.L1: 1000,
            Kernel.Cauchy: 0.1,
            Kernel.GemanMcClure: 1.0,
            Kernel.Huber: 1.345,
            Kernel.Tukey: 4.6851,
            Kernel.Welsch: 2.9846,
            Kernel.DCE: 0.1
        }
        if kernel_param is not None:
            return kernel_param
        else:
            return default_kernel_params[kernel]

    def penalty(self, scale):
        """
        Penalty function (Phi: R+ --> R+) in the joint loss
        Eq (16) in Black and Rangarajan, 1996; Eq (3) in Yang et al., 2020
        @param scale: [N-array] Scaling constants for covariance components
        @return penalty: [float] Penalty value for a measurement
        """
        # In case factor error is not a 1-D np array
        scale = np.array(scale).ravel()
        assert(all(scale >= 0)), "Error: negative covariance scale"
        # NOTE: In outlier processes, the penalty term is \Phi(weight)
        # For consistency, we compute \Phi(1/scale)
        # And we multiple it by 1/2 to match the least-squares loss (1/2e^2)
        if self._kernel_ == Kernel.L2:
            return 0
        elif self._kernel_ == Kernel.L1:
            return 1/2*np.sum(scale / (self._kernel_param_**2))
        elif self._kernel_ == Kernel.GemanMcClure:
            return 1/2*np.sum(
                (self._kernel_param_ * (1/np.sqrt(scale + 1e-20) - 1)**2)
            )

    def updatePenalty(self):
        """
        Update the penalty values for factors after rescaling
        """
        assert(hasattr(self, "_scale_")), \
            "Error: no covariance scale available"
        self._penalty_ = {
            fac: self.penalty(self._scale_[fac])
            for fac in self._factors2rescale_
            if not self._freeze_[fac]
        }

    def rescaleCovariances(self):
        """
        Rescale the factor covariances
        """
        assert(hasattr(self, "_result_")), \
            "Error: Need optimization results to rescale"
        for fac in self._factors2rescale_:
            # Get factor errors
            factor = self._graph_.at(fac)
            unwhitenedError = factor.unwhitenedError(self._result_)
            # error whitening
            whitenedError = unwhitenedError / self._sigmas_[fac]

            # Update the covariance scaling constants based on the errors
            size = len(whitenedError)  # measurement dim
            if self._kernel_ == Kernel.L2:
                self._scale_[fac] = np.ones(size)
            elif self._kernel_ == Kernel.L1:
                if np.linalg.norm(whitenedError)**2 <= self._chi2inv_[size]:
                    self._scale_[fac] = np.abs(
                        self._kernel_param_ * whitenedError
                    )
                else:
                    self._scale_[fac] = 1e22 * np.ones(size)
            elif self._kernel_ == Kernel.GemanMcClure:
                self._scale_[fac] = (
                    1 + (whitenedError / self._kernel_param_)**2
                )**2

            # Freeze the penalty value for outliers
            if self._kernel_ in [Kernel.L1]:
                self._freeze_[fac] = (np.sum(self._scale_[fac]) >= 1e22)

    def rebuildGraph(self):
        """
        Rescale the factor covariances in the factor graph
        NOTE: Don't change the order of the factors being added to graph
        """
        graph = gtsam.NonlinearFactorGraph()
        for id in range(self._graph_.size()):
            if id in self._factors2rescale_:
                factor = self._graph_.at(id)
                # NOTE: For now (05/22), GTSAM Python wrapper does not have the
                # factor->cloneWithNewNoiseModel() function
                # (GTSAM cpp version has this function)
                # We are doing it in a hacky way
                reweightedFactor = type(factor)(
                    *factor.keys(), factor.measured(),
                    gtsam.noiseModel.Diagonal.Sigmas(
                        np.sqrt(self._scale_[id])*self._sigmas_[id]
                    )
                )
                graph.add(reweightedFactor)
            else:
                graph.add(self._graph_.at(id).clone())
        self._graph_ = graph

    def isConverged(self):
        """
        Check if the joint optimization has converged
        @return is_converged: [bool] True if converged
        """
        # Log stopping reason
        if self._prev_loss_ < self._loss_:
            if self._verbose_:
                printWarn(
                    "WARN: Joint optimization stops because error increased"
                )
            return True
        if self._loss_ <= self._abs_tol_ and self._verbose_:
            if self._verbose_:
                print(
                    "Converged! Absolute error " +
                    f"{self._loss_:.6f} < {self._abs_tol_:.6f}"
                )
            return True
        # Relative loss decrease btw iterations
        rel_decrease = (self._prev_loss_ - self._loss_) / self._prev_loss_
        if self._prev_loss_ >= self._loss_ and rel_decrease <= self._rel_tol_:
            if self._verbose_:
                print(
                    "Converged! Relative decrease " +
                    f"{rel_decrease:.6f} < {self._rel_tol_:.6f}"
                )
            return True
        if self._iter_ >= self._max_iter_:
            if self._verbose_:
                print(f"Maximum iteration number {self._max_iter_} reached.")
            return True
        return False

    def optimize(self):
        """
        Optimize by alternating minimization
        @return result: [gtsam.Values] Optimized state variables
        """
        while not self.isConverged():
            # STEP 1: Solve standard least-squares optimization w/ LM algorithm
            params = gtsam.LevenbergMarquardtParams()
            if self._verbose_:
                params.setVerbosity("ERROR")
            self._LM_ = gtsam.LevenbergMarquardtOptimizer(
                self._graph_, self._result_, params
            )
            # Optimize
            self._result_ = self._LM_.optimize()
            # Update joint loss
            self._prev_loss_ = self._loss_
            # Update the inlier measurements' penalty values
            self.updatePenalty()
            # Update the joint loss
            self._loss_ = self._graph_.error(self._result_) + \
                sum(self._penalty_.values())
            # Number of iterations
            self._iter_ += 1

            # STEP 2: Rescale the factor covariances
            self.rescaleCovariances()
            self.rebuildGraph()
            if self._verbose_:
                print(
                    f"Joint loss at iteration {self._iter_}: {self._loss_}"
                )
        return self._result_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel", "-k", type=int, default=0, help="robust kernel type"
    )
    parser.add_argument(
        "--kernel_param", "-kp", type=float, default=None,
        help="robust kernel param"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print?"
    )
    args = parser.parse_args()

    # A unit test: a landmark is observed from the 4 vertices of a square
    graph = gtsam.NonlinearFactorGraph()
    init = gtsam.Values()
    factor2rescale = list()

    # Odometry factors
    odom1 = gtsam.BetweenFactorPose2(
        1, 2, gtsam.Pose2(2, 0, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.01)
    )
    odom2 = gtsam.BetweenFactorPose2(
        2, 3, gtsam.Pose2(0, 2, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.01)
    )
    odom3 = gtsam.BetweenFactorPose2(
        3, 4, gtsam.Pose2(-2, 0, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.01)
    )
    odom4 = gtsam.BetweenFactorPose2(
        4, 1, gtsam.Pose2(0, -2, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.01)
    )

    # Landmark pose prediction factors
    pred1 = gtsam.BetweenFactorPose2(
        1, 0, gtsam.Pose2(1, 1, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.5)
    )
    pred2 = gtsam.BetweenFactorPose2(
        2, 0, gtsam.Pose2(-1, 1, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.5)
    )
    pred3 = gtsam.BetweenFactorPose2(
        3, 0, gtsam.Pose2(-1, -1, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.5)
    )
    # Outlier
    pred4 = gtsam.BetweenFactorPose2(
        4, 0, gtsam.Pose2(2, 2, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.5)
    )

    # Prior factor on the first pose
    prior = gtsam.PriorFactorPose2(
        1, gtsam.Pose2(0, 0, 0), gtsam.noiseModel.Isotropic.Sigma(3, 0.001)
    )

    graph.add(odom1)  # 0
    graph.add(odom2)  # 1
    graph.add(odom3)  # 2
    graph.add(odom4)  # 3
    graph.add(pred1)  # 4
    graph.add(pred2)  # 5
    graph.add(pred3)  # 6
    graph.add(pred4)  # 7
    graph.add(prior)  # 8

    # NOTE: only rescale the pose prediction factors' covariances
    factor2rescale = [4, 5, 6, 7]

    init.insert(1, gtsam.Pose2(0.1, 0.1, 0.1))
    init.insert(2, gtsam.Pose2(2, 0.1, 0.1))
    init.insert(3, gtsam.Pose2(2, 2, -0.1))
    init.insert(4, gtsam.Pose2(0.1, 2, -0.1))
    init.insert(0, gtsam.Pose2(0.9, 1.1, 0))

    dccs_optimizer = DCCSOptimizer(
        graph, init, args.kernel, args.kernel_param, factor2rescale,
        verbose=args.verbose
    )
    dccs_optimizer.optimize()
    print(dccs_optimizer._result_)
