/**
 * @file Graph.cpp
 * @brief Solve the pose graph optimization in an outlier-robust way
 * @author Ziqi Lu, ziqilu@mit.edu
 * Copyright 2021 The Ambitious Folks of the MRG
 */
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "../include/DataLoader.h"
#include "../include/DataSaver.h"

// TODO(ziqi): generalize to multi-object in each scene
int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "Usage: ./program odom_file_path det_file_path " << std::endl;
    return -1;
  }
  std::string odom_path = argv[1], det_path = argv[2];
  DataLoader data_loader(odom_path, det_path);

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values init_values;

  // prior factor noise order:rpyxyz
  gtsam::noiseModel::Diagonal::shared_ptr prior_noise =
      gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01).finished());
  // odom betfactor noise order:rpyxyz
  gtsam::noiseModel::Diagonal::shared_ptr odom_noise =
      gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01).finished());
  // detection betfactor noise order:rpyxyz
  gtsam::noiseModel::Diagonal::shared_ptr det_noise =
      gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());
  // null hypothesis noise model
  gtsam::noiseModel::Diagonal::shared_ptr nh_noise =
      gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector(6) << 100, 100, 100, 100, 100, 100).finished());

  gtsam::Pose3 pose, det, prev_pose;
  size_t count = 0;
  std::set<size_t> lm_ids;
  while (data_loader.next(&pose, &det)) {
    if (count == 0) {
      graph.addPrior<gtsam::Pose3>(gtsam::Symbol('x', count), pose,
                                   prior_noise);
    } else {
      gtsam::Pose3 odom_pose = prev_pose.inverse() * pose;
      graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
          gtsam::Symbol('x', count - 1), gtsam::Symbol('x', count), odom_pose,
          odom_noise));
    }

    // NOTE: det = Pose3() means there's no detection at this time
    if ((!det.equals(gtsam::Pose3()))) {
      // If this is the first observation of a landmark, initialize it!
      if (lm_ids.find(1) == lm_ids.end()) {
        init_values.insert(gtsam::Symbol('l', 1), pose * det);
        lm_ids.insert(1);
      }
      // TODO(ziqi): lm id hard-coded for now
      std::vector<gtsam::Key> mixture_keys = {gtsam::Symbol('x', count),
                                              gtsam::Symbol('l', 1)};
      std::vector<double> mixture_weights = {0.9, 0.1};
      std::vector<gtsam::BetweenFactor<gtsam::Pose3>> mixture_comps;
      mixture_comps.push_back(gtsam::BetweenFactor<gtsam::Pose3>(
          gtsam::Symbol('x', count), gtsam::Symbol('l', 1), det, det_noise));
      mixture_comps.push_back(gtsam::BetweenFactor<gtsam::Pose3>(
          gtsam::Symbol('x', count), gtsam::Symbol('l', 1), det, nh_noise));
      graph.add(MaxMixtureFactor<gtsam::BetweenFactor<gtsam::Pose3>>{
          mixture_keys, mixture_comps, mixture_weights});
    }

    prev_pose = pose;
    init_values.insert(gtsam::Symbol('x', count), pose);

    count++;
  }

  // Print large factor errors for initial values
  // for (const auto& fac : graph) {
  //   if (fac->error(init_values) > 5) {
  //     fac->printKeys();
  //     cout << "error: " << fac->error(init_values) << endl;
  //   }
  // }

  // Solve the optimization
  gtsam::GaussNewtonParams params;
  params.setVerbosity("ERROR");
  // params.setAbsoluteErrorTol(1e-06);
  // params.setRelativeErrorTol(1e-10);
  gtsam::Values result =
      gtsam::GaussNewtonOptimizer(graph, init_values, params).optimize();
  gtsam::Pose3 lm_result = result.at<gtsam::Pose3>(gtsam::Symbol('l', 1).key());
  std::cout << lm_result << endl;
  // std::cout << result.at<gtsam::Pose3>(gtsam::Symbol('x', 1110).key()) <<
  // endl;

  for (const auto& fac : graph) {
    if (fac->find(gtsam::Symbol('l', 1).key()) != fac->end()) {
      int id =
          dynamic_cast<MaxMixtureFactor<gtsam::BetweenFactor<gtsam::Pose3>>&>(
              *fac)
              .getComponent(result);
      if (id == 1) {
        fac->printKeys();
        cout << "error: " << fac->error(result) << endl;
      }
    }
  }

  DataSaver saver(graph, result);
  saver.computePoses();
  cout << "Done" << endl;
  return 0;
}
