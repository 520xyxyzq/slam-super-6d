/**
 * @file DataSaver.h
 * @brief Make training data files for the pose estimator
 * @author Ziqi Lu, ziqilu@mit.edu
 *
 * Copyright 2021 The Ambitious Folks of the MRG
 */

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/ProjectionFactor.h>

#include <fstream>
#include <string>
#include <utility>

#include "SmartMaxMixtureFactor.h"
class DataSaver {
 public:
  /**
   * @brief Constructor for the data saver
   * @param[in] graph Factor graph
   * @param[in] result Result from solving the graph
   */
  DataSaver(const gtsam::NonlinearFactorGraph &graph,
            const gtsam::Values &result);

  /**
   * @brief Compute the object detections from the result
   * @param[in] det_file Path to the result file
   * @param[in] cam2odom Camera to odom transformation
   * TODO(ziqi): should take multi file names for multi objects
   */
  void computeDets(std::string det_file, gtsam::Pose3 cam2odom);

  /**
   * @brief Extract poses from the result
   * @param[in] pose_file Path to the result file
   * @param[in] fps FPS (make this consistent to other files)
   */
  void computePoses(std::string pose_file);

  /**
   * @brief Check whether an object's centroid is in image
   * @param[in] pose Object pose (to camera) or inverse of extrinsics
   */
  bool isInImage(const gtsam::Pose3 pose);

  /**
   * @brief Filter function for extracting landmarks from estimates
   * @param[in] key key for a GTSAM variable
   * @return Whether the key points to a landmark variable
   */
  static bool isLandmarkKey(gtsam::Key key) {
    return (gtsam::Symbol(key).chr() == 'l');
  }

  /**
   * @brief Filter function for extracting robot poses from estimates
   * @param[in] key key for a GTSAM variable
   * @return Whether the key points to a robot pose variable
   */
  static bool isPoseKey(gtsam::Key key) {
    return (gtsam::Symbol(key).chr() == 'x');
  }

 private:
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values result_;
  gtsam::Cal3_S2::shared_ptr K_;
  std::pair<int, int> img_dim_;
  double fps_;
};

DataSaver::DataSaver(const gtsam::NonlinearFactorGraph &graph,
                     const gtsam::Values &result)
    : graph_(graph), result_(result) {
  // TODO(ziqi): This is YCB specific, should read them as params
  K_ = gtsam::Cal3_S2::shared_ptr(
      new gtsam::Cal3_S2(1066.778, 1067.487, 0, 312.9869, 241.3109));
  img_dim_ = {640, 480};
  // FPS of odometry (should make this consistent with other files)
  fps_ = 10.0;
}

bool DataSaver::isInImage(const gtsam::Pose3 rel_pose) {
  // TODO(ziqi): Use cheirality check by gtsam to see whether obj center in img
  // TODO(ziqi): Find transform from pose Origin to object centroid
  gtsam::PinholeCamera<gtsam::Cal3_S2> cam(gtsam::Pose3(), *K_);
  try {
    // TODO(ziqi): Check if the obj pose's origin is at center
    gtsam::Point2 point = cam.project(rel_pose.translation());
    if (point.x() > img_dim_.first || point.y() > img_dim_.second ||
        point.x() < 0 || point.y() < 0) {
      return false;
    }
    return true;
  } catch (gtsam::CheiralityException &e) {
    std::cout << "CheiralityException" << std::endl;
    return false;
  }
}

void DataSaver::computeDets(std::string det_file, gtsam::Pose3 cam2odom) {
  // Filter out landmark and pose result values
  gtsam::Values lm_values =
      result_.filter<gtsam::Pose3>(DataSaver::isLandmarkKey);
  gtsam::Values pose_values =
      result_.filter<gtsam::Pose3>(DataSaver::isPoseKey);
  // For each landmark variable compute relative pose
  for (auto const &lm_var : lm_values) {
    ofstream result_file;
    result_file.open(det_file);
    result_file.precision(12);
    // for each robot pose
    for (auto const &pose_var : pose_values) {
      gtsam::Pose3 rel_pose =
          cam2odom.inverse() *
          pose_values.at<gtsam::Pose3>(pose_var.key).inverse() *
          lm_values.at<gtsam::Pose3>(lm_var.key);
      if (isInImage(rel_pose)) {  // Cheirality check
        auto rel_t = rel_pose.translation();
        auto rel_q = rel_pose.rotation().quaternion();
        result_file << gtsam::Symbol(pose_var.key).index() / fps_;
        result_file << " " << rel_t.x() << " " << rel_t.y() << " " << rel_t.z();
        result_file << " " << rel_q(1) << " " << rel_q(2) << " " << rel_q(3)
                    << " " << rel_q(0) << std::endl;
      } else {
        cout << "Warning: camera at " << gtsam::Symbol(pose_var.key)
             << " out of img" << endl;
      }
    }
    result_file.close();
  }
}

void DataSaver::computePoses(std::string pose_file) {
  gtsam::Values pose_values =
      result_.filter<gtsam::Pose3>(DataSaver::isPoseKey);
  ofstream result_file;
  result_file.open(pose_file);
  result_file.precision(12);
  for (auto const &pose_var : pose_values) {
    auto pose = pose_values.at<gtsam::Pose3>(pose_var.key);
    auto t = pose.translation();
    auto q = pose.rotation().quaternion();
    result_file << gtsam::Symbol(pose_var.key).index() / fps_;
    result_file << " " << t.x() << " " << t.y() << " " << t.z();
    result_file << " " << q(1) << " " << q(2) << " " << q(3) << " " << q(0)
                << std::endl;
  }
  result_file.close();
}
