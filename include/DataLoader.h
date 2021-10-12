/**
 * @file DataLoader.h
 * @brief Read odometry and detection data from files to a factor graph
 * @author Ziqi Lu, ziqilu@mit.edu
 *
 * Copyright 2021 The Ambitious Folks of the MRG
 */

#pragma once

#include <gtsam/geometry/Pose3.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <string>
// TODO(ziqi): generalize to multi object per scene (use det vector)

/**
 * @brief Data loader for the odometry and detection data.
 */
class DataLoader {
 public:
  /**
   * @brief Constructor for the data loader
   * @param[in] odom_file directory of odom file (string)
   * @param[in] det_file directory of odom file (string)
   */
  DataLoader(const std::string& odom_file, const std::string& det_file);

  /**
   * @brief Output the next time step
   * @param[in] odom_pose directory of odom file (string)
   * @param[in] det_pose directory of odom file (string)
   * @return False if the file is not in valid tum format
   */
  bool next(gtsam::Pose3* odom_pose, gtsam::Pose3* det_pose);

 private:
  std::map<double, gtsam::Pose3> odom_map_, det_map_;
};

DataLoader::DataLoader(const std::string& odom_file,
                       const std::string& det_file) {
  // Open files
  std::ifstream odom_data(odom_file);
  std::ifstream det_data(det_file);

  // check whether data directory is correct
  if (!odom_data.is_open()) {
    throw ::invalid_argument(odom_file + " does not exist");
  }
  if (!det_data.is_open()) {
    throw ::invalid_argument(det_file + " does not exist");
  }

  // Read odometry
  double t, x, y, z, qx, qy, qz, qw;
  while (odom_data >> t >> x >> y >> z >> qx >> qy >> qz >> qw) {
    gtsam::Pose3 pose(gtsam::Rot3::Quaternion(qw, qx, qy, qz),
                      gtsam::Point3(x, y, z));
    // key=t represents the odometry btw previous time step and t
    odom_map_.insert({t, pose});
  }

  // Read detections
  while (det_data >> t >> x >> y >> z >> qx >> qy >> qz >> qw) {
    gtsam::Pose3 pose(gtsam::Rot3::Quaternion(qw, qx, qy, qz),
                      gtsam::Point3(x, y, z));
    det_map_.insert({t, pose});
  }

  odom_data.close();
  det_data.close();
}

bool DataLoader::next(gtsam::Pose3* odom_pose, gtsam::Pose3* det_pose) {
  if (odom_map_.empty()) return false;

  // pop front the odom_pose map
  auto odom_pair = odom_map_.begin();
  double odom_t = odom_pair->first;
  *odom_pose = odom_pair->second;
  odom_map_.erase(odom_pair);
  // pop front the det_pose map
  auto det_pair = det_map_.begin();
  double det_t = det_pair->first;
  if (odom_t == det_t) {
    *det_pose = det_pair->second;
    det_map_.erase(det_pair);
  } else {
    det_pose = nullptr;
  }
  return true;
}
