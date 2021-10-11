#include "../include/DataLoader.h"

int main() {
  DataLoader data_loader(
      "/home/ziqi/Desktop/slam-super-6d/experiments/ycbv/odom/results/0001.txt",
      "/home/ziqi/Desktop/slam-super-6d/experiments/ycbv/dets/results/"
      "0001_ycb_poses.txt");

  gtsam::Pose3 odom, det;
  while (data_loader.next(&odom, &det)) {
    std::cout << odom << std::endl;
  }
  std::cout << "done" << std::endl;

  return 0;
}
