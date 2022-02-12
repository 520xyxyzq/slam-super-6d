from evo.tools import file_interface
import numpy as np
from rosbags.rosbag1 import Reader
from utils import get_dirname, parse_inputs, get_save_path

def get_bag_file_path(dirname, ycb_item, number):
    bag_file_path = f"{dirname}/../../bags/odin/{ycb_item}{number}.bag"

    return bag_file_path

def parse_gt_to_file(dirname, ycb_item, number):
    bag_file_path = get_bag_file_path(dirname, ycb_item, number)
    print(f"Looking at bag {bag_file_path}")

    bag = Reader(bag_file_path)
    with bag as b:
        gt_traj = file_interface.read_bag_trajectory(b, '/vicon/ZED2_ZQ/ZED2_ZQ')

    save_path = get_save_path(dirname, ycb_item, number)
    gt_tum_filename = f"{save_path}/cam_gt_raw.txt"
    file_interface.write_tum_trajectory_file(gt_tum_filename, gt_traj)
    print(f"wrote TUM traj to {gt_tum_filename}")

if __name__ == "__main__":
    # ycb_item, number = parse_inputs()
    ycb_items = ["cracker", "sugar", "spam"]
    numbers = [1, 2, 3, 4]
    dirname = get_dirname()
    for ycb_item in ycb_items:
        for number in numbers:
            parse_gt_to_file(dirname, ycb_item, number)