#!/usr/bin/env python3
# Plot trajectories for the jackal experiments
# Ziqi Lu ziqilu@mit.edu

import argparse
import os

import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
from evo.core import sync
from evo.tools import file_interface, plot


def readTum(txt):
    """
    Read trajectory from a tum file
    @param txt: [str] txt file name
    @return traj: [evo.trajectory.PoseTrajectory3D] evo traj
    """
    traj = file_interface.read_tum_trajectory_file(txt)
    return traj


def main(trajs, gt, objs, out, save=False):
    """
    Read estimated camera trajs and ground truth, and plot
    @param trajs: [list of strs] camera trajectory files (.txt)
    @param gt: [str] camera ground truth traj file (.txt)
    @param objs: [list of str] obj ground truth files (.txt)
    @param out: [str] out video file name
    @param save: [bool] Save the video? only play the video if False
    """
    # Make figure
    fig = plt.figure(0)
    # ax = plot.prepare_axis(fig, plot.PlotMode.xyz)
    ax = fig.add_subplot(projection="3d")
    plt.rcParams.update({'font.family': "Times New Roman", 'font.size': "20"})
    # Read estimated camera trajs
    est_trajs = []
    for ii, traj in enumerate(trajs):
        assert(os.path.isfile(traj)), "Error: %s not a file" % traj
        est_trajs.append(readTum(traj))

    # Read ground truth object poses
    obj_poses = []
    for ii, obj in enumerate(objs):
        assert(os.path.isfile(obj)), "Error: %s not a file" % obj
        obj_poses.append(np.loadtxt(obj))

    # Read gt camera traj
    assert(os.path.isfile(gt)), "Error: %s not a file" % gt
    gt_traj = readTum(gt)

    colors = ["b", "r", "g", "y"]
    for ii, est_traj in enumerate(est_trajs):
        # Associate trajectories using time stamps
        plot.traj(ax, plot.PlotMode.xyz, est_traj, "-", colors[ii])

    gt_traj, est_traj0 = sync.associate_trajectories(gt_traj, est_trajs[0])
    gt_traj.align_origin(est_traj0)
    plot.traj(ax, plot.PlotMode.xyz, gt_traj, "--", "k")

    plt.legend(
        ["Robust SLAM estimates", "Ground truth"],
        fontsize=12, facecolor="white", loc=[0.0, 0.7]
    )
    # Static transform: original to DOPE coordinate transformation
    static_transform = gtsam.Pose3(
        gtsam.Rot3.Quaternion(0.5, 0.5, -0.5, 0.5), gtsam.Point3(0, 0, 0)
    )

    poses = []
    for ii, obj_pose in enumerate(obj_poses):
        if len(obj_pose.shape) == 1:
            pose = gtsam.Pose3(
                gtsam.Rot3.Quaternion(
                    obj_pose[-1], *(obj_pose[4:-1])
                ),
                gtsam.Point3(
                    *(obj_pose[1:4])
                )
            )
        else:
            pose = gtsam.Pose3(
                gtsam.Rot3.Quaternion(
                    obj_pose[0, -1], *(obj_pose[0, 4:-1])
                ),
                gtsam.Point3(
                    *(obj_pose[0, 1:4])
                )
            )
        poses.append(pose)

    # Poses to plot
    for ii, pose in enumerate(poses):
        gtsam_plot.plot_pose3(
            0, pose * static_transform
        )

    ax.set_axis_off()
    ax.set_facecolor("white")
    plt.tight_layout()
    lines = fig.get_children()[-1].lines
    for line in lines[-6:]:
        line.set_color("k")
        line.set_linestyle("--")

    for line in lines[-12:-6]:
        line.set_color("b")

    if save:
        def init():
            return fig,

        def ani(elv):
            ax.view_init(azim=-97+elv, elev=-116+(abs(180-elv) - 180)/3)
            ax.dist = 12
            return fig,

        from matplotlib import animation
        anim = animation.FuncAnimation(
            fig, ani, init_func=init, frames=360, interval=1, blit=True
        )
        anim.save(out, fps=30, extra_args=['-vcodec', 'libx264'], dpi=300)
    else:
        for elv in range(0, 360, 1):
            ax.view_init(azim=-97+elv, elev=-116+(abs(180-elv) - 180)/3)
            ax.dist = 12
            plt.draw()
            plt.pause(0.01)


if __name__ == "__main__":
    # Read command line args
    parser = argparse.ArgumentParser()
    root = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--gt", "-g", type=str, help="Ground truth camera traj (.txt)",
        default=os.path.dirname(os.path.dirname(root)) +
        "/experiments/ycbv/odom/ground_truth/0049.txt"
    )
    parser.add_argument(
        "--traj", "-t", nargs="+", type=str,
        help="Traj estimate files (.txt)",
        default=[
            # root + "/0049/0049_after_Gau.txt",
            root + "/0049/0049_before.txt"
        ]
    )
    parser.add_argument(
        "--obj", "-o", type=str, nargs="+", help="Object pose files",
        default=[
            # root + "/0049/0049_004_after_Gau.txt",
            # root + "/0049/0049_010_after_Gau.txt",
            root + "/0049/0049_004_before.txt",
            root + "/0049/0049_010_before.txt",
            os.path.dirname(os.path.dirname(root)) + "/experiments/ycbv/"
            "inference/004_sugar_box_16k/ground_truth/0049_ycb_gt.txt",
            os.path.dirname(os.path.dirname(root)) + "/experiments/ycbv/" +
            "inference/010_potted_meat_can_16k/ground_truth/0049_ycb_gt.txt"
        ]
    )
    parser.add_argument(
        "--out", type=str, help="Name of the output video",
        default="/home/ziqi/Desktop/out.mp4"
    )
    parser.add_argument(
        "--save", "-s", action="store_true", help="save the video?"
    )
    args = parser.parse_args()

    main(args.traj, args.gt, args.obj, args.out, args.save)
