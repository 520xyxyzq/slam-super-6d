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


def main(trajs, gt, save=False, out=None):
    """
    Read estimated camera trajs and ground truth, and plot
    @param trajs: [list of strs] camera trajectory files (.txt)
    @param gt: [str] camera ground truth traj file (.txt)
    @param save: [bool] Save the anim video?
    @param out: [str] Absolute file name to save the video
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

    # Read gt camera traj
    assert(os.path.isfile(gt)), "Error: %s not a file" % gt
    gt_traj = readTum(gt)

    colors = ["b", "r", "g", "y"]
    transforms = []
    for ii, est_traj in enumerate(est_trajs):
        # Associate trajectories using time stamps
        gt_traj, est_traj = sync.associate_trajectories(gt_traj, est_traj)
        rotmat, trans, scale = est_traj.align(gt_traj)
        transform = np.eye(4)
        transform[:3, :3] = rotmat
        transform[:3, 3] = trans
        transforms.append(gtsam.Pose3(transform))
        plot.traj(ax, plot.PlotMode.xyz, est_traj, "-", colors[ii])

    plot.traj(ax, plot.PlotMode.xyz, gt_traj, "--", "k")
    plt.legend(
        ["LM", "ACT", "Ground truth"],
        fontsize=18, facecolor="white", loc="upper left"
    )
    # Static transform: original to DOPE coordinate transformation
    static_transform = gtsam.Pose3(
        gtsam.Rot3.Quaternion(0.5, 0.5, -0.5, 0.5), gtsam.Point3(0, 0, 0)
    )
    # TODO: make these poses params, hard coded for now
    # Poses to plot
    poses = [
        gtsam.Pose3(
            # gtsam.Rot3.Quaternion(
            #     -0.38276518, -0.38718693, 0.80789133, 0.22558523
            # ),
            # gtsam.Point3(
            #     -0.2681047, -0.33712401, 0.66198888
            # )
            # gtsam.Rot3.Quaternion(
            #     0.60055857, 0.39775486, -0.66538435, -0.19591871
            # ),
            # gtsam.Point3(
            #     -0.12029817, -0.08271626, 0.56962272
            # )
            gtsam.Rot3.Quaternion(
                0.677064, 0.30900178, -0.60665223, -0.2794196
            ),
            gtsam.Point3(
                -0.00526817, -0.02934786, 0.60167321
            )
        ),
        gtsam.Pose3(
            # gtsam.Rot3.Quaternion(
            #     -0.41223828, -0.38008439, 0.79846393, 0.21920495
            # ),
            # gtsam.Point3(
            #     -0.24288919, -0.31745482, 0.65880243
            # )
            # gtsam.Rot3.Quaternion(
            #     0.60664291, 0.40313693, -0.65451343, -0.20267503
            # ),
            # gtsam.Point3(
            #     -0.12518198, -0.07105284, 0.5690265
            # )
            gtsam.Rot3.Quaternion(
                0.68118777, 0.31343135, -0.60262113, -0.27311496
            ),
            gtsam.Point3(
                0.00514104, -0.04108549, 0.6061079
            )
        )
    ]
    for ii, pose in enumerate(poses):
        gtsam_plot.plot_pose3(
            0,
            transforms[ii] * pose * static_transform
        )

    cam_pose = gtsam.Pose3(
        # gtsam.Rot3.Quaternion(0.97527451, -0.00756, 0.1391672, 0.1715),
        # gtsam.Point3(-0.3505, -0.088197, 0.0261)
        gtsam.Rot3.Quaternion(0.9999, -0.00029449, -0.0080, -0.010435),
        gtsam.Point3(-0.076166, -0.00347731, -0.0057497)

    )
    rel_obj_pose = gtsam.Pose3(
        # gtsam.Rot3.Quaternion(-0.36790, -0.1900, 0.8376372, 0.35622771),
        # gtsam.Point3(-0.193826, -0.26122, 0.7237367)
        gtsam.Rot3.Quaternion(0.6491, 0.342165, -0.64313, -0.2191),
        gtsam.Point3(0.025748344813, -0.062944505404, 0.672104875438)
    )
    gtsam_plot.plot_pose3(0, cam_pose * rel_obj_pose * static_transform)

    # Set view angles
    ax.view_init(azim=-80, elev=-143)
    # Set view distance
    ax.dist = 6.9
    ax.set_axis_off()
    ax.set_facecolor("white")
    plt.tight_layout()

    # Re-specify the axes (poses) line styles and colors
    lines = fig.get_children()[-1].lines
    # Re-format GT pose axes
    for line in lines[-3:]:
        line.set_color("k")
        line.set_linestyle("--")
    # Re-format 2nd axes
    for line in lines[-6:-3]:
        line.set_color("r")
    # Re-format 1st axes
    for line in lines[-9:-6]:
        line.set_color("b")

    plt.show()

    if save:
        def init():
            return fig,

        def ani(elv):
            # NOTE: tune the functions to get visually pleasing anim motions
            ax.view_init(azim=-80+elv, elev=-143+(abs(180-elv) - 180)/5)
            return fig,
        from matplotlib import animation
        anim = animation.FuncAnimation(
            fig, ani, init_func=init, frames=360, interval=1, blit=True
        )
        anim.save(out, fps=30, extra_args=['-vcodec', 'libx264'], dpi=300)

        # Uncomment to check the video before saving it
        # for elv in range(0, 360, 1):
        #     ax.view_init(azim=-80+elv, elev=-143+(abs(180-elv) - 180)/5)
        #     plt.draw()
        #     plt.pause(0.01)


if __name__ == "__main__":
    # Read command line args
    parser = argparse.ArgumentParser()
    jackal_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    root = os.path.dirname(os.path.dirname(jackal_dir))
    parser.add_argument(
        "--gt", "-g", type=str, help="Ground truth camera traj (.txt)",
        default=root+"/Supplement/Tab. V/010_potted_meat_can/002/cam_gt.txt"
    )
    parser.add_argument(
        "--traj", "-t", nargs="+", type=str,
        help="Traj estimate files (.txt)",
        default=[
            root + "/Supplement/Tab. V/" +
            "010_potted_meat_can/002/PGO_traj_before_LM.txt",
            root + "/Supplement/Tab. V/" +
            "010_potted_meat_can/002/PGO_traj_before.txt"
        ]
    )
    parser.add_argument(
        "--save", "-s", action="store_true",
        help="Want to save the anim video?"
    )
    parser.add_argument(
        "--out", "-o", help="The absolute video file name",
        default="/home/ziqi/Desktop/out.mp4"
    )
    args = parser.parse_args()

    main(args.traj, args.gt, args.save, args.out)
