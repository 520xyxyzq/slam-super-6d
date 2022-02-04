#!/usr/bin/env python2.7

from scipy.spatial.transform import Rotation as Rot
import rospkg
import numpy as np


if __name__ == "__main__":
    svo_path = rospkg.RosPack().get_path("svo_ros")
    filename = svo_path + "/log/poses_tum"
    data = np.loadtxt(filename + ".txt")

    # TUM format: timestamp x y z q_x q_y q_z q_w
    positions = data[:, 1:4]
    quats = Rot.from_quat(data[:, (7, 4, 5, 6)]) # We want real part first

    Ry = Rot.from_euler('y', -np.pi/2)
    Rx = Rot.from_euler('x', np.pi/2)

    R = Ry * Rx

    correct_quats = quats * R # Needs to double check the order
    correct_positions = np.matmul(positions, R.as_dcm().T)
    new_data = np.vstack((
        data[:,0],
        correct_positions.T,
        correct_quats.as_quat()[:, (3, 0, 1, 2)].T
    )).T

    np.savetxt(filename + "_corrected.txt", new_data)