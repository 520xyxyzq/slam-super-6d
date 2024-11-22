#!/usr/bin/env python2
# Estimate ground truth poses from apriltag detections
# Ethan Yang ethany@mit.edu

import time

import numpy as np
import rospy
from apriltag_ros.msg import AprilTagDetectionArray
from scipy.spatial.transform import Rotation as Rot


class TruthEvaluate(object):
    def __init__(self):
        # all measurements in m
        self.tag_size = 0.05
        self.margin = 0.01
        self.paper_height = 279.4 * 10**-3
        self.paper_width = 215.9 * 10**-3

        rospy.loginfo("init TruthEvaluate")
        try:
            bag_name = rospy.get_param('bag_name', None) + '_gt.txt'
            rospy.loginfo('Bag_name: ' + bag_name)
            obj_type = rospy.get_param('obj_name', None)
            rospy.loginfo('Obj_name: ' + obj_type)
            self.tags = np.array(rospy.get_param('tags', {})[obj_type])
            rospy.loginfo('Obj tags: ' + str(self.tags))
            self.obj_dim = np.array(rospy.get_param(
                'obj_dim', {})[obj_type]) * 10**-2
            rospy.loginfo('Obj dim (mm): ' + str(self.obj_dim))
        except Exception:
            raise RuntimeError("""Error loading parameters.
                    Check the config folder and the names.""")

        # apriltag coord system has x right and y up
        # with x-y plane on the paper
        center_dist = self.margin+self.tag_size/2
        x_coords = [center_dist, self.paper_width - center_dist]
        y_coords = [center_dist, self.paper_height/2,
                    self.paper_height - center_dist]
        apriltag_coords = np.array(
            [[x_coords[i], y_coords[j], 0]
                for j in range(3) for i in range(2)])

        # Assume that object is centered on the paper
        # z of obj is just half height of obj
        self.obj_center = np.array(
            [self.paper_width/2, self.paper_height/2, self.obj_dim[1]/2])
        # Relative translations of test object CENTER wrt tags
        self.object2tags = np.array(
            [self.obj_center - apriltag_coords[i, :]
                for i in range(len(self.tags))])

        """
        rospy.loginfo('coords')
        rospy.loginfo(apriltag_coords)
        rospy.loginfo('objcenter')
        rospy.loginfo(self.obj_center)
        rospy.loginfo('obj2tags')
        rospy.loginfo(self.object2tags)
        """

        # Subscribe to tags' id-poses
        rospy.Subscriber("/tag_detections",
                         AprilTagDetectionArray, self.tag_det_cb)

        # Set up logging files
        self.output_filtered_file = open(
            rospy.get_param('filtered_output_path')+bag_name, "w")

        # Uncomment for visualizing transform in Rviz
        # self.pub = tf.TransformBroadcaster()

        # Variables to track when detections stop
        self.got_tags = False
        self.last_time_got = time.time()

    def tag_det_cb(self, msg):
        self.last_time_got = time.time()
        self.got_tags = True

        time_stamp = msg.header.stamp
        dets = msg.detections  # detections msg
        obj_trans = []
        obj_quats = []
        # Publish apriltag pose
        """
        if dets:
            trans = dets[0].pose.pose.pose.position
            orien  = dets[0].pose.pose.pose.orientation
            self.pub.sendTransform((trans.x, trans.y, trans.z),
                    (orien.x, orien.y, orien.z, orien.w),
                    dets[0].pose.header.stamp, 'tag', 'camera')
        """

        for ii in range(len(dets)):
            tag_quat = np.array([dets[ii].pose.pose.pose.orientation.x,
                                 dets[ii].pose.pose.pose.orientation.y,
                                 dets[ii].pose.pose.pose.orientation.z,
                                 dets[ii].pose.pose.pose.orientation.w])
            obj_quat = self.rot_trans(tag_quat)
            euler_angles = Rot.from_quat(
                obj_quat).as_euler('zxy', degrees=True)
            # Filter out clear outliers (hardcoded)
            if abs(euler_angles[0]) >= 80:
                continue

            obj_quats.append(obj_quat)
            obj2tag = self.object2tags[self.tags == dets[ii].id[0]].ravel()
            obj_trans.append(np.array([dets[ii].pose.pose.pose.position.x,
                                       dets[ii].pose.pose.pose.position.y,
                                       dets[ii].pose.pose.pose.position.z])
                             + Rot.from_quat(tag_quat).as_dcm()
                             .dot(obj2tag))

        obj_trans = np.array(obj_trans)
        obj_quats = np.array(obj_quats)
        self.obj_trans = [0, 0, 0]
        self.obj_quat = [0, 0, 0, 0]
        if len(obj_quats) != 0:
            self.obj_trans = np.mean(obj_trans, axis=0)
            sum_rot_mat = np.zeros((3, 3))
            # use SVD to average poses
            for quat in obj_quats:
                sum_rot_mat += Rot.from_quat(quat).as_dcm()
            u, d, vt = np.linalg.svd(sum_rot_mat)
            if np.linalg.det(np.matmul(u, vt)) >= 0:
                obj_rot = np.matmul(u, vt)
            else:
                obj_rot = np.matmul(np.matmul(u, np.array(
                    [[1, 0, 0], [0, 1, 0], [0, 0, -1]])), vt)
            self.obj_quat = Rot.from_dcm(obj_rot).as_quat()

            rospy.loginfo('self.obj_quat: ' + str(self.obj_quat))
            """
            # Publish object pose
            self.pub.sendTransform(self.obj_trans, self.obj_quat,
                    dets[0].pose.header.stamp, 'obj', 'camera')
            """

        outputs = ["{:.20f}".format(time_stamp.to_sec())]
        for i in self.obj_trans:
            outputs.append("{:.12f}".format(i))
        for i in self.obj_quat:
            outputs.append("{:.12f}".format(i))
        output_str = ' '.join(outputs)
        self.output_filtered_file.write(output_str + '\n')

    def rot_trans(self, quat_in):
        '''
        Util function to convert apriltag coordinate system to object's
        Input:
            quat_in (4-array)
        Output:
            quat_out (4-array)
        '''
        rotmat = Rot.from_quat(quat_in).as_dcm()
        # Obj x axis is mapped from apriltag y axis,
        # Obj z axis is mapped from negative apriltag x axis
        trans_matrix = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        return Rot.from_dcm(rotmat.dot(trans_matrix)).as_quat()

    def main(self):
        while not rospy.is_shutdown():
            if self.got_tags and time.time() - self.last_time_got > 5:
                # exit when no tag detections after 5 seconds
                rospy.loginfo('Finishing TruthEvaluate')
                rospy.signal_shutdown('No more detections, shutting down')
            continue


if __name__ == "__main__":
    rospy.init_node("truth_evaluate_node", anonymous=True)
    node = TruthEvaluate()
    try:
        node.main()
    except rospy.ROSInterruptException:
        pass
