### Introduction

We adapted the YCB-Video dataset toolbox to our pipeline. The intructions are at the end.
This is the toolbox for [The YCB-Video dataset](https://rse-lab.cs.washington.edu/projects/posecnn/) introduced for 6D object pose estimation.
It provides accurate 6D poses of 21 objects from the YCB dataset observed in 92 videos with 133,827 frames.

### License

The YCB-Video dataset is released under the MIT License (refer to the LICENSE file for details).

### Citing

If you find our dataset useful in your research, please consider citing:

	@article{xiang2017posecnn,
	author    = {Xiang, Yu and Schmidt, Tanner and Narayanan, Venkatraman and Fox, Dieter},
	title     = {PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes},
	journal   = {arXiv preprint arXiv:1711.00199},
	year      = {2017}
	}

### Annotation format
The *-meta.mat file in the YCB-Video dataset contains the following fields:
- center: 2D location of the projection of the 3D model origin in the image
- cls_indexes: class labels of the objects
- factor_depth: divde the depth image by this factor to get the actual depth vaule
- intrinsic_matrix: camera intrinsics
- poses: 6D poses of objects in the image
- rotation_translation_matrix: RT of the camera motion in 3D
- vertmap: coordinates in the 3D model space of each pixel in the image

### Usage

1. Download the YCB-Video dataset from [here](https://rse-lab.cs.washington.edu/projects/posecnn/).

2. Set your path of the YCB-Video dataset in [globals.m](https://github.com/yuxng/YCB_Video_toolbox/blob/master/globals.m) (required).

3. [show_pose_annotations.m](https://github.com/yuxng/YCB_Video_toolbox/blob/master/show_pose_annotations.m) displays the overlays of 3D shapes onto images according to our annotations. Check the code of this function to understand the annotation format.

4. [show_pose_results.m](https://github.com/yuxng/YCB_Video_toolbox/blob/master/show_pose_results.m) displays the 6D pose estimation results from PoseCNN. Unzip [results_PoseCNN.zip](https://github.com/yuxng/YCB_Video_toolbox/blob/master/results_PoseCNN.zip) before calling the function.

5. [evaluate_poses_stereo.m](https://github.com/yuxng/YCB_Video_toolbox/blob/master/evaluate_poses_stereo.m) evaluates our results on the stereo pairs. Check the code of this function to understand the evaluation metric.

6. [evaluate_poses_keyframe.m](https://github.com/yuxng/YCB_Video_toolbox/blob/master/evaluate_poses_keyframe.m) evaluates our results on the keyframes.

7. [plot_accuracy_stereo.m](https://github.com/yuxng/YCB_Video_toolbox/blob/master/plot_accuracy_stereo.m) plots all the accuracy-threshold curves from the stereo pairs.

8. [plot_accuracy_keyframe.m](https://github.com/yuxng/YCB_Video_toolbox/blob/master/plot_accuracy_keyframe.m) plots all the accuracy-threshold curves from the keyframes.

### Usage for our work based on DOPE

1. Set the path to YCB-Video dataset in globals.m

2. Set the inputs (results directory and object class index) in dope_evaluate_poses_keyframe.m and run.

3. If want to run results from multiple models at the same time, run dope_evaluate_poses_keyframe_batch.m instead.

4. Run dope_plot_accuracy_keyframe.m and view plots in /plots.

5. If want to plot results from multiple models in one plot, set the legends and index_plot in dope_plot_accuracy_keyframe.m and run.
