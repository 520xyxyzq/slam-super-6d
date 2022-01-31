# YCB video experiment
The automated process and results of running RGBD ORB-SLAM3 (in `odom`) and NVIDIA DOPE (in `dets`) on YCB videos

## Dependencies
- [evo](https://github.com/MichaelGrupp/evo)
- [OpenCV](https://github.com/opencv/opencv)
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) (optional, if you want to rerun the experiment)
- [DOPE](https://github.com/NVlabs/Deep_Object_Pose) (optional, if you want to rerun the experiment)

## How to run the experiments
- Run ORB-SLAM3 on YCB videos
`./ycb_orbslam.sh $path/to/test/folder $path/to/ycb-v/data $path/to/ORB_SLAM3`
(This process could take ~6 hours)

- Note: We disabled ORBSLAM's loop closing module following the instructions of this
[issue](https://github.com/raulmur/ORB_SLAM2/issues/256)

- Note: The ORB-SLAM family has a repeatability [issue](https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/71)

- Run NVIDIA DOPE on YCB videos
`./ycb_dope.sh $path/to/test/folder $path/to/dope/folder`
(This process could take ~2 hours)

## Results
- ORB-SLAM3:
    - <ycb_seq_id>_keyframe.txt: Keyframes
    - <ycb_seq_id>.txt: The entire trajectory
    - Coordinate system: camera convention (z-out, y-right, x-down)

- NVIDIA DOPE:
    - <ycb_seq_id>_ycb_poses: object pose detection at all times in a YCB sequence (all 0's if no detection)
    - Coordinate system: camera convention (z-out, y-right, x-down)

## Files
- _ycb_original.json:
    - exported_object_classes: YCB-V object class names
    - fixed_model_transform: Transpose of obj transform to align and center original YCB object coordinate; Note that the unit is [cm]; it is the transformation of the original obj frame wrt the new frame.
    - cuboid_dimensions: dimensions of the object's 3D bounding box (x, y, z in the new centered obj frame)
    - seqs: YCB seqs in which the object shows up
