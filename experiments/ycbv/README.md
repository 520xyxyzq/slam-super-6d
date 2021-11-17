# YCB video experiment
The automated process and results of running RGBD ORB-SLAM3 (in `odom`) and NVIDIA DOPE (in `dets`) on YCB videos

## Dependencies
- [evo](https://github.com/MichaelGrupp/evo)
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) (optional, if you want to rerun the experiment)
- [DOPE](https://github.com/NVlabs/Deep_Object_Pose) (optional, if you want to rerun the experiment)

## How to run the experiments
- Run ORB-SLAM3 on YCB videos
`./ycb_orbslam.sh $path/to/test/folder $path/to/ycb-v/data $path/to/ORB_SLAM3`
(This process could take ~6 hours)

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
