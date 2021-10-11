# YCB video experiment
The automated process and results of running RGBD ORB-SLAM3 (in `odom`) and Nvidia DOPE (in `dets`) on YCB videos

## Dependencies
- [evo](https://github.com/MichaelGrupp/evo)
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) (optional, if you want to rerun the experiment)

## How to run the experiments
- Run ORB-SLAM3 on YCB videos  
`./ycb_orbslam.sh $path/to/test/folder $path/to/ycb-v/data $path/to/ORB_SLAM3`  
(This process could take ~6 hours)

- Run Nvidia DOPE on YCB videos  
`./ycb_dope.sh $path/to/test/folder $path/to/dope/folder`  
(This process could take ~2 hours)
