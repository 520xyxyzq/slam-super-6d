# YCB video experiment
We combine [DOPE](https://github.com/NVlabs/Deep_Object_Pose) pose predictions with [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) camera odometry to build per-video object-level world representations.
We leverage the consistent state estimates to pseudo-label YCB-v training images and fine-tune the DOPE estimator for selected YCB objects.

## How to run the experiments
- [Odometry](odom/): Run RGBD ORB-SLAM3 on YCB videos (~6 hours for 92 seqs)

    ```
    cd odom
    ./ycb_orbslam.sh ./ $path/to/ycb-v/data $path/to/ORB_SLAM3
    ```
    - We disabled ORBSLAM's loop closing module following the instructions of this [issue](https://github.com/raulmur/ORB_SLAM2/issues/256)

    - The ORB-SLAM family has a repeatability [issue](https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/71)

- [Inference](inference/): Run NVIDIA DOPE on YCB videos
    ```
    ./inference/infer.sh
    ```
    - [DOPE estimator models](https://drive.google.com/drive/folders/1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg) trained from synthetic data.
    - Remember to change the [config files](inference/config_inference/) and the seq ids in [infer.sh](inference/infer.sh)

- Pseudo-labeling: Generate pseudo-labeled data:
    ```
    ./generate_labels <obj_name> <inference_result_dir> <labeling_mode> <label_output_dir> <ycbv_data_dir>
    ./generate_data <obj_class_name> <label_output_dir> *obj0.txt 1 <ycbv_data_dir> <data_save_dir>
    ```
    - Play with the parameters (in [pseudo_labeler.py](../../src/pseudo_labeler.py)) to get better performance, e.g. optimizer, labeling mode, etc.

- [Training](train2/): Train or fine-tune the DOPE estimator
    - TODO: change training script to load data directly from ycbv data folder using label files.

- [Evaluation](evaluation/): Evaluate the pose estimators using metric ADD and ADD-S

- Data Generation: Generate synthetic data using [NVISII](https://github.com/owl-project/NVISII).


## Results and Files

We save odometry and object pose files following the [TUM](https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats) format. **Note that DOPE uses a different coordinate system** for YCB objects, see [here](https://research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/readme_0.txt) for details.

- Odometry:
    - odom/results/<ycb_seq_id>.txt: The camera odometry at each time stamp
    - odom/ground_truth/<ycb_seq_id>.txt: The ground truth camera trajectory
    - Coordinate system: camera convention (z-out, y-right, x-down)

- Inference:
    - inference/<obj_name>/<training_data>/<ycb_seq_id>.txt: Object pose prediction at each time stamp in a YCB sequence
    - inference/<obj_name>/ground_truth/<ycb_seq_id>_ycb_gt.txt: Ground truth object poses
    - Coordinate system: camera convention (z-out, y-right, x-down)
    - Missed predictions are saved as all-zeros lines

- Pseudo labels:
    - pseudo_labels/<obj_name>/<labeling_mode>/<ycb_seq_id>.txt: Pseudo ground truth object poses

- _ycb_original.json:
    - exported_object_classes: YCB-V object class names
    - fixed_model_transform: Transpose of obj transform to align and center original YCB object coordinate; Note that the unit is [cm]; it is the transformation of the original obj frame wrt the new frame.
    - cuboid_dimensions: dimensions of the object's 3D bounding box (x, y, z in the new centered obj frame)
    - seqs: YCB seqs in which the object shows up
