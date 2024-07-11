# SLAM-Super-6D
SLAM-Supported Semi-Supervised Learning for 6D Object Pose Estimation

Check out our [paper](https://arxiv.org/pdf/2203.04424.pdf)!

**TLDR**: We exploit robust pose graph optimization results to pseudo-label robot-collected RGB images and fine-tune 6D object pose estimators during object-based navigation.

![Method Overview](media/figure1.png)

The two most important features of this work

- A SLAM-aided self-training procedure for 6D object pose estimation.

- Automatic covariance tuning (ACT), a robust pose graph optimization method, enabling flexible uncertainty modeling for learning-based measurements.

## Results

We combine object pose prediction with camera odometry to infer object-level 3D scene geometry.
We leverage the consistent state estimates to pseudo-label training images and fine-tune the pose estimator.

### Per-frame object pose prediction before and after self-training

<p align="middle">
    <img src="media/YCB-v-test.gif" width="400">
    <img src="media/robot-test.gif" width="400">
</p>

## Installing

Create a [conda](https://www.anaconda.com/products/individual) environment to install the dependencies.

```
cd /path/to/slam-super-6d
conda env create -f environment.yml
```

### Test the DOPE model before and after self-training on the YCB-v dataset
- Download a DOPE weights file from [here](https://drive.google.com/drive/folders/0B20zFgsL_Sorfm12bzdqdk9LUVJVSVVvQUJ3RDBmRms2SEEwM0xBbTNxdUptUlF6RGdtdEU?resourcekey=0-tjXxtBBZfyvGaZycR8Y4VA&usp=sharing) to your favorite folder. (Initial: before self-training; Self-trained: after self-training; Supervised: after supervised training.)
- Change [this line](https://github.com/520xyxyzq/slam-super-6d/blob/645701adaf80a273c10adb863878d8f6af228f61/experiments/ycbv/inference/config_inference/config_pose.yaml#L11) to point to the weights file.
- Save the test images to `/path/to/image/folder/`.
- Run
```
cd /path/to/slam-super-6d
python3 experiments/ycbv/inference/inference.py --data /path/to/image/folder/ --outf /output/folder/
```
- Get the object pose preditions saved in the [TUM format](https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats) at `/output/folder/0000.txt`.
- Please check out the [DOPE](https://github.com/NVlabs/Deep_Object_Pose) Github repo for more [details](https://github.com/NVlabs/Deep_Object_Pose/tree/master/scripts/train2) on how to train/run DOPE networks.

### Test the pseudo labeling module
Given a sequence of unlabeled images, how to generate pseudo labels (pseudo ground truth poses)?
- Step 1: Choose your favorite pose estimator and camera odometry pipeline.
- Step 2: Predict the object poses in the images and save them to `${obj_1}.txt`, `${obj_2}.txt`, ..., `${obj_n}.txt` in [TUM format](https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats). (If the estimator failed for a certain frame, use `timestamp`+7 `0`s for the corresponding line.)
- Step 3: Estimate camera motion and save the noisy pose measurements to `${odom}.txt` in [TUM format](https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats).
- Step 4: Generate pseudo ground truth poses:
    - If the objects are from the [YCB video dataset](https://rse-lab.cs.washington.edu/projects/posecnn/), [download](https://drive.google.com/file/d/1LGH1N1F8BRDkym75Du02R6qvDat7_40T/view?usp=sharing) their [PoseRBPF](https://github.com/NVlabs/PoseRBPF) auto-encoder model weights and codebooks to [this](src/checkpoints/) and [this](src/codebooks/) folder, and use the _Hybrid_ mode for pseudo-labeling:
    ```
    cd /path/to/slam-super-6d
    python3 src/pseudo_labeler.py --joint --optim 1 --mode 2 --dets ${obj_1}.txt ${obj_2}.txt ... ${obj_n}.txt --odom ${odom}.txt --obj ${obj_1_name} ${obj_2_name} ... ${obj_n_name} --imgs "/path/to/unlabeled/images/*.png" --intrinsics ${fx} ${fy} ${cx} ${cy} ${s} --out ${output}
    ```
    - Otherwise, use the _Inlier_ labeling mode, which disables the rendered-to-real RoI comparison:
    ```
    cd /path/to/slam-super-6d
    python3 src/pseudo_labeler.py --joint --optim 1 --mode 1 --dets ${obj_1}.txt ${obj_2}.txt ... ${obj_n}.txt --odom ${odom}.txt --out ${output}
    ```
- Step 5: Get the pseudo ground truth poses at `${output}/obj1.txt`, `${output}/obj2.txt`, ..., `${output}/objn.txt` in [TUM format](https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats).

As an example, you should be able to get the [this](experiments/ycbv/pseudo_labels/010_potted_meat_can_16k/Hybrid_1/0002_obj0.txt) file (up to floating point discrepancies) if you run (_Hybrid_ pseudo-labeling):
```
cd /path/to/slam-super-6d
python3 src/pseudo_labeler.py --joint --optim 1 --mode 2 --dets ./experiments/ycbv/inference/010_potted_meat_can_16k/Initial/0002.txt --odom ./experiments/ycbv/odom/results/0002.txt --obj 010_potted_meat_can --imgs "/path/to/YCB-V/data/0002/*-color.png" --out /output/folder/
```

And get [this](experiments/ycbv/pseudo_labels/010_potted_meat_can_16k/Inlier_1/0002_obj0.txt) file if you run (_Inlier_ pseudo-labeling):
```
cd /path/to/slam-super-6d
python3 src/pseudo_labeler.py --joint --optim 1 --mode 1 --dets ./experiments/ycbv/inference/010_potted_meat_can_16k/Initial/0002.txt --odom ./experiments/ycbv/odom/results/0002.txt --out /output/folder/
```


### More detailed user guide coming soon...

## Developing

We're using [pre-commit](https://pre-commit.com/) for automatic linting. To install `pre-commit` run:
```
pip3 install pre-commit
```
You can verify your installation went through by running `pre-commit --version` and you should see something like `pre-commit 2.14.1`.

To get started using `pre-commit` with this codebase, from the project repo run:
```
pre-commit install
```
Now, each time you `git add` new files and try to `git commit` your code will automatically be run through a variety of linters. You won't be able to commit anything until the linters are happy with your code.

## Acknowledgement

Thanks to Jonathan Tremblay for suggestions on DOPE network training and synthetic data generation.
