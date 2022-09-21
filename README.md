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

### Testing the DOPE model before and after self-training on the YCB-v dataset
- Download a DOPE weights file from [here](https://drive.google.com/drive/folders/1fkMdr9Y8ls2EQTLtDmy8dULPHMEHWKKs?usp=sharing) to your favorite folder. (Initial: before self-training; Self-trained: after self-training; Supervised: after supervised training.)
- Change [this line](https://github.com/520xyxyzq/slam-super-6d/blob/645701adaf80a273c10adb863878d8f6af228f61/experiments/ycbv/inference/config_inference/config_pose.yaml#L11) to point to the weights file.
- Save the test images to `/path/to/image/folder/`.
- Run
```
cd /path/to/slam-super-6d
python3 experiments/ycbv/inference/inference.py --data /path/to/image/folder/ --outf /output/folder/
```
- Get the object pose preditions saved in the TUM format at `/output/folder/0000.txt`.
- Please check out the [DOPE](https://github.com/NVlabs/Deep_Object_Pose) Github repo for more [details](https://github.com/NVlabs/Deep_Object_Pose/tree/master/scripts/train2) on how to train/run DOPE networks.

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
