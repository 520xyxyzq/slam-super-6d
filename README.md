# SLAM-Super-6D
SLAM-Supported Semi-Supervised Learning for 6D Object Pose Estimation

Check out our [paper](https://arxiv.org/pdf/2203.04424.pdf)!

TLDR: We exploit robust pose graph optimization results to pseudo-label robot-collected RGB images and fine-tune 6D object pose estimators during object-based navigation.

![Method Overview](https://drive.google.com/uc?export=view&id=1sdTURiSuy8IlMDEZNW4DhoKklaWhk4pY)

The two most important features of this work

- A SLAM-aided self-training procedure for 6D object pose estimation.

- Automatic covariance tuning (ACT), a robust pose graph optimization method, enabling flexible uncertainty modeling for learning-based measurements.

## YCB video experiment

We combine [DOPE](https://github.com/NVlabs/Deep_Object_Pose) pose predictions with [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) camera odometry to build per-video object-level maps.
We exploit the state estimates to pseudo-label YCB-v training images and fine-tune the DOPE estimator for selected YCB objects.

### Comparison of robust pose graph optimization methods for pseudo-labeling

|003_cracker_box |0001    |0004     |0007   |0016    |0017    |0019    |0025    |0029    |0035   |0041    |0044    |0045   |0050    |0054    |0059    |0066    |0070    |0074    |0082    |0085    |#best   |
|---             |:-:     |:-:      |:-:    |:-:     |:-:     |:-:     |:-:     |:-:     |:-:    |:-:     |:-:     |:-:    |:-:     |:-:     |:-:     |:-:     |:-:     |:-:     |:-:     |:-:     |:-:     |
|LM              |62.3    |58.7     |13.2   |69.4    |37.6    |110.1   |101.6   |86.1    |9.6    |21.5    |79.4    |140.0      |46.8    |152.7   |152.6    |79.0    |117.3   |139.5   |250.3    |183.2    |0       |
|Cauchy          |12.4    |**10.8** |10.2   |13.8    |29.5    |94.4    |171.4   |179.9   |**6.6**|9.6     |133.3   |169.9      |16.3    |**13.9**|131.7    |23.8    |**25.5**|102.6   |267.3    |137.5    |4     |
|Huber           |31.4    |25.4     |10.2   |34.2    |21.6    |52.5    |57.0    |72.4    |8.0    |11.6    |38.7    |**127.8**  |21.2    |68.0    |74.8     |46.8    |39.1    |68.7    |267.3    |153.4    |1     |
|Geman-McClure   |**11.5**|168.4    |10.2   |115.0   |48.4    |94.4    |171.4   |179.9   |6.6    |**9.5** |133.3   |169.9      |46.8    |77.1    |131.7    |182.7   |27.0    |102.6   |267.3    |137.5    |2     |
|cDCE            |28.7    |25.4     |10.5   |32.5    |21.1    |**45.2**|58.9    |**70.5**|7.8    |11.5    |38.0    |128.1      |24.4    |56.7    |55.6     |41.3    |37.7    |60.9    |**238.7**|**114.2**|4     |
|ACT (Ours)      |15.7    |12.0     |**9.4**|**12.6**|**20.3**|52.0    |**15.4**|238.3   |6.9    |10.8    |**28.8**|135.4      |**10.4**|18.7    |**21.3** |**22.3**|26.1    |**34.6**|795.2    |180.0    |**9**|


### Pose estimation and SLAM results on test sequences

<p align="middle">
    <img src="media/YCB-v-test.gif" width="400">
    <img src="media/YCB-v-test-slam.gif" width="400">
    <!-- <img src="https://drive.google.com/uc?export=view&id=1hCG_yahIi0OuEeuma-M1SOw3QnW9C-DA" width="400"> -->
    <!-- <img src="https://drive.google.com/uc?export=view&id=164MzFQubQy-YUjDYhjq-R3KpF7Aor86J" width="400"> -->
</p>

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

For modules in the `thirdparty` directory, we will not do any linting. If you are adding code in a `thirdparty` directory, you should commit with:
```
git commit --no-verify -m "My commit message here...."
```
