# SLAM-Super-6D
SLAM-Supported Semi-Supervised Learning for 6D Object Pose Estimation

Check out our [paper](https://arxiv.org/pdf/2203.04424.pdf)!

TLDR: We exploit robust pose graph optimization results to pseudo-label robot-collected RGB images and fine-tune 6D object pose estimators during object-based navigation.

![Method Overview](media/figure1.png)

The two most important features of this work

- A SLAM-aided self-training procedure for 6D object pose estimation.

- Automatic covariance tuning (ACT), a robust pose graph optimization method, enabling flexible uncertainty modeling for learning-based measurements.

## YCB video experiment

### Comparison of robust pose graph optimization methods

<table>
<tr>
<td colspan=1>
     Pseudo label accuracy on YCB video sequences
</td>

|003_cracker_box | 0001     | 0004     | 0007    | 0016     | 0017     | 0019     | 0025     | #best |
|---             |---       |---       |---      |---       |---       |---       |---       |---    |
|LM              | 62.3     | 58.7     | 13.2    | 69.4     | 37.6     | 110.1    | 101.6    | 0     |
|Cauchy          | 12.4     | **10.8** | 10.2    | 13.8     | 29.5     | 94.4     | 171.4    | 4     |
|Huber           | 31.4     | 25.4     | 10.2    | 34.2     | 21.6     | 52.5     | 57.0     | 1     |
|Geman-McClure   | **11.5** | 168.4    | 10.2    | 115.0    | 48.4     | 94.4     | 171.4    | 2     |
|cDCE            | 28.7     | 25.4     | 10.5    | 32.5     | 21.1     | **45.2** | 58.9     | 4     |
|ACT(Ours)       | 15.7     | 12.0     | **9.4** | **12.6** | **20.3** | 52.0     | **15.4** | **9** |
</tr>

### Pose estimation results on test sequences

<p align="middle">
    <img src="media/YCB-v-test.gif" width="400">
    <img src="media/YCB-v-test-slam.gif" width="400">
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
