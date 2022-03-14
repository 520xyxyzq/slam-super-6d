# Computing ground truth from apriltags
The automated process and results of computing ground truth of object poses in the Jackal experiments.

## How to run the experiments
- Install all dependencies listed in `package.xml`.
- `catkin_make`
- Run `roslaunch apriltag_truth main.launch bag_name:=$BAG`
- Run `./runner.sh` to run the code for all bags.

## Results
- Object poses in TUM format `output/filtered/*_gt.txt`
    - `timestamp tx ty tz qx qy qz qw`

## Visualizing results
- Displays the video with ground truth object pose axis projected.
- Axes order is BGR.
- Run `roslaunch apriltag_truth video.launch bag_name:=$BAG`
