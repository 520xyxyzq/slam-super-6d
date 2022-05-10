# Computing ground truth from apriltags
The automated process and results of computing ground truth of object poses in the Jackal experiments.

## How to run the experiments
- Install all dependencies listed in `package.xml`.
- `catkin_make`
- Default location for bags is `./apriltag_truth/bags`, but can be configured with launch arg `bag_full_path`
- Run `roslaunch apriltag_truth main.launch bag_name:=$BAG obj_name:=$OBJ`
- Configure different objects using `./apriltag_truth/config/object_settings.yaml`
- Example with `cracker_1.bag`: `roslaunch apriltag_truth main.launch bag_name:=cracker_1 obj_name:=cracker`

## Results
- Object poses in TUM format `output/filtered/*_gt.txt`
    - `timestamp tx ty tz qx qy qz qw`

## Visualizing results
- Displays the video with ground truth object pose axis projected.
- Run `roslaunch apriltag_truth video.launch bag_name:=$BAG`
