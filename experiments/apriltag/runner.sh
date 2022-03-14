#!/bin/bash
# Script to run code on all bags
declare -a bags=("cracker_1" "cracker_2" "cracker_3" "cracker_4" "sugar_1" "sugar_2" "sugar_3" "sugar_4" "spam_1" "spam_2" "spam_3" "spam_4")

for val in ${bags[@]}; do
    #roslaunch apriltag_truth main.launch bag_name:=$val > logs/$val.log
    echo $val
    roslaunch apriltag_truth main.launch bag_name:=$val
done
