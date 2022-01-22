# Inference out of ROS
# Where you stored the images
# WARNING: The folder will be emptied at the beginning
img_folder=${1:-~/Desktop/data}
# Where you stored the YCB-V data
ycb_folder=${2:-/media/ziqi/LENOVO_USB_HDD/data/YCB-V/data}
# Where you want to save the results
out_folder=${3:-~/Desktop/train/}

# Get the folder where this bash script is saved
current_folder=$(dirname $(realpath $0))

# Remove all the image files (if exist)
if ls -1qA $img_folder/ | grep -q .
    then rm $img_folder/*-color.png
fi

for n in "2" "5" "8" "14" "17" "23" "26" "29" "34" "39" "43" "47" "49" "53" "59" "60" "61" "73" "77" "87" # spam
#"1" "4" "7" "16" "17" "19" "25" "29" "35" "41" "44" "45" "50" "54" "59" "66" "70" "74" "82" "85" # cracker
#"1" "14" "15" "20" "25" "29" "33" "36" "37" "43" "49" "51" "54" "55" "58" "60" "74" "77" "85" "89" # sugar
do
    # Convert seq number to ycb id
    printf -v seq "%04g" $n
    # Copy imgs over to test folder
    cp $ycb_folder/$seq/*-color.png $img_folder/
    # YCB seqs 0~59 and 60~91 used different cameras
    if (($n<60))
        then config_file=$current_folder/config_inference/camera_info.yaml
    else
        config_file=$current_folder/config_inference/camera_info_ycb2.yaml
    fi
    # Pose estimation
    python3 $current_folder/inference.py --outf $out_folder --data $img_folder \
    --camera $config_file --seq $n
    # Remove all the image files
    rm $img_folder/*-color.png
done
