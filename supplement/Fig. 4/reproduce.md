# Fig. 5: Object SLAM results evaluation
## Seq 0043

matplotlib params:
```
csfont={'fontname':'Times New Roman'}
axes.text(0.25, 0, 0.5, "004_sugar_box", fontsize=15, **csfont)
axes.text(0.25, 0, 0.95, "010_potted_meat_can", fontsize=15, **csfont)
axes.view_init(azim=-87, elev=-125)
axes.set_axis_off()
plt.rcParams.update({'font.family': "Times New Roman"})
plt.legend(fontsize=15, loc=(0.2, 0.7))
plt.tight_layout()
```

CML:
```
python3 ../../src/pseudo_labeler.py --odom odom/results/0059.txt --out ~/Desktop/ --gt_cam odom/ground_truth/0059.txt --dets inference/010_potted_meat_can_16k/Hybrid_1/0059.txt inference/003_cracker_box_16k/Hybrid_90_1/0059.txt --gt_obj dets/ground_truth/010_potted_meat_can_16k/0059_ycb_gt.txt dets/ground_truth/003_cracker_box_16k/0059_ycb_gt.txt -k 0 -op 1 -j -l 1 -m 0 -p
```

## Seq 0059

matplotlib params:
```
csfont={'fontname':'Times New Roman'}
axes.text(-0.3, 0, 0.3, "003_cracker_box", fontsize=15, **csfont)
axes.text(-0.3, 0, 1.3, "010_potted_meat_can", fontsize=15, **csfont)
axes.view_init(azim=-87, elev=-125)
axes.set_axis_off()
plt.rcParams.update({'font.family': "Times New Roman"})
plt.legend(fontsize=15, loc=(0.35, 0.65))
plt.tight_layout()
```

CML:
```
python3 ../../src/pseudo_labeler.py --odom odom/results/0059.txt --out ~/Desktop/ --gt_cam odom/ground_truth/0059.txt --dets inference/010_potted_meat_can_16k/Hybrid_1/0059.txt inference/003_cracker_box_16k/Hybrid_90_1/0059.txt --gt_obj dets/ground_truth/010_potted_meat_can_16k/0059_ycb_gt.txt dets/ground_truth/003_cracker_box_16k/0059_ycb_gt.txt -k 0 -op 1 -j -l 10 -m 0 -p
```

### Error:
```
evo_ape tum 0049(59)_before(after).txt ../experiments/ycbv/odom/ground truth/0049(59).txt --align_origin
```
