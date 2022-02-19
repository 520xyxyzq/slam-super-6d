#!/usr/bin/env python3
# Plot pseudo label errors with stds
# Ziqi Lu ziqilu@mit.edu

import matplotlib.pyplot as plt
import numpy as np

means = [
    [12.7119, 13.1909, 43.3475],
    [11.1388, 11.0377, 18.2813],
    [15.8889, 15.781, 19.6753]
]
stds = [
    [11.8592, 12.7612, 304.1944],
    [7.672, 7.5219, 1186.0902],
    [7.7169, 7.4224, 17.2674]
]
labels = ("Hybrid", "Inlier", "PoseEval")
objs = ["003_cracker_box", "004_sugar_box", "010_potted_meat_can"]
# NOTE: change this index for different objects!!
ind = 0

y_pos = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(10, 3))
plt.rcParams.update({'font.family': "Times New Roman"})
h = ax.barh(
    y_pos, means[ind], xerr=stds[ind], height=0.8, align='center',
    color=[[1, 0.5, 0.5, 1], [0.5, 1, 0.5, 1], [85/255, 153/255, 1, 0.5]],
    ecolor='black', capsize=10
)

csfont = {'fontname': 'Times New Roman'}
ax.set_title(objs[ind], **csfont)
ax.set_xlim(0, 50)
ax.axes.yaxis.set_visible(False)
ax.title.set_fontsize(25)
if ind != len(objs) - 1:
    ax.axes.xaxis.set_visible(False)
else:
    # ax.set_xlabel(
    #     "Average errors in pseudo-labels (pixels)", fontsize=20, **csfont
    # )
    plt.xticks(fontsize=20, **csfont)
plt.gca().invert_yaxis()
plt.legend(h, labels, fontsize=20)


# Save the figure and show
# plt.tight_layout()
plt.show()
