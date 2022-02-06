#!/usr/bin/env python2.7

from __future__ import print_function
import rospkg
import sys
import os
import cv2
import apriltag

def parse_inputs(argv):
    if not len(sys.argv) == 3:
        print("Needs to pass in arg for YCB object and dataset number, received now {} args".format(len(sys.argv)))
        sys.exit(-1)

    real_path = os.path.realpath(__file__)
    dirname = os.path.dirname(real_path)
    ycb_item = sys.argv[1]

    if not ycb_item in ["cracker", "sugar", "spam"]:
        print("YCB item needs to be one of 'cracker', 'sugar' or 'spam', is now {}".format(ycb_item)) 
        sys.exit(-1)

    try:
        number = int(argv[2])
    except ValueError:
        print("number passed in as arg 2 is not valid int, passed in {}".format(argv[2]))
        sys.exit(-1)

    if not number in [1, 2, 3, 4]:
        print("number must be one of 1, 2, 3, 4, is now {}".format(number))
        sys.exit(-1)

    print("Doing YCB item '{}', dataset number {}, in directory {}".format(ycb_item, number, dirname))

    return ycb_item, number, dirname


if __name__ == "__main__":
    ycb_item, number, dirname = parse_inputs(sys.argv)
    if ycb_item == "cracker":
        image_folder = "003_cracker_box_16k"
    elif ycb_item == "sugar":
        image_folder = "004_sugar_box_16k"
    elif ycb_item == "spam":
        image_folder = "010_potted_meat_can_16k"

    image_path = "{}/../../{}/00{}/left".format(dirname, image_folder, number)

    # image = cv2.imread(image_path + "/00000.png")
    # print(dir(image))
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # options = apriltag.DetectorOptions(families="tag36h11")
    # detector = apriltag.Detector(options)
    # results = detector.detect(gray)

    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    for image_file in os.listdir(image_path):
        if image_file.endswith(".png"):
            image_file = image_path + "/" + image_file
            image = cv2.imread(image_file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            results = detector.detect(gray)
            print("[INFO] {} total AprilTags detected\nresults: {}".format(len(results), results))