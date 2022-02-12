import os
import sys

def get_dirname():
    real_path = os.path.realpath(__file__)
    dirname = os.path.dirname(real_path)

    return dirname


def get_save_path(dirname, ycb_item, number):

    if ycb_item == "cracker":
        image_folder = "003_cracker_box_16k"
    elif ycb_item == "sugar":
        image_folder = "004_sugar_box_16k"
    elif ycb_item == "spam":
        image_folder = "010_potted_meat_can_16k"

    save_path = "{}/../{}/00{}".format(dirname, image_folder, number)

    return save_path


def parse_inputs():
    if not len(sys.argv) == 3:
        print("Needs to pass in arg for YCB object and dataset number, received now {} args".format(len(sys.argv)))
        sys.exit(-1)

    ycb_item = sys.argv[1]

    if not ycb_item in ["cracker", "sugar", "spam"]:
        print("YCB item needs to be one of 'cracker', 'sugar' or 'spam', is now {}".format(ycb_item)) 
        sys.exit(-1)

    try:
        number = int(sys.argv[2])
    except ValueError:
        print("number passed in as arg 2 is not valid int, passed in {}".format(sys.argv[2]))
        sys.exit(-1)

    if not number in [1, 2, 3, 4]:
        print("number must be one of 1, 2, 3, 4, is now {}".format(number))
        sys.exit(-1)

    print("Doing YCB item '{}' and dataset number {}".format(ycb_item, number))

    return ycb_item, number


def get_save_path(dirname, ycb_item, number):
    if ycb_item == "cracker":
        image_folder = "003_cracker_box_16k"
    elif ycb_item == "sugar":
        image_folder = "004_sugar_box_16k"
    elif ycb_item == "spam":
        image_folder = "010_potted_meat_can_16k"

    save_path = "{}/../../{}/00{}".format(dirname, image_folder, number)

    return save_path