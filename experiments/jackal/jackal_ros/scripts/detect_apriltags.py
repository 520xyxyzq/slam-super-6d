import sys
import os
from unittest import result
import cv2
from pupil_apriltags import Detector
import numpy as np

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


def draw(img, origin, basis_vectors, K):
    img_origin = K@origin.ravel()
    img_origin = img_origin[:2]/img_origin[-1]
    corner = tuple(img_origin.astype(int))
    img_basis_vectors = K@basis_vectors.T
    img_basis_vectors = img_basis_vectors[:2]/img_basis_vectors[-1]
    img = cv2.line(img, corner, tuple(img_basis_vectors[:,0].astype(int)), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(img_basis_vectors[:,1].astype(int)), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(img_basis_vectors[:,2].astype(int)), (0,0,255), 5)
    return img


def project(K, X):
    """
    Computes the pinhole projection of a (3 or 4)xN array X using
    the camera intrinsic matrix K. Returns the pixel coordinates
    as an array of size 2xN.
    """
    if len(X.shape) == 3:
        uvw = K[None]@X[:,:3,:]
        uvw /= uvw[:,None,2,:]
        uv = uvw[:,:2,:]
        uv = np.vstack((uv[:,0,:].ravel(), uv[:,1,:].ravel()))
    else:
        uvw = K@X[:3,:]
        uvw /= uvw[2,:]
        uv = uvw[:2,:]
    return uv.astype(int)

def Rt2T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, -1] = t

    return T

def draw_frame(img, K, R, t, scale=1):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.
    Control the length of the axes by specifying the scale argument.
    """
    T = Rt2T(R, t)
    X = T @ np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]])
    u,v = project(K, X)

    # plt.plot([u[0], u[1]], [v[0], v[1]], color='red') # X-axis
    # plt.plot([u[0], u[2]], [v[0], v[2]], color='green') # Y-axis
    # plt.plot([u[0], u[3]], [v[0], v[3]], color='blue') # Z-axis
    img = cv2.line(img, (u[0], v[0]), (u[1], v[1]), (255,0,0), 5)
    img = cv2.line(img, (u[0], v[0]), (u[2], v[2]), (0,255,0), 5)
    img = cv2.line(img, (u[0], v[0]), (u[3], v[3]), (0,0,255), 5)

    return img


if __name__ == "__main__":
    ycb_item, number, dirname = parse_inputs(sys.argv)
    if ycb_item == "cracker":
        image_folder = "003_cracker_box_16k"
    elif ycb_item == "sugar":
        image_folder = "004_sugar_box_16k"
    elif ycb_item == "spam":
        image_folder = "010_potted_meat_can_16k"

    image_path = "{}/../../{}/00{}/left".format(dirname, image_folder, number)

    detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.8,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
    fx = 276.32251
    fy = 276.32251
    cx = 353.70087
    cy = 179.08852
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    dist = [0]*4
    camera_params = [fx, fy, cx, cy]
    failed_detections = 0
    for image_file in os.listdir(image_path):
        if image_file.endswith(".png"):
            image_file = image_path + "/" + image_file
            image = cv2.imread(image_file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            results = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=32*1e-3)
            # print("[INFO] {} total AprilTags detected\nresults: {}".format(len(results), results))
            if len(results) >= 1:
                for result in results:
                    R = result.pose_R
                    t = result.pose_t.ravel()
                    image = draw_frame(image, K, R, t, scale=0.5)
                cv2.imshow('img',image)
                cv2.waitKey(0) # waits until a key is pressed
                cv2.destroyAllWindows() # destroys the window showing image        break
            else:
                failed_detections += 1
                print(f"Failed detection of image {image_file}\nIn total {failed_detections} failed detections")

