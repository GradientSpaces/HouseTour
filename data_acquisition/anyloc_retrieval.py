from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
)
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 512, 288

images = Path("PATH_TO_RESIZED_KEYFRAMES")

outputs = Path("outputs/sfm_anyloc/")
sfm_pairs = outputs / "pairs.txt"
sfm_dir = outputs / "sfm_anyloc"
global_descriptors = outputs / "global_descriptors.h5"
features = outputs / "features.h5"
matches = outputs / "matches.h5"
valid_pairs = outputs / "valid_pairs.txt"

feature_conf = extract_features.confs["disk"]
matcher_conf = match_features.confs["disk+lightglue"]

pairs_from_retrieval.main(global_descriptors, sfm_pairs, num_matched=20)

feature_path = extract_features.main(feature_conf, images, feature_path=features)
match_path = match_features.main(
    matcher_conf, sfm_pairs, features=features, matches=matches
)


def visualize(keypoint1, keypoint2, matches, mask):
    """
    Visualizes matching keypoints between two images with colored lines connecting corresponding points.
    This function loads keypoints and global descriptors from HDF5 files, calculates cosine similarity
    between global descriptors, and creates a visualization showing matched points between two images
    with colored lines connecting them. The visualization includes the cosine similarity score and
    match statistics.
    Args:
        keypoint1 (str): Filename/key for the first image's keypoints in the HDF5 file
        keypoint2 (str): Filename/key for the second image's keypoints in the HDF5 file
        matches (list): List of matching point indices between the two images
        mask (list): Binary mask indicating valid matches after RANSAC filtering
    Requires:
        - HDF5 files containing keypoints and global descriptors
        - Image files corresponding to the keypoint filenames
        - Global variables: features, global_descriptors, images (paths to required files)
    Returns:
        None. Displays the visualization using matplotlib.
    """

    points2d = {}
    with h5py.File(features, 'r') as f:
        for key, val in f.items():
            if key == keypoint1:
                points2d[keypoint1] = val['keypoints'][:]
            if key == keypoint2:
                points2d[keypoint2] = val['keypoints'][:]

    with h5py.File(global_descriptors, 'r') as f:
        for key, val in f.items():
            if key == keypoint1:
                global_desc1 = val['global_descriptor'][:]
            if key == keypoint2:
                global_desc2 = val['global_descriptor'][:]
    
    cosine_similarity = np.dot(global_desc1, global_desc2) / (np.linalg.norm(global_desc1) * np.linalg.norm(global_desc2))
    
    img1 = cv2.imread(str(images / keypoint1))
    img2 = cv2.imread(str(images / keypoint2))

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    img = np.concatenate((img1, img2), axis=1)
    n_viz = len(matches)

    W0 = img1.shape[1]
    cmap = plt.get_cmap('jet')
    mask_idx = 0
    for i, match in enumerate(matches):
        idx1, idx2 = i, match
        if idx2 == -1:
            continue
        if mask[mask_idx] == 0:
            mask_idx += 1
            continue
        x1, y1 = points2d[keypoint1][idx1]
        x2, y2 = points2d[keypoint2][idx2]
        plt.plot([x1, x2 + W0], [y1, y2], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        mask_idx += 1
    
    plt.imshow(img)
    plt.title(f"Cosine similarity: {cosine_similarity:.3f} | No of matches: {n_viz} | After RANSAC: {mask_idx}")
    plt.axis('off')
    plt.show()

def get_ransac_mask(keypoint1, keypoint2, matches):
    """
    Compute RANSAC mask for a pair of keypoints using homography estimation.
    This function takes keypoint matches between two images and computes a binary mask 
    indicating which matches are inliers according to RANSAC homography estimation.
    Args:
        keypoint1 (str): Identifier for the first set of keypoints
        keypoint2 (str): Identifier for the second set of keypoints
        matches (array-like): Array of match indices where matches[i] contains the index 
            of the keypoint in keypoint2 that matches the i-th keypoint in keypoint1. 
            A value of -1 indicates no match.
    Returns:
        numpy.ndarray: Binary mask array of same length as matches, where 1 indicates 
            an inlier match and 0 indicates an outlier match according to the RANSAC 
            homography estimation.
    """

    points2d = {}
    with h5py.File(features, 'r') as f:
        for key, val in f.items():
            if key == keypoint1:
                points2d[keypoint1] = val['keypoints'][:]
            if key == keypoint2:
                points2d[keypoint2] = val['keypoints'][:]

    pts1 = []
    pts2 = []
    for i, match in enumerate(matches):
        idx1, idx2 = i, match
        if idx2 == -1:
            continue
        x1, y1 = points2d[keypoint1][idx1]
        x2, y2 = points2d[keypoint2][idx2]
        pts1.append([x1, y1])
        pts2.append([x2, y2])
    
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return mask.reshape(-1)

valid_matches = []

with h5py.File(matches, 'r+') as f:
    for key, val in f.items():
        key_idx = key.split(".")[0].split("_")[-1]
        if not (130 <= int(key_idx) <= 132):
            continue
        print(f"Key: {key}")
        print("----------------")
        for k, v in val.items():
            matches0 = v["matches0"][:]
            no_of_matches = matches0[matches0 != -1].shape[0]
            print(f"Key: {k}")
            print(f"No of matches: {no_of_matches}")
            if no_of_matches < 4:
                continue
            mask = get_ransac_mask(key, k, matches0)
            print(f"After RANSAC: {len(mask[mask == 1])}")
            visualize(key, k, matches0, mask)
        print("\n")

with open(valid_pairs, 'w') as f:
    for pair in valid_matches:
        f.write(f"{pair[0]} {pair[1]}\n")
