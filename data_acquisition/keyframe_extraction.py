from SuperGluePretrainedNetwork.models.matching import Matching

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm 
import time
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import cv2
import matplotlib.pyplot as plt
import subprocess
import json

def deblur(input_path, output_path):
    os.chdir("DeblurGANv2")
    subprocess.run(["python", "predict.py", '--input_path', input_path, '--output_dir', output_path])
    os.chdir("..")

# Image 240 - 3040

device = "cuda" if torch.cuda.is_available() else "cpu"

matching = Matching(config={'superglue' : {'weights' : 'indoor'}})
matching = matching.eval()
matching = matching.to(device)

def is_image_blurry(image_path, threshold=100):
    """
    Determines if an image is blurry by calculating the Laplacian variance.
    The function uses the Laplacian operator to measure the second derivative of the image,
    which helps detect edges and rapid intensity changes. A low variance of the Laplacian
    indicates a blurry image.
    Args:
        image_path (str): Path to the image file to be analyzed
        threshold (int, optional): Threshold value for determining blurriness. Defaults to 100.
                                 Lower values make the function more strict in detecting blur.
    Returns:
        bool: True if the image is considered blurry, False otherwise
    """

    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    is_blurry = variance < threshold
    return is_blurry

def get_matches(image_path_1, image_path_2):
    """
    Computes image matching between two images using a pre-trained deep learning model.
    This function takes two image paths, processes them through a matching model, and returns
    matching statistics and corresponding keypoints between the images.
    Args:
        image_path_1 (str): File path to the first image
        image_path_2 (str): File path to the second image
    Returns:
        dict: A dictionary containing:
            - percentage (float): Ratio of matches to total keypoints
            - high_confidence_matches (int): Number of matches with confidence > 0.7
            - match_idxs (numpy.ndarray): Coordinates of matched keypoints in first image
            - matched_correspondences (numpy.ndarray): Coordinates of corresponding keypoints in second image
    """

    data = {}
    for i, image_path in enumerate([image_path_1, image_path_2]):
        image = Image.open(image_path)
        #Resize image
        image = image.resize((640, 480))
        image = np.array(image, dtype=np.float32).mean(axis=-1) / 255.0
        image = np.pad(image, [(0, int(np.ceil(s/8))*8 - s) for s in image.shape[:2]])
        # Show the image
        image = torch.from_numpy(image[None,None]).float().to(device)
        data[f'image{i}'] = image

    pred = matching(data)

    # Get the percentage of matches 
    is_match = pred['matches0'][0].cpu().numpy() > -1
    # print(f"No of matches: {is_match.sum()}")
    # print(f"Percentage of matches to keypoints: {is_match.sum() / len(is_match)}")
    high_confidence_matches =  pred['matching_scores0'][0][is_match] > 0.7
    # print(f"No of high confidence matches: {high_confidence_matches.sum()}")

    # Get the image indexes of high confidence matches
    match_idxs = pred['keypoints0'][0][is_match][high_confidence_matches].cpu().detach().numpy()
    matches0 = pred['matches0'][0][is_match][high_confidence_matches].cpu().detach().numpy()
    keypoints = pred['keypoints1'][0][matches0].cpu().detach().numpy()


    return {"percentage" : is_match.sum() / len(is_match), "high_confidence_matches" : high_confidence_matches.sum(), "match_idxs" : match_idxs, "matched_correspondences" : keypoints}


def get_keyframes(image_dir):
    """
    Extracts keyframes from a sequence of images based on optical flow and feature matching criteria.
    This function analyzes consecutive frames to identify significant visual changes and selects
    frames that represent distinct views or scenes. It uses feature matching and optical flow
    calculations to determine when to select a new keyframe.
    Parameters:
        image_dir : str
            Path to the directory containing sequential image frames named as 'frame_XXXX.png'
    Returns:
        tuple
            A tuple containing two elements:
            - keyframes (list): List of paths to the selected keyframe images
            - keyframe_timestamps (dict): Dictionary mapping keyframe paths to their timestamps in seconds
    """

    keyframes = []
    keyframe_timestamps = {}
    fps = 15
    image_dir_list = os.listdir(image_dir)
    for i in tqdm(range(0, len(image_dir_list), 1)):
        keyframe_path = f"{image_dir}/frame_{str(i).zfill(4)}.png"
        if keyframes == []:
            keyframes.append(keyframe_path)
            keyframe_timestamps[keyframe_path] = (i * (1/fps))
            print(f"Image {i} will be added as {len(keyframes)}th keyframe")
            continue

        image_path_1 = keyframes[-1]
        image_path_2 = keyframe_path
        preds = get_matches(image_path_1, image_path_2)
        match_idxs = np.array(preds['match_idxs'])
        keypoints = np.array(preds['matched_correspondences'])
        optical_flow = np.linalg.norm(match_idxs - keypoints, axis=1)
        mean_optical_flow = np.mean(optical_flow)

        if (0.65 > preds['percentage'] > 0.1 and mean_optical_flow / 640 > 0.02 and preds['high_confidence_matches'] < 150) or mean_optical_flow / 640 > 0.08 or preds['high_confidence_matches'] < 10:
            keyframes.append(keyframe_path)
            keyframe_timestamps[keyframe_path] = (i * (1/fps))
            print(f"Image {i} will be added as {len(keyframes)}th keyframe")

    return keyframes, keyframe_timestamps


if __name__ == "__main__":
    keyframes, timestamps = get_keyframes('../images')
    print(len(keyframes))
    print("Done")
    # Create a folder to store the keyframes
    if not os.path.exists('../keyframes'):
        os.makedirs('../keyframes')
    for i, keyframe in enumerate(keyframes):
        image_path = f"../keyframes/keyframe_{str(i).zfill(4)}.png"
        os.system(f"cp {keyframe} {image_path}")

    with open(os.path.join('../keyframes', 'timestamps.json'), 'w') as json_file:
        json.dump(timestamps, json_file, indent=4)
    print("Keyframes saved")