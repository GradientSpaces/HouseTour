from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.image_pairs import make_pairs
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

from time import time
import os
import numpy as np
from matplotlib import pyplot as pl
import h5py
from tqdm import tqdm
import torch

device = 'cuda'
batch_size = 1
schedule = 'cosine'
lr = 0.01
niter = 300
model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
width, height = 512, 288

model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
keyframe_dir = '../keyframes_resized/'

def match_keyframes(fname1, fname2):
    """
    Match keyframes between two images using DUST3R model.
    This function processes two images and finds corresponding keypoints between them using
    the DUST3R neural network for feature detection and matching.
    Args:
        fname1 (str): Filename of the first keyframe image.
        fname2 (str): Filename of the second keyframe image.
    Returns:
        tuple:
            - matches_im0 (numpy.ndarray): Nx2 array of keypoint coordinates in first image.
            - matches_im1 (numpy.ndarray): Nx2 array of keypoint coordinates in second image.
            - match_conf (numpy.ndarray): Confidence scores for the matches.
            - images (list): List containing the loaded input images.
    """

    im1_path = os.path.join(keyframe_dir, fname1)
    im2_path = os.path.join(keyframe_dir, fname2)

    start_time = time()

    images = load_images([im1_path, im2_path], size=512, verbose=False)
    # pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference([tuple(images)], model, device, batch_size=batch_size, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
    conf1, conf2 = pred1['conf'].squeeze(0).detach().T, pred2['conf'].squeeze(0).detach().T

    # scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)

    # imgs = scene.imgs
    # focals = scene.get_focals()
    # poses = scene.get_im_poses()
    # pts3d = scene.get_pts3d()
    # confidence_masks = scene.get_masks()
    # confidences = scene.get_conf()

    # visualize reconstruction
    # scene.show()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=4,
                                                   device=device, dist='dot', block_size=2**15)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    conf1 = conf1[matches_im0[:, 0], matches_im0[:, 1]]
    conf2 = conf2[matches_im1[:, 0], matches_im1[:, 1]]
    match_conf = np.min([conf1, conf2], axis=0)
    valid_matches = valid_matches_im0 & valid_matches_im1 #& (match_conf > 2.0)
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    print(f"Time taken: {time() - start_time:.2f}s")

    return matches_im0, matches_im1, match_conf, images

def visualize_matches(matches_im0, matches_im1, imgs):
    n_viz = 20
    num_matches = matches_im0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        # print(f"Match {i}: ({x0}, {y0}) -> ({x1}, {y1})")
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)



if __name__ == '__main__':    
    vocab_matches = []
    with open("../SuperMAP/vocab_matches.txt", 'r') as file:
        for line in file:
            vocab_matches.append(line.strip().split()[:2])

    n_total = len(vocab_matches)

    window_size = 3

    vocab_matches_dict = {}
    for pair in vocab_matches:
        if pair[0] not in vocab_matches_dict:
            vocab_matches_dict[pair[0]] = [pair[1]]
        else:
            vocab_matches_dict[pair[0]].append(pair[1])
    
    keyframe_imgs = sorted(os.listdir(keyframe_dir))

    for i, img in enumerate(keyframe_imgs):
        if img not in vocab_matches_dict:
            vocab_matches_dict[img] = [keyframe_imgs[i + j] for j in range(1, window_size + 1) if i + j < len(keyframe_imgs)]
            n_total += len(vocab_matches_dict[img])
        for j in range(1, window_size + 1):
            if i + j < len(keyframe_imgs):
                if keyframe_imgs[i + j] not in vocab_matches_dict[img]:
                    vocab_matches_dict[img].append(keyframe_imgs[i + j])
                    n_total += 1

    pbar = tqdm(total=n_total)
    with h5py.File('matches.h5', 'a') as hdf:
        for i, keyframe in enumerate(vocab_matches_dict):
            fname_i = keyframe.split('.')[0]
            group = hdf.require_group(fname_i)
            for j in vocab_matches_dict[keyframe]:
                matches_im0, matches_im1, match_conf, imgs = match_keyframes(keyframe, j)
                if matches_im0 is None:
                    pbar.update(1)
                    pbar.set_postfix_str(f"Couldn't Add {fname_i} and {j}")
                    continue
                match_idx_im0 = np.array([x * height + y for x, y in matches_im0])
                match_idx_im1 = np.array([x * height + y for x, y in matches_im1])

                data = np.stack([match_idx_im0, match_idx_im1], axis=-1).T

                if len(keyframe_imgs) < 500:
                    max_matches = 3000
                elif len(keyframe_imgs) < 800:
                    max_matches = 2000
                else:
                    max_matches = 1000

                if data.shape[1] > max_matches:
                    indexes = np.linspace(0, data.shape[1] - 1, max_matches, dtype=int)
                    data = data[:, indexes]

                # if data.shape[1] > 1000:
                #     top_conf_matches = np.argsort(match_conf)[::-1][:1000]
                #     data = data[:, top_conf_matches]
                
                print(f'found {data.shape[1]} matches')

                fname_j = j.split('.')[0]
                group.create_dataset(fname_j, data=data)
                pbar.update(1)
                pbar.set_postfix_str(f"Processed {fname_i} and {fname_j}")
    
    pbar.close()


