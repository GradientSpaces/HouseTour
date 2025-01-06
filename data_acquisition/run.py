from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import read_binaries
from dust3r.demo import get_3D_model_from_scene

from read_bin import read_images_binary

from scipy.spatial.transform import Rotation as R
from scipy.linalg import inv
import numpy as np
import torch
import os
import argparse


def find_models(root_dir):
    model_paths = []
    
    # Walk through all subdirectories and files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if 'images.bin' is in the current list of filenames
        if 'images.bin' in filenames:
            # Create the full path to the 'images.bin' file
            full_path = os.path.join(dirpath, 'images.bin')
            model_paths.append(dirpath)
    
    return model_paths


if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.02
    niter = 1000

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory

    args = argparse.ArgumentParser()
    args.add_argument('--folder', type=str, required=True, help='The path to the binaries folder')
    args = args.parse_args()

    list_models = find_models(args.folder)
    list_models = [model_path for model_path in list_models if len(read_images_binary(model_path + '/images.bin')) > 10]
    list_models = sorted(list_models, key=lambda m: min(read_images_binary(m + '/images.bin').keys()), reverse=True)

    model_path = list_models[0]
    images_bin_path = model_path + '/images.bin'
    cameras_bin_path = model_path + '/cameras.bin'
    # resizes_dir = '../keyframes_resized'
    resizes_dir = args.folder + '/keyframes_resized'

    images_binary = read_binaries.read_images_binary(images_bin_path)
    images_binary = dict(sorted(images_binary.items(), key=lambda x: x[1].name))
    cameras = read_binaries.read_cameras_binary(cameras_bin_path)

    poses = []
    image_paths = []
    add = 0

    # keyframes_list = os.listdir('../keyframes_resized')
    keyframes_list = os.listdir(resizes_dir)
    idxs = np.linspace(0, len(images_binary) - 1, 100, dtype=int)
    images_binary = {k: v for i, (k, v) in enumerate(images_binary.items()) if i in idxs}

    for image_id, image in images_binary.items():
        # if add % 6 != 0:
        #     add += 1
        #     continue
        qvec = image.qvec
        tvec = image.tvec
        # To matrix
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = R.from_quat(qvec, scalar_first=True).as_matrix()
        cam_to_world[:3, 3] = tvec
        cam_to_world = inv(cam_to_world)
        poses.append(cam_to_world)
        if image.name not in keyframes_list:
            continue
        # image_paths.append(f"../keyframes_resized/{image.name}")
        image_paths.append(f"{resizes_dir}/{image.name}")
        add += 1

    print(f"Number of images: {len(image_paths)}")

    poses_tensor = torch.tensor(poses, dtype=torch.float16).to(device)
    # images = load_images(['croco/assets/Chateau1.png', 'croco/assets/Chateau2.png'], size=512)
    images = load_images(image_paths, size=512)

    pairs = make_pairs(images, scene_graph='complete', prefilter="seq3", symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    print("Running global aligner")
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.preset_pose(poses_tensor)
    scene.preset_focal([cameras[img.camera_id].params[0] for _,img in images_binary.items()])
    loss = scene.compute_global_alignment(init="known_poses", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    pts3d_np = [pts.cpu().detach().numpy() for pts in pts3d]
    
    # Save the 3D points
    np.save('pts3d.npy', pts3d_np)

    # visualize reconstruction
    # scene.show()

    # save the scene
    get_3D_model_from_scene(outdir = './', silent=False, scene=scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=True, transparent_cams=False, cam_size=0.05)