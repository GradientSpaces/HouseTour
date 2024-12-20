# from autodq import eval_q_and_a_bool
import os

cache_dir="/scratch/users/atacelen/.cache/"
os.environ['HF_HOME'] = cache_dir

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import json
from tqdm import tqdm

warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def get_keyframes_w_timestamps(dir_path):
    with open(os.path.join(dir_path, "keyframes", "timestamps.json")) as ts_file:
        timestamps = json.load(ts_file)
    
    timestamps_keys = list(timestamps.keys())
    keyframe_timestamps = {}

    keyframes_path = os.path.join(dir_path, "keyframes")
    keyframes = [file for file in os.listdir(keyframes_path) if file.endswith('.png')]
    keyframes = list(sorted(keyframes))

    for k in keyframes:
        k_idx = int(k.split(".")[0].split("_")[1])
        try:
            keyframe_timestamps[k] = timestamps[timestamps_keys[k_idx]]
        except:
            print(k_idx)
            print(k)

    return keyframe_timestamps

def load_candidates(scene_id):
    path_to_recon = f"/scratch/users/atacelen/Reconstructions3D/{scene_id}_video"
    
    ts = get_keyframes_w_timestamps(path_to_recon)
    
    with open(os.path.join(path_to_recon, "candidates.txt")) as f:
        candidates = f.read()
        candidates = candidates.split("\n")
    
    image_arrays = []
    frame_time = []
     
    image_dir = os.path.join(path_to_recon, "keyframes_resized")
    for img_file in candidates:
        img_path = os.path.join(image_dir, img_file)       
        img = Image.open(img_path).convert('RGB')         
        image_arrays.append(np.array(img))
        frame_time.append(ts[img_file])
    video_time = max(frame_time)
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    image_arrays = np.stack(image_arrays, axis=0)

    return image_arrays, frame_time, video_time

def generate_caption(video_idx, question):
    max_frames_num = 96
    print(f"Video ID: {video_idx}")
    # video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
    video, frame_time, video_time = load_candidates(video_idx)

    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
    video = [video]
    
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{question}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    with torch.cuda.amp.autocast():
        cont = model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    return text_outputs

if __name__ == "__main__":
    # Captioning Eval
    with open("/scratch/users/atacelen/Reconstructions3D/annotations_cleaned_v2.json", "r") as f:
        capt = json.load(f)

    with open("/scratch/users/atacelen/Reconstructions3D/val_indexes.txt", "r") as f:
        val_idxs = f.read()
        val_idxs = val_idxs.split("\n")
        val_idxs = [int(v) for v in val_idxs]

    capt = [d for d in capt if d["scene_id"] in val_idxs]

    pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"

    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    model.eval()

    preds = []

    output_file = "eval_llava_next_capt.jsonl"

    for d in tqdm(capt):
        video_path = f"/scratch/users/atacelen/HouseTourVideos/Videos/{d["scene_id"]}_video.mp4"
        question = d["text"]["instruction"]
        gt_answer = d["text"]["response"]
        predicted_answer = generate_caption(d["scene_id"], question)

        print(f"Scene: {d['scene_id']}")
        print(f"Pred: {predicted_answer}")
        print(f"Label: {gt_answer}")
        result = {
            "scene_id": d["scene_id"],
            "question": question,
            "predicted_answer": predicted_answer,
            "ground_truth_answer": gt_answer
        }
        preds.append(result)
        # print(result)
    
    with open(output_file, "w") as file:
        for item in preds:
            json.dump(item, file)
            file.write("\n")  # Add a newline after each JSON object

