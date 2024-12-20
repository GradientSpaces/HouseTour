import os
import random
import torch
from PIL import Image
from typing import Dict, Optional, Sequence
import transformers
import pathlib
import json
import copy
import math
import einops

from grandtour.conversation.chat import Conversation,SeparatorStyle
from grandtour.datasets.base_dataset import BaseDataset
from grandtour.processors.img_array_processors import ImgArrayProcessor

class HouseTour_Dataset_Pretrain(BaseDataset):
    def __init__(
        self, 
        vis_processor, 
        vis_root, 
        ann_root,
        num_video_query_token=32,
        max_txt_len= 512,
    ):
        """
        vis_root: str, path to the directory containing the candidate images
        ann_root: str, path to the annotation file
        """
        super().__init__(vis_processor=vis_processor, text_processor=None)

        with open(ann_root, 'r') as f:
            self.annotation = json.load(f)

        self.window_size = len(self.annotation[0]["candidates"])
        self.num_video_query_token = num_video_query_token
        self.vis_root = vis_root
        self.max_txt_len = max_txt_len
        self.transform = vis_processor

    def _get_image_path(self, scene_id, candidate_name):
        scene_id_folder = f"{scene_id}_video"
        return os.path.join(self.vis_root, scene_id_folder, "keyframes", candidate_name)
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        sample = self.annotation[idx]
        scene_id = sample['scene_id']
        candidates_paths = [self._get_image_path(scene_id, candidate_name) for candidate_name in sample['candidates']]
        text = sample['text']

        img_arr_tensors = self.transform(candidates_paths)
        img_arr_tensors = einops.rearrange(img_arr_tensors, 'c t h w -> t c h w')
        img_arr_tensors = img_arr_tensors.half()

        return {
            "image": img_arr_tensors,
            "text_input": text
        }




           

