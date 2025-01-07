import os
import random
import torch
from torch.utils.data.dataloader import default_collate
from PIL import Image
from typing import Dict, Optional, Sequence
import transformers
import pathlib
import json
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import math

from cosmos_tokenizer.conversation.chat import Conversation,SeparatorStyle
from cosmos_tokenizer.datasets.base_dataset import BaseDataset
from cosmos_tokenizer.processors.img_array_processors import ImgArrayProcessor

DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100

llama_v2_image_conversation = Conversation(
    system=" ",
    roles=("USER", "ASSISTANT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

class HouseTour_Dataset(BaseDataset):
    def __init__(
        self, 
        vis_processor, 
        text_processor, 
        vis_root, 
        ann_root,
        num_video_query_token=32,
        tokenizer_name = 'ckpt/Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf/',
        data_type = 'image_arr', 
        model_type= 'llama_v2',
        sample_type= 'rand',
        max_txt_len= 512,
        resize_size= -1,
        # stride= 32,
    ):
        """
        vis_root: str, path to the directory containing the candidate images
        ann_root: str, path to the annotation file
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        with open(ann_root, 'r') as f:
            self.annotation = json.load(f)

        self.num_video_query_token = num_video_query_token
        self.vis_root = vis_root
        self.resize_size = resize_size
        # self.stride = stride
        self.max_txt_len = max_txt_len

        self.transform = ImgArrayProcessor(image_size=self.resize_size)

        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]

        self.data_type = data_type
        self.model_type = model_type

    def _get_image_path(self, scene_id, candidate_name):
        scene_id_folder = f"{scene_id}_video"
        return os.path.join(self.vis_root, scene_id_folder, "keyframes_resized", candidate_name)
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        sample = self.annotation[idx]
        scene_id = sample['scene_id']
        candidates_paths = [self._get_image_path(scene_id, candidate_name) for candidate_name in sample['candidates']]
        text = sample['text']

        img_arr_tensors = self.transform(candidates_paths)

        cur_n_frm = img_arr_tensors.size(1)
        # cur_token_len = self.num_video_query_token * math.ceil(cur_n_frm / self.stride) if self.stride > 0 else self.num_video_query_token
        cur_token_len = cur_n_frm

        # Prepare multimodal input
        multim_text = copy.deepcopy(text)
        multim_text['instruction'] = "<Video>" + DEFAULT_IMAGE_PATCH_TOKEN * cur_token_len + "</Video> " + multim_text['instruction']

        # preprocess for llama_v2
        conversations = []
        conv = copy.deepcopy(llama_v2_image_conversation)
        roles = conv.roles

        conv.append_message(role=roles[0], message=multim_text['instruction'])
        conv.append_message(role=roles[1], message=multim_text['response'])

        conversations.append(conv.get_prompt())

        # Tokenize
        input_ids = self.tokenizer(conversations, return_tensors='pt', padding='longest', max_length=self.max_txt_len, truncation=True).input_ids
        targets = copy.deepcopy(input_ids)

        sep = "[/INST] "
        for conversation, target in zip(conversations, targets):
            # total_len = int(target.ne(tokenizer.pad_token_id).sum())
            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                round_len = len(self.tokenizer(rou).input_ids)
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2  # 为什么减去2,speical token 的数目

                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX
        
        data_dict = dict(input_ids=input_ids[0], labels=targets[0])
        data_dict['image'] = img_arr_tensors

        return {
            "image": img_arr_tensors,
            "text_input": data_dict['input_ids'],
            "labels": data_dict['labels']
        }
    
    def get_scene_id(self, idx):
        return self.annotation[idx]['scene_id']
    
    def collater(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("text_input", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        batch['conv_type'] = 'multi'
        return batch




           

