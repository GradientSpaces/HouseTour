import os
import sys
import argparse
import logging
import random
import json
import re
from pathlib import Path

import numpy as np
import torch

# === EARLY CACHE CONFIGURATION ===
# Parse only --cache_dir early to set HF cache before any transformers import
early_parser = argparse.ArgumentParser(add_help=False)
early_parser.add_argument(
    "--cache_dir", type=Path, default=Path("/scratch/users/atacelen/.cache/"),
    help="Hugging Face HF_HOME cache directory (must set before transformers import)"
)
early_args, _remaining_argv = early_parser.parse_known_args()
os.environ['HF_HOME'] = str(early_args.cache_dir)

from transformers import Qwen2VLProcessor
from qwen2_vl_3d.modeling_qwen2_vl_3d import Qwen2VL3DForConditionalGeneration
from qwen_vl_utils import process_vision_info
import diffuser.utils as utils
from diffuser.datasets.cam_traj import CameraTrajectoriesDataset
from diffuser.models.diffusion import GaussianDiffusion
from tqdm import tqdm

SYSTEM_MESSAGE = (
    'You are a knowledgeable and engaging real estate agent on a property tour. '
    'Your task is to provide detailed and vivid descriptions of the properties you '
    'encounter. Some of the provided pictures include 3D position data—use this '
    'information as well. Your language should be clear, professional, and appealing, '
    'aiming to create an immersive experience for potential buyers.'
)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate text and trajectories using Qwen2VL3D model'
    )
    parser.add_argument(
        '--cache-dir', type=Path, default=Path(os.environ.get('HF_HOME', '/scratch/users/atacelen/.cache/')),
        help='HuggingFace cache directory'
    )
    parser.add_argument(
        '--model-id', type=str, default='Qwen/Qwen2-VL-7B-Instruct',
        help='HuggingFace model identifier'
    )
    parser.add_argument(
        '--model-path', type=Path, required=True,
        help='Path to fine-tuned model checkpoint'
    )
    parser.add_argument(
        '--data_dir', type=Path, default=Path('/scratch/users/atacelen/house_tour_dataset/Reconstructions3D'),
        help='Path to 3D reconstructions'
    )
    parser.add_argument(
        '--traj-path', type=Path, required=True,
        help='Path to diffuser config logs'
    )
    parser.add_argument(
        '--traj-data', type=Path, required=True,
        help='Path to trajectory data file'
    )
    parser.add_argument(
        '--annotations', type=Path, default=None,
        help='Path to annotations JSON file (default: data_dir/annotations_cleaned_v2.json)'
    )
    parser.add_argument(
        '--train-indexes', type=Path, default=None,
        help='Path to train indexes file (default: data_dir/train_indexes.txt)'
    )
    parser.add_argument(
        '--val-indexes', type=Path, default=None,
        help='Path to val indexes file (default: data_dir/val_indexes.txt)'
    )
    parser.add_argument(
        '--chat-template', type=Path, default=Path('custom_chat_template.json'),
        help='Path to chat template JSON file'
    )
    parser.add_argument(
        '--output-file', type=Path, default=Path('eval_qwen2_vl_3d.jsonl'),
        help='Path to output JSONL file'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Computation device (e.g., cuda or cpu)'
    )
    parser.add_argument(
        '--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'],
        help='Torch dtype for model'
    )
    parser.add_argument(
        '--max-new-tokens', type=int, default=1024,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--top-p', type=float, default=0.9,
        help='Top-p sampling threshold'
    )
    parser.add_argument(
        '--top-k', type=int, default=100,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--penalty-alpha', type=float, default=0.2,
        help='Penalty alpha for generation'
    )
    parser.add_argument(
        '--repetition-penalty', type=float, default=1.0,
        help='Repetition penalty for generation'
    )
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        stream=sys.stdout
    )


def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_data(sample, trajectory_conditions, trajectory_images, data_dir: Path):
    '''Format a sample into chat messages with 3D images and optional trajectories.'''  
    if trajectory_conditions is not None:
        idx_fn = lambda a: trajectory_images.index(a) if a in trajectory_images else -1
        pos_idxs = [idx_fn(c) for c in sample['candidates']]
    else:
        pos_idxs = [-1] * len(sample['candidates'])
    scene_id = sample['scene_id']
    messages = [
        {'role': 'system',
         'content': [{'type': 'text', 'text': SYSTEM_MESSAGE}]},
        {'role': 'user', 'content': []}
    ]
    for cand, idx in zip(sample['candidates'], pos_idxs):
        img_path = data_dir / f'{scene_id}_video' / 'keyframes_resized' / cand
        messages[1]['content'].append({'type': 'image', 'image': str(img_path)})
        if idx != -1:
            messages[1]['content'].append({'type': 'trajectory'})
    messages[1]['content'].append({'type': 'text', 'text': sample['text']['instruction']})
    messages.append({'role': 'assistant',
                     'content': [{'type': 'text', 'text': sample['text']['response']}]})
    return messages, trajectory_conditions, pos_idxs


def load_qwen_with_diffuser(model_path: Path, diffuser: GaussianDiffusion,
                            dtype: str, device: str):
    '''Load the Qwen2VL3D model and attach the diffuser module.'''
    model = Qwen2VL3DForConditionalGeneration.from_pretrained(
        str(model_path), torch_dtype=getattr(torch, dtype),
        device_map='auto', attn_implementation='flash_attention_2'
    )
    model.diffuser = diffuser
    return model


def generate_text_from_sample(model, processor, sample, args, chat_template):
    '''Generate text and trajectory from a single sample.'''
    messages, traj_cond, pos_idxs = sample
    text_input = processor.apply_chat_template(
        messages[1:2], tokenize=False, add_generation_prompt=True,
        chat_template=chat_template
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text_input], images=image_inputs,
                       return_tensors='pt').to(args.device)
    inputs['position_3d_idxs'] = [torch.tensor(pos_idxs, dtype=torch.long)]
    if traj_cond:
        inputs['trajectory_conditions_idxs'] = [
            torch.tensor(list(traj_cond.keys()), dtype=torch.long)
        ]
        inputs['trajectory_conditions'] = [[
            torch.tensor(v.tolist(), device=args.device)
            for v in traj_cond.values()
        ]]
    else:
        inputs['trajectory_conditions_idxs'] = [torch.tensor([], dtype=torch.long)]
        inputs['trajectory_conditions'] = None

    with torch.no_grad():
        gen_ids, traj_out = model.generate_text_and_trajectory(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=processor.tokenizer.eos_token_id,
            do_sample=True,
            top_p=args.top_p,
            top_k=args.top_k,
            penalty_alpha=args.penalty_alpha,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature
        )
    trimmed = [out[len(orig):] for orig, out in zip(inputs.input_ids, gen_ids)]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return text, traj_out[0]


def prepare_datasets(args):
    '''Load annotations and camera trajectories, and format train/val sets.'''
    recon = args.data_dir
    ann_file = args.annotations or recon / 'annotations_cleaned_v2.json'
    train_idxs = args.train_indexes or recon / 'train_indexes.txt'
    val_idxs = args.val_indexes or recon / 'val_indexes.txt'
    with open(ann_file, 'r') as f:
        lang_data = json.load(f)
    traj_ds = CameraTrajectoriesDataset(str(args.traj_data))
    train_ids = [int(x) for x in open(train_idxs) if x.strip()]
    val_ids = [int(x) for x in open(val_idxs) if x.strip()]
    train_lang = [s for s in lang_data if s['scene_id'] in train_ids]
    val_lang = [s for s in lang_data if s['scene_id'] in val_ids]
    traj_map = {s.scene_id.item(): s for s in traj_ds}
    train = [
        format_data(s,
                    traj_map[s['scene_id']].conditions if s['scene_id'] in traj_map else None,
                    traj_map[s['scene_id']].images if s['scene_id'] in traj_map else None,
                    args.data_dir)
        for s in train_lang
    ]
    val = [
        format_data(s,
                    traj_map[s['scene_id']].conditions if s['scene_id'] in traj_map else None,
                    traj_map[s['scene_id']].images if s['scene_id'] in traj_map else None,
                    args.data_dir)
        for s in val_lang
    ]
    logging.info(f'Train size: {len(train)}, Val size: {len(val)}')
    return train, val


def main():
    args = parse_args()
    setup_logging()
    logging.info('Starting evaluation')
    
    set_random_seeds(args.seed)
    with open(args.chat_template, 'r') as f:
        chat_tpl = json.load(f)['chat_template']
    
    # Load diffuser configurations
    model_cfg = utils.load_config(str(args.traj_path), 'model_config.pkl')
    diffuser_model = model_cfg()
    diff_cfg = utils.load_config(str(args.traj_path), 'diffusion_config.pkl')
    diff_cfg._dict.pop('is_manifold_aware', None)
    diffuser = diff_cfg(diffuser_model)
    diffuser.get_bottleneck_feats = True
    
    # Initialize processor and model
    processor = Qwen2VLProcessor.from_pretrained(args.model_id)
    model = load_qwen_with_diffuser(args.model_path, diffuser,
                                    args.dtype, args.device)
    
    # Add trajectory tokens
    specials = processor.tokenizer.special_tokens_map.get('additional_special_tokens', [])
    processor.tokenizer.add_special_tokens(
        {'additional_special_tokens': specials + ['<|traj_start|>', '<|traj_end|>', '<|traj_pad|>']}
    )
    model.traj_start_token = processor.tokenizer.convert_tokens_to_ids('<|traj_start|>')
    model.traj_end_token = processor.tokenizer.convert_tokens_to_ids('<|traj_end|>')
    model.traj_pad_token = processor.tokenizer.convert_tokens_to_ids('<|traj_pad|>')
    
    # Prepare datasets
    _, val_ds = prepare_datasets(args)
    
    # Evaluate
    results = []
    for sample in tqdm(val_ds, desc='Evaluating'):
        text_pred, _ = generate_text_from_sample(model, processor,
                                                 sample, args, chat_tpl)
        messages = sample[0]
        img_path = messages[1]['content'][0]['image']
        scene_dir = Path(img_path).parent.parent.name
        scene_id = scene_dir.replace('_video', '')
        instr = messages[1]['content'][-1]['text']
        gt_resp = messages[2]['content'][0]['text']
        results.append({
            'scene_id': scene_id,
            'instruction': instr,
            'predicted_answer': text_pred,
            'ground_truth_answer': gt_resp
        })
        print(f"Predicted Answer: {results[-1]['predicted_answer']}")
        print()
        print(f"GT: {results[-1]['ground_truth_answer']}")
    
    #Write
    with open(args.output_file, 'w') as outf:
        for item in results:
            json.dump(item, outf)
            outf.write('\n')
    logging.info(f'Evaluation complete, results saved to {args.output_file}')

if __name__ == '__main__':
    main()