import os
import sys
import json
import logging
import random
import copy
from pathlib import Path
from time import time
import argparse

# === EARLY CACHE CONFIGURATION ===
# Parse only --cache_dir early to set HF cache before any transformers import
early_parser = argparse.ArgumentParser(add_help=False)
early_parser.add_argument(
    "--cache_dir", type=Path, default=Path("/scratch/users/atacelen/.cache/"),
    help="Hugging Face HF_HOME cache directory (must set before transformers import)"
)
early_args, _remaining_argv = early_parser.parse_known_args()
os.environ['HF_HOME'] = str(early_args.cache_dir)

import numpy as np
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import set_seed
from transformers import Qwen2VLProcessor
from qwen2_vl_3d.modeling_qwen2_vl_3d import Qwen2VL3DForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer
import diffuser.utils as utils
from diffuser.datasets.cam_traj import CameraTrajectoriesDataset
import re

import diffuser.utils as utils
from diffuser.datasets.cam_traj import CameraTrajectoriesDataset

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("trl").setLevel(logging.WARNING)
    logging.getLogger("diffuser").setLevel(logging.WARNING)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Qwen2-VL-3D with LoRA & diffusion adapter",
        parents=[early_parser]
    )
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                        help="HF model identifier")
    parser.add_argument("--template", type=Path, default=Path("custom_chat_template.json"),
                        help="Path to custom chat template JSON")
    parser.add_argument("--lora_adapter_dir", type=Path, required=True,
                        help="Path to the Qwen2-VL LoRA adapter (from train_qwen2_vl.py)")                    
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Root dir for Reconstructions3D and annotations")
    parser.add_argument("--diffuser_dir", type=Path, required=True,
                        help="Residual Diffuser directory (model & diffusion config)")
    parser.add_argument("--traj_data", type=Path, required=True,
                        help="Path to trajectory JSONL data file")
    parser.add_argument("--output_dir", type=Path, default=Path("./training_checkpoints/qwen2_vl_3d"),
                        help="Directory for saving checkpoints and models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_lr", type=float, default=1e-6)
    parser.add_argument("--adapter_lr", type=float, default=1e-4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--push_to_hub", action="store_true")
    return parser.parse_args()

SYSTEM_MESSAGE = (
    "You are a knowledgeable and engaging real estate agent on a property tour. "
    "Provide vivid descriptions using images and 3D position data."
)


def load_chat_template(path: Path) -> dict:
    return json.loads(path.read_text()).get("chat_template", {})

def format_data(sample: dict, traj_cond, traj_imgs, recon_path: Path):
    # random subset of images
    count = max(1, int(len(sample['candidates']) * random.uniform(0.3, 1.0)))
    idxs = sorted(random.sample(range(len(sample['candidates'])), count))
    imgs = [sample['candidates'][i] for i in idxs]
    positions = [traj_imgs.index(img) if traj_cond and img in traj_imgs else -1 for img in imgs]
    # build messages
    msgs = [{"role": "system", "content": [{"type":"text","text":SYSTEM_MESSAGE}]}]
    user_content = []
    for img, pos in zip(imgs, positions):
        user_content.append({"type":"image",
                             "image": str(recon_path / f"{sample['scene_id']}_video" / "keyframes_resized" / img)})
        if pos >= 0:
            user_content.append({"type":"trajectory"})
    user_content.append({"type":"text","text": sample['text']['instruction']})
    msgs.append({"role":"user","content": user_content})
    msgs.append({"role":"assistant","content":[{"type":"text","text":sample['text']['response']} ]})
    return msgs, traj_cond, positions

def add_vision_id(text: str) -> str:
    tokens = re.split(r'(<\|[^|]+?\|>)', text)
    out, count, last = [], 0, None
    for t in filter(None, tokens):
        if t == '<|vision_start|>':
            count += 1; last = count; out.append(f"Picture {count}: "); out.append(t)
        elif t == '<|traj_start|>' and last:
            out.append(f"Position {last}: "); out.append(t)
        else:
            out.append(t)
    return ''.join(out)


def collate_fn(batch, processor, chat_template):
    # Prepare texts with chat template and vision IDs
    texts = [processor.apply_chat_template(x[0], tokenize=False, chat_template=chat_template)
             for x in batch]
    texts = [add_vision_id(t) for t in texts]

    # TODO: This line doesn't seem batchified! 
    # Process images
    images = [process_vision_info(x[0])[0] for x in batch]

    # Tokenize texts and images
    enc = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # Labels: clone input_ids and mask padding
    labels = enc.input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask image tokens in labels
    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652, 151653, 151655]
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for t in image_tokens:
        labels[labels == t] = -100
    enc["labels"] = labels

    # Add trajectory-related fields
    enc["position_3d_idxs"] = [
        torch.tensor(x[2], dtype=torch.long) for x in batch
    ]
    enc["trajectory_conditions_idxs"] = [
        torch.tensor([k for k, v in (x[1] or {}).items()], dtype=torch.long)
        for x in batch
    ]
    enc["trajectory_conditions"] = [
        [torch.Tensor(v.tolist()) for v in (x[1] or {}).values()]
        if x[1] is not None else None
        for x in batch
    ]
    # If no sample has valid conditions, clear the field
    if not any(enc["trajectory_conditions"]):
        enc["trajectory_conditions"] = None

    return enc

def prepare_datasets(args, processor, chat_template):
    ann = json.loads((args.data_dir / 'annotations_cleaned_v2.json').read_text())
    traj_ds = CameraTrajectoriesDataset(str(args.traj_data))
    train_ids = [int(l) for l in (args.data_dir / 'train_indexes.txt').read_text().split()]
    val_ids = [int(l) for l in (args.data_dir / 'val_indexes.txt').read_text().split()]
    map_traj = {int(s.scene_id.item()):s for s in traj_ds}
    def make(sids):
        out=[]
        for s in [d for d in ann if d['scene_id'] in sids]:
            cond = map_traj.get(s['scene_id'])
            out.append(format_data(
                s,
                getattr(cond,'conditions',None),
                getattr(cond,'images',None),
                args.data_dir,
            ))
        return out
    train, val = make(train_ids), make(val_ids)
    logging.info(f"Train: {len(train)}, Val: {len(val)} samples")
    return train, val  

class CustomSFTTrainer(SFTTrainer):
    # Save the whole model, instead of just the adapter 
    def save_model(self, output_dir=None, _internal_call=False):
        model_copy = copy.deepcopy(self.model)
        if hasattr(model_copy, 'merge_and_unload'):
            merged_model = model_copy.merge_and_unload()
        else:
            merged_model = model_copy
        merged_model.save_pretrained(output_dir or self.args.output_dir)

def main():
    args = parse_args()
    setup_logging(args.log_level)
    logging.info("Arguments: %s", vars(args))
    set_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    model_cls = utils.load_config(str(args.diffuser_dir), 'model_config.pkl')()
    diff_cfg = utils.load_config(str(args.diffuser_dir), 'diffusion_config.pkl')
    if hasattr(diff_cfg,'_dict') and 'is_manifold_aware' in diff_cfg._dict:
        del diff_cfg._dict['is_manifold_aware']
    diffuser = diff_cfg(model_cls)
    diffuser.get_bottleneck_feats = True

    base = Qwen2VL3DForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='cuda'
    )
    base.diffuser = diffuser

    proc_dir = Path(args.output_dir) / "qwen2_vl_processor"
    if proc_dir.exists():
        processor = Qwen2VLProcessor.from_pretrained(proc_dir)
    else:
        processor = Qwen2VLProcessor.from_pretrained(args.model_id)
        
        # Add Special Tokens for trajectory
        traj_special_tokens = {"additional_special_tokens":processor.tokenizer.special_tokens_map["additional_special_tokens"] + ["<|traj_start|>", "<|traj_end|>", "<|traj_pad|>"]}
        num_added_tokens = processor.tokenizer.add_special_tokens(traj_special_tokens)

        proc_dir.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(proc_dir)
        processor = Qwen2VLProcessor.from_pretrained(proc_dir)
    
    base.resize_token_embeddings(len(processor.tokenizer))
    
    base.traj_start_token = processor.tokenizer.convert_tokens_to_ids("<|traj_start|>")
    base.traj_end_token = processor.tokenizer.convert_tokens_to_ids("<|traj_end|>")
    base.traj_pad_token = processor.tokenizer.convert_tokens_to_ids("<|traj_pad|>")

    chat_template = load_chat_template(args.template)

    peft_cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                          lora_dropout=args.lora_dropout, bias='none',
                          target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM')
    
    #TODO: Add a new parser argument for the pretrained model path
    model = PeftModel.from_pretrained(base, str(args.output_dir / args.lora_adapter_dir), is_trainable=True)
    for n,p in model.named_parameters(): p.requires_grad=False
    for n,p in model.named_parameters():
        if 'lora_' in n or 'adapter' in n or 'conv_transpose' in n:
            p.requires_grad=True
    for p in model.diffuser.parameters(): p.requires_grad=False

    lora_p = [p for n,p in model.named_parameters() if 'lora_' in n]
    adapter_p = [p for n,p in model.named_parameters() if ('adapter' in n or 'conv_transpose' in n)]
    optimizer = AdamW([{'params':lora_p,'lr':args.lora_lr}, {'params':adapter_p,'lr':args.adapter_lr}])

    train_ds, val_ds = prepare_datasets(args, processor, chat_template)

    sft_args = SFTConfig(
        output_dir=str(args.output_dir), 
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True, 
        optim='adamw_torch_fused',
        learning_rate=args.learning_rate, 
        lr_scheduler_type='cosine',
        logging_steps=10, 
        eval_steps=500, 
        save_steps=500,
        eval_strategy='steps', 
        save_strategy='steps',
        metric_for_best_model='eval_loss', 
        greater_is_better=False,
        load_best_model_at_end=True, 
        bf16=True, 
        tf32=True,
        max_grad_norm=0.1, 
        warmup_ratio=args.warmup_ratio,
        report_to='none', 
        push_to_hub=args.push_to_hub,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    sft_args.remove_unused_columns=False

    trainer = CustomSFTTrainer(
        model=model, args=sft_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=lambda b: collate_fn(b, processor, chat_template),
        optimizers=(optimizer,None), processing_class=processor.tokenizer
    )

    ck0 = args.output_dir / 'checkpoint-0'
    ck0.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ck0))

    trainer.train()
    trainer.save_model()

    if hasattr(model,'merge_and_unload'):
        full = model.merge_and_unload()
        full.save_pretrained(str(args.output_dir/'full_model'))

    logging.info("Training complete")


if __name__ == "__main__":
    main()




