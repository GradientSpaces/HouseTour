import os
import sys
import argparse
import logging
import random
import json
from pathlib import Path

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
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2-VL with LoRA and TRL SFTTrainer"
    )

    parser.add_argument("--model-id", type=str,
                        default="Qwen/Qwen2-VL-7B-Instruct",
                        help="Pretrained HuggingFace model identifier")
    parser.add_argument("--data_dir", type=Path,
                        default=Path("/scratch/users/atacelen/house_tour_dataset/Reconstructions3D"),
                        help="Path to 3D reconstructions")
    parser.add_argument("--annotations", type=Path,
                        default=None,
                        help="Annotations JSON file (default: data_dir/annotations_cleaned_v2.json)")
    parser.add_argument("--train-indexes", type=Path,
                        default=None,
                        help="Train indexes file (default: data_dir/train_indexes.txt)")
    parser.add_argument("--val-indexes", type=Path,
                        default=None,
                        help="Validation indexes file (default: data_dir/val_indexes.txt)")
    parser.add_argument("--output-dir", type=Path,
                        required=True,
                        help="Directory to save fine-tuned model and checkpoints")
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--train-batch-size", type=int, default=1,
                        help="Per-device training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=1,
                        help="Per-device evaluation batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--warmup-ratio", type=float, default=0.03,
                        help="Warmup ratio for scheduler")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (cuda or cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Torch dtype for model")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout
    )


def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_data(sample: dict, data_dir: Path) -> list:
    """
    Convert raw annotation to chat message format.
    """
    scene_id = sample["scene_id"]
    candidates = sorted(sample.get("candidates", []))
    system_msg = {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_MESSAGE}]
    }
    user_content = []
    for img in candidates:
        img_path = data_dir/ "Reconstructions3D" / f"{scene_id}_video" / "keyframes_resized" / img
        user_content.append({"type": "image", "image": str(img_path)})
    user_content.append({"type": "text", "text": sample["text"]["instruction"]})
    user_msg = {"role": "user", "content": user_content}
    assistant_msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": sample["text"]["response"]}]
    }
    return [system_msg, user_msg, assistant_msg]


def prepare_dataset(args) -> tuple[list, list]:
    """
    Load annotations and split into train/val formatted datasets.
    """
    recon = args.data_dir
    ann_file = args.annotations or recon / "Reconstructions3D" / "annotations_cleaned_v2.json"
    train_idx = args.train_indexes or recon / "Reconstructions3D" / "train_indexes.txt"
    val_idx = args.val_indexes or recon / "Reconstructions3D" / "val_indexes.txt"

    with open(ann_file, 'r') as f:
        data = json.load(f)
    train_ids = [int(x) for x in open(train_idx) if x.strip()]
    val_ids = [int(x) for x in open(val_idx) if x.strip()]

    train = [format_data(s, recon) for s in data if s['scene_id'] in train_ids]
    val = [format_data(s, recon) for s in data if s['scene_id'] in val_ids]
    logging.info(f"Loaded {len(train)} train and {len(val)} validation samples.")
    return train, val

def collate_fn(examples: list, processor: Qwen2VLProcessor) -> dict:
    texts = [processor.apply_chat_template(e, tokenize=False) for e in examples]
    images = [process_vision_info(e)[0] for e in examples]
    batch = processor(text=texts, images=images, return_tensors='pt', padding=True)

    labels = batch.input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    if isinstance(processor, Qwen2VLProcessor):
        image_ids = [151652, 151653, 151655]
    else:
        image_ids = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for tid in image_ids:
        labels[labels == tid] = -100
    batch['labels'] = labels
    return batch


def main():
    args = parse_args()
    setup_logging()
    logging.info("Starting fine-tuning script")
    set_random_seeds(args.seed)

    global SYSTEM_MESSAGE
    SYSTEM_MESSAGE = (
        "You are an real estate agent, who is touring a real estate. "
        "Describe these properties in detail!"
    )

    # Load model & processor
    processor = Qwen2VLProcessor.from_pretrained(args.model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=getattr(torch, args.dtype),
        attn_implementation="flash_attention_2",
        device_map="auto"
    )

    # PEFT: LoRA config
    peft_cfg = LoraConfig(
        r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(model, peft_cfg)
    peft_model.print_trainable_parameters()

    # Add special vision tokens if needed (no-op if already present)
    specials = processor.tokenizer.special_tokens_map.get("additional_special_tokens", [])
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": specials + [processor.image_token]}
    )

    # Prepare data
    train_data, val_data = prepare_dataset(args)

    # SFTTrainer config
    sft_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        eval_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=(args.dtype=="bfloat16"),
        tf32=True,
        max_grad_norm=0.3,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        report_to="none"
    )
    sft_args.remove_unused_columns = False

    # Initialize trainer
    trainer = SFTTrainer(
        model=peft_model,
        args=sft_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=lambda x : collate_fn(x, processor),
        peft_config=peft_cfg,
        processing_class=processor.tokenizer
    )

    # Save initial checkpoint
    init_ckpt = args.output_dir / "checkpoint-0"
    init_ckpt.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(init_ckpt))

    # Train
    trainer.train()

    # Final save
    trainer.save_model(str(args.output_dir))
    logging.info("Fine-tuning complete. Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()