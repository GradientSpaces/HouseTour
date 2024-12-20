import yaml
import os
from PIL import Image
import torch
from torchvision import transforms
import einops
from torch.amp import autocast as autocast
from torch.utils.data import DataLoader, Subset
import logging
import time
import wandb
import argparse
import math
import pynvml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

from grandtour.models.grandtour import GrandTour
from grandtour.conversation.chat import Chat, Conversation, default_conversation, SeparatorStyle, conv_llava_llama_2, StoppingCriteriaSub
from grandtour.processors.img_array_processors import ImgArrayProcessor
from grandtour.datasets.data_utils import prepare_sample
from grandtour.common.logger import MetricLogger, SmoothedValue
from grandtour.common.dist_utils import is_main_process
from grandtour.common.optims import LinearWarmupCosineLRScheduler
from grandtour.datasets.house_tour_dataset import HouseTour_Dataset


def log_stats(self, stats, split_name, output_dir):
    if isinstance(stats, dict):
        log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
        with open(os.path.join(output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
    elif isinstance(stats, list):
        pass

def get_gpu_usage():
    pynvml.nvmlInit()
    
    gpu_count = pynvml.nvmlDeviceGetCount()
    gpu_usage_info = []

    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
        # Memory usage
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used = mem_info.used / 1024**3   # Convert bytes to GB
        memory_total = mem_info.total / 1024**3 # Convert bytes to GB
        memory_percent = (memory_used / memory_total) * 100
        
        # GPU utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = utilization.gpu

        gpu_usage_info.append({
            "gpu_id": i,
            "memory_used_GB": memory_used,
            "memory_total_GB": memory_total,
            "memory_percent": memory_percent,
            "gpu_util_percent": gpu_util,
        })
    
    pynvml.nvmlShutdown()
    
    for gpu in gpu_usage_info:
        print(f"GPU {gpu['gpu_id']}: {gpu['memory_used_GB']:.2f} GB / {gpu['memory_total_GB']:.2f} GB "
              f"({gpu['memory_percent']:.2f}%), GPU Utilization: {gpu['gpu_util_percent']}%")

def find_token_sequence(input_ids, token_sequence):
    """
    Find the start index of a token sequence in input_ids.

    Args:
    - input_ids (torch.Tensor): A 1D tensor of token IDs representing the input sequence.
    - token_sequence (list of int): The list of token IDs to search for.

    Returns:
    - index (int): The starting index of the token sequence, or -1 if not found.
    """
    input_len = input_ids.size(0)
    seq_len = len(token_sequence)

    for i in range(input_len - seq_len + 1):
        if (input_ids[i:i + seq_len] == torch.tensor(token_sequence).to(input_ids.device)).all():
            return i + seq_len
    return -1

def evaluate(model, data_loader, cuda_enabled, step):
    """
    Evaluate the model and calculate accuracy and F1 score.
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    print("Evaluating...")
    with torch.no_grad():
        for samples in tqdm(data_loader):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            images = samples['images'].to(model.device)
            input_ids = samples['input_ids'].to(model.device)
            
            images_embed, _ = model.encode_videoQformer_visual(images)
            
            img_tok_indices = (input_ids[0] == model.IMAGE_PATCH_TOKEN_ID).nonzero(as_tuple=True)[0]
            seg_prev, seg_after = input_ids[:, : img_tok_indices[0]], input_ids[:, img_tok_indices[-1] + 1: ]

            # Seg_after should stop after [/INST]
            end_inst_idx = find_token_sequence(seg_after[0], [518, 29914, 25580, 29962]) #[/INST]
            seg_after = seg_after[:, :end_inst_idx]

            print(f"Q: {model.llama_tokenizer.decode(seg_after[0])}")

            if model.lora:
                seg_embs = [model.llama_model.get_base_model().model.embed_tokens(seg_t) for seg_t in [seg_prev, seg_after]]
            else:
                seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in [seg_prev, seg_after]]
            
            embs = torch.cat((seg_embs[0], images_embed, seg_embs[1]), dim=1)

            max_length=2000
            max_new_tokens=1

            current_max_len = embs.shape[1] + max_new_tokens
            if current_max_len - max_length > 0:
                print('Warning: The number of tokens in current conversation exceeds the max length. '
                      'The model will not see the contexts outside the range.')
            begin_idx = max(0, current_max_len - max_length)


            embs = embs[:, begin_idx:]
            stop_words_ids = [torch.tensor([2]).to(model.device)]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

            output = model.llama_model.generate(
                inputs_embeds=embs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
            )

            all_preds.append(output[0][0].item())
            all_labels.append(samples['labels'][0][-3].item())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")  # Use weighted F1 for imbalance

    wandb.log({
        "val/accuracy": accuracy,
        "val/f1": f1,
        "step": step,
    })
    
    model.train()  # Switch back to training mode
    return {"accuracy": accuracy, "f1_score": f1}

def train_epoch(
        epoch,
        iters_per_epoch,
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler = "linear_warmup_cosine_lr",
        scaler=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
        output_dir ="/scratch/users/atacelen/housetour/results/"
    ):
    use_amp = scaler is not None
    
    # if iter-based runner, schedule lr based on inner epoch.
    print(
        "Start training epoch {}, {} iters per inner epoch.".format(
            epoch, iters_per_epoch
        )
    )
    
    for batch_idx, samples in enumerate(tqdm(train_loader)):
        samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

        lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = model(samples)["loss"]
        
        # after_train_step()
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # update gradients every accum_grad_iters iterations
        if (batch_idx + 1) % accum_grad_iters == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()                     
            else:    
                optimizer.step()
            optimizer.zero_grad()
        
        wandb.log({
            "train/loss": loss.item(),
            "train/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch + 1,
            "step": epoch * len(train_loader) + batch_idx,
        })
        if (batch_idx + 1) % 5000 == 0:
            step = epoch * len(train_loader) + batch_idx + 1
            eval_scores = evaluate(model, val_loader, cuda_enabled, step)
            checkpoint_path = os.path.join(output_dir, f"train_tf_e{epoch}_iter{batch_idx + 1}_acc{eval_scores['accuracy']:.3f}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")

def train(cfg):
    start_time = time.time()
    best_agg_metric = 0
    best_epoch = 0

    max_epochs = cfg["run"]["max_epochs"]
    eval_only = cfg["run"]["eval_only"]

    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GrandTour.from_config(cfg["model"]).to(device)
    
    print("GPU Usage after loading the model...")
    get_gpu_usage()

    #Read the training scene ids
    with open(os.path.join(cfg["data"]["vis_root"], "train_indexes.txt"), "r") as f:
        train_scenes = f.read()
        train_scenes = train_scenes.split("\n")
        train_scenes = [int(i) for i in train_scenes]
    
    #Read the validation scene ids
    with open(os.path.join(cfg["data"]["vis_root"], "val_indexes.txt"), "r") as f:
        val_scenes = f.read()
        val_scenes = val_scenes.split("\n")
        val_scenes = [int(i) for i in val_scenes]

    if not eval_only:
        print("Start training...")
        # Load the dataset
        dataset = HouseTour_Dataset(
            vis_processor=ImgArrayProcessor(image_size=model.visual_encoder.image_size),
            text_processor=None,
            vis_root=cfg["data"]["vis_root"],
            ann_root=cfg["data"]["ann_root"],
            num_video_query_token=cfg["data"]["num_video_query_token"],
            tokenizer_name=cfg["data"]["tokenizer_name"],
            data_type=cfg["data"]["data_type"],
            model_type=cfg["data"]["model_type"],
            sample_type=cfg["data"]["sample_type"],
            max_txt_len=cfg["data"]["max_txt_len"],
            stride=cfg["data"]["stride"],
        )

        train_idxs = [i for i in range(len(dataset)) if dataset.get_scene_id(i) in train_scenes]
        val_idxs = [i for i in range(len(dataset)) if dataset.get_scene_id(i) in val_scenes]

        train_dataset = Subset(dataset, train_idxs)
        val_dataset = Subset(dataset, val_idxs)

        train_loader = DataLoader(
            train_dataset,
            collate_fn = dataset.collater,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["train"]["num_workers"],
            persistent_workers=False,
            pin_memory=True,
            drop_last=False,
        )

        val_loader = DataLoader(
            val_dataset,
            collate_fn = dataset.collater,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["train"]["num_workers"],
            persistent_workers=False,
            pin_memory=True,
            drop_last=False,
        )

        iters_per_epoch = math.ceil(len(train_dataset) / cfg["train"]["batch_size"])

        # Load the optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["train"]["init_lr"]),
            weight_decay=cfg["train"]["weight_decay"],
        )

        # Load the lr_scheduler
        lr_scheduler = LinearWarmupCosineLRScheduler(
            optimizer = optimizer,
            max_epoch = max_epochs,
            iters_per_epoch = iters_per_epoch,
            min_lr = float(cfg["train"]["min_lr"]),
            init_lr = float(cfg["train"]["init_lr"]),
            warmup_start_lr = float(cfg["train"]["warmup_lr"])
        )

        # Load the scaler
        scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["use_amp"])

        print("GPU Usage after loading the model...")
        get_gpu_usage()

        print("Initial Evaluation Before Training...")
        # evaluate_model(
        #     model,
        #     val_loader
        # )

        # Train the model
        for epoch in range(max_epochs):
            train_epoch(
                epoch=epoch,
                iters_per_epoch=iters_per_epoch,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                log_freq=cfg["train"]["log_freq"],
                cuda_enabled=device.type == "cuda",
                accum_grad_iters=cfg["train"]["accum_grad_iters"],
                output_dir=cfg["run"]["output_dir"],
            )
    
    print(f"Total training time: {time.time() - start_time} seconds")
    print("Training complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", required=True)
    args = parser.parse_args()
    cfg_path = args.cfg_path    

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    wandb.init(
        project="grandtour_tf",
        config=cfg,
        name=f"train_e{cfg['run']['max_epochs']}_w{cfg['model']['window_size']}_l{cfg['model']['lora_r']}",
    )

    print("Entering Training...")
    train(cfg)