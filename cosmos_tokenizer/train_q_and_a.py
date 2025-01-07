import yaml
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import einops
from torch.amp import autocast as autocast
import logging
import time
import wandb
import argparse
import math
import pynvml
from tqdm import tqdm

from cosmos_tokenizer.models.model import CosmosLlama
from cosmos_tokenizer.conversation.chat import Chat, Conversation, default_conversation, SeparatorStyle, conv_llava_llama_2
from cosmos_tokenizer.processors.img_array_processors import ImgArrayProcessor
from cosmos_tokenizer.datasets.data_utils import prepare_sample
from cosmos_tokenizer.common.logger import MetricLogger, SmoothedValue
from cosmos_tokenizer.common.dist_utils import is_main_process
from cosmos_tokenizer.common.optims import LinearWarmupCosineLRScheduler
from cosmos_tokenizer.datasets.house_tour_dataset import HouseTour_Dataset


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

def evaluate(model, val_loader, cuda_enabled, epoch):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for samples in tqdm(val_loader, desc="Evaluating Q&A"):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            
            with torch.cuda.amp.autocast(enabled=cuda_enabled):
                loss = model(samples)["loss"]
            
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    wandb.log({
        "val/loss": avg_val_loss,
        "epoch": epoch + 1,
    })

    print(f"Validation Loss after epoch {epoch + 1}: {avg_val_loss:.4f}")

    model.train()

    return avg_val_loss

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
            eval_loss = evaluate(model, val_loader, cuda_enabled, step)
            checkpoint_path = os.path.join(output_dir, f"train_cosmos_qa_e{epoch}_iter{batch_idx + 1}_loss{eval_loss:.3f}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")

def train(cfg):
    start_time = time.time()
    best_agg_metric = 0
    best_epoch = 0

    max_epochs = cfg["run"]["max_epochs"]
    eval_only = cfg["run"]["eval_only"]

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CosmosLlama(cfg).to(device)

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
            vis_processor=ImgArrayProcessor(image_size=-1),
            text_processor=None,
            vis_root=cfg["data"]["vis_root"],
            ann_root=cfg["data"]["ann_root"],
            num_video_query_token=cfg["data"]["num_video_query_token"],
            tokenizer_name=cfg["data"]["tokenizer_name"],
            data_type=cfg["data"]["data_type"],
            model_type=cfg["data"]["model_type"],
            sample_type=cfg["data"]["sample_type"],
            max_txt_len=cfg["data"]["max_txt_len"],
            # stride=cfg["data"]["stride"],
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
                output_dir=cfg['run']['output_dir']
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
        project="grandtour_qa",
        config=cfg,
        name=f"cosmos_l{cfg['model']['lora_r']}",
    )

    train(cfg)

    







