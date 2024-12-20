import yaml
import os
from PIL import Image
import torch
from torchvision import transforms
import einops
from torch.amp import autocast as autocast
from torch.utils.data import ConcatDataset
import logging
import time
import wandb
import argparse
import math
import pynvml
from tqdm import tqdm

from grandtour.models.grandtour import GrandTour
from grandtour.conversation.chat import Chat, Conversation, default_conversation, SeparatorStyle, conv_llava_llama_2
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

def evaluate(model, val_loader, cuda_enabled, epoch):
    model.eval()
    val_loss_tf, val_loss_qa, val_loss_capt = 0.0, 0.0, 0.0
    tf_loader, qa_loader, capt_loader = val_loader

    with torch.no_grad():
        for samples in tqdm(tf_loader, desc="Evaluating T/F"):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            
            with torch.cuda.amp.autocast(enabled=cuda_enabled):
                loss = model(samples)["loss"]
            
            val_loss_tf += loss.item()
            

    with torch.no_grad():
        for samples in tqdm(qa_loader, desc="Evaluating Q&A"):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            
            with torch.cuda.amp.autocast(enabled=cuda_enabled):
                loss = model(samples)["loss"]
            
            val_loss_qa += loss.item()
            
    
    with torch.no_grad():
        for samples in tqdm(capt_loader, desc="Evaluating Captioning"):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            
            with torch.cuda.amp.autocast(enabled=cuda_enabled):
                loss = model(samples)["loss"]
            
            val_loss_capt += loss.item()
            

    avg_val_loss_tf = val_loss_tf / len(tf_loader)
    avg_val_loss_qa = val_loss_qa / len(qa_loader)
    avg_val_loss_capt = val_loss_capt / len(capt_loader)

    wandb.log({
        "val/loss": avg_val_loss_tf + avg_val_loss_qa + avg_val_loss_capt,
        "val/loss_tf": avg_val_loss_tf,
        "val/loss_qa": avg_val_loss_qa,
        "val/loss_capt": avg_val_loss_capt,
        "epoch": epoch + 1,
    })

    model.train()


def train_epoch(
        epoch,
        iters_per_epoch,
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler = "linear_warmup_cosine_lr",
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
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

    
    
    evaluate(model, val_loader, cuda_enabled, epoch)


def train(cfg):
    start_time = time.time()
    best_agg_metric = 0
    best_epoch = 0

    max_epochs = cfg["run"]["max_epochs"]
    eval_only = cfg["run"]["eval_only"]

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GrandTour.from_config(cfg["model"]).to(device)

    print("GPU Usage after loading the model...")
    get_gpu_usage()

    if not eval_only:
        print("Start training...")
        # Load the dataset
        full_dataset_tf = HouseTour_Dataset(
            vis_processor=ImgArrayProcessor(image_size=model.visual_encoder.image_size),
            text_processor=None,
            vis_root=cfg["data"]["vis_root"],
            ann_root=cfg["data"]["ann_root_tf"],
            num_video_query_token=cfg["data"]["num_video_query_token"],
            tokenizer_name=cfg["data"]["tokenizer_name"],
            data_type=cfg["data"]["data_type"],
            model_type=cfg["data"]["model_type"],
            sample_type=cfg["data"]["sample_type"],
            max_txt_len=cfg["data"]["max_txt_len"],
            stride=cfg["data"]["stride"],
        )
        full_dataset_qa = HouseTour_Dataset(
            vis_processor=ImgArrayProcessor(image_size=model.visual_encoder.image_size),
            text_processor=None,
            vis_root=cfg["data"]["vis_root"],
            ann_root=cfg["data"]["ann_root_qa"],
            num_video_query_token=cfg["data"]["num_video_query_token"],
            tokenizer_name=cfg["data"]["tokenizer_name"],
            data_type=cfg["data"]["data_type"],
            model_type=cfg["data"]["model_type"],
            sample_type=cfg["data"]["sample_type"],
            max_txt_len=cfg["data"]["max_txt_len"],
            stride=cfg["data"]["stride"],
        )
        full_dataset_capt = HouseTour_Dataset(
            vis_processor=ImgArrayProcessor(image_size=model.visual_encoder.image_size),
            text_processor=None,
            vis_root=cfg["data"]["vis_root"],
            ann_root=cfg["data"]["ann_root_capt"],
            num_video_query_token=cfg["data"]["num_video_query_token"],
            tokenizer_name=cfg["data"]["tokenizer_name"],
            data_type=cfg["data"]["data_type"],
            model_type=cfg["data"]["model_type"],
            sample_type=cfg["data"]["sample_type"],
            max_txt_len=cfg["data"]["max_txt_len"],
            stride=cfg["data"]["stride"],
        )

        train_dataset_tf, val_dataset_tf = x(full_dataset_tf, [0.98, 0.02])
        train_dataset_qa, val_dataset_qa = torch.utils.data.random_split(full_dataset_qa, [0.98, 0.02])
        train_dataset_capt, val_dataset_capt = torch.utils.data.random_split(full_dataset_capt, [0.9, 0.1])

        train_dataset_full = ConcatDataset([train_dataset_tf, train_dataset_qa, train_dataset_capt])

        print(f"Length Train Dataset: {len(train_dataset_full)}")
        print(f"Length Val Datasets T/F - Q&A - Capt. {(len(val_dataset_tf), len(val_dataset_qa), len(val_dataset_capt))}")

        train_loader = torch.utils.data.DataLoader(
            train_dataset_full,
            collate_fn = full_dataset_tf.collater,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=cfg["train"]["num_workers"],
            persistent_workers=False,
            pin_memory=True,
            drop_last=False,
        )

        val_loader_tf = torch.utils.data.DataLoader(
            val_dataset_tf,
            collate_fn = full_dataset_tf.collater,
            batch_size=cfg['train']['batch_size'],
            shuffle=True,
            num_workers=cfg['train']['num_workers'],
            pin_memory=True,
            drop_last=False,
        )
        val_loader_qa = torch.utils.data.DataLoader(
            val_dataset_qa,
            collate_fn = full_dataset_qa.collater,
            batch_size=cfg['train']['batch_size'],
            shuffle=True,
            num_workers=cfg['train']['num_workers'],
            pin_memory=True,
            drop_last=False,
        )
        val_loader_capt = torch.utils.data.DataLoader(
            val_dataset_capt,
            collate_fn = full_dataset_capt.collater,
            batch_size=cfg['train']['batch_size'],
            shuffle=True,
            num_workers=cfg['train']['num_workers'],
            pin_memory=True,
            drop_last=False,
        )

        iters_per_epoch = math.ceil(len(train_dataset_full) / cfg["train"]["batch_size"])

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
            train_stats = train_epoch(
                epoch=epoch,
                iters_per_epoch=iters_per_epoch,
                model=model,
                train_loader=train_loader,
                val_loader=(val_loader_tf, val_loader_qa, val_loader_capt),
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                log_freq=cfg["train"]["log_freq"],
                cuda_enabled=device.type == "cuda",
                accum_grad_iters=cfg["train"]["accum_grad_iters"],
            )
            print(f"Epoch {epoch} Train Stats: {train_stats}")
    
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
        project="grandtour",
        config=cfg,
        name=f"train_full",
    )

    train(cfg)