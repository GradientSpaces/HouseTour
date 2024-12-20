import yaml
import os
from PIL import Image
import torch
from torchvision import transforms
import einops
from torch.amp import autocast as autocast
import logging
import time
import wandb
import argparse
import math
import pynvml
from sklearn.model_selection import train_test_split

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

def train_epoch(
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler = "linear_warmup_cosine_lr",
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
        output_dir=None,
    ):
    use_amp = scaler is not None

    if not hasattr(data_loader, "__next__"):
        # convert to iterator if not already
        data_loader = iter(data_loader)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    
    # if iter-based runner, schedule lr based on inner epoch.
    print(
        "Start training epoch {}, {} iters per inner epoch.".format(
            epoch, iters_per_epoch
        )
    )

    header = "Train: data epoch: [{}]".format(epoch)
    
    if start_iters is None:
        # epoch-based runner
        inner_epoch = epoch
    else:
        # In iter-based runner, we schedule the learning rate based on iterations.
        inner_epoch = start_iters // iters_per_epoch
        header = header + "; inner epoch [{}]".format(inner_epoch)
    
    for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
        print(f"Iteration {i}/{iters_per_epoch}")

        # if using iter-based runner, we stop after iters_per_epoch iterations.
        if i >= iters_per_epoch:
            break
        samples = next(data_loader)
        samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
        samples.update(
            {
                "epoch": inner_epoch,
                "num_iters_per_epoch": iters_per_epoch,
                "iters": i,
            }
        )
        lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = model(samples)["loss"]
        
        # after_train_step()
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # update gradients every accum_grad_iters iterations
        if (i + 1) % accum_grad_iters == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()                     
            else:    
                optimizer.step()
            optimizer.zero_grad()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if is_main_process() and wandb.run is not None:
            wandb.log({'train/loss': loss.item(),
                       'train/lr': optimizer.param_groups[0]["lr"]},
                      step=epoch * iters_per_epoch + i)
        
        # Save every 5000 iterations
        if i % 5000 == 0:
            torch.save(model.state_dict(), output_dir + f"/model_ws_4_e{epoch}_tf_qa_capt_{i}_loss_{loss.item():.2f}.pth")
            print(f"Model saved to model_ws_4_e{epoch}_tf_qa_capt_{i}_loss_{loss.item():.2f}.pth") 
        
    # after train_epoch()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats: " + str(metric_logger.global_avg()))
    return {
        k: "{:.3f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }

def train(cfg):
    start_time = time.time()
    best_agg_metric = 0
    best_epoch = 0

    max_epochs = cfg["run"]["max_epochs"]
    eval_only = cfg["run"]["eval_only"]

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GrandTour.from_config(cfg["model"]).to(device)

    # Load Q&A training weights
    ckpt_path = cfg["model"]["grandtour_model"]
    print("Loading model from ", ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    print("GPU Usage after loading the model...")
    get_gpu_usage()

    if not eval_only:
        print("Start training...")
        # Load the dataset
        dataset = HouseTour_Dataset(
            vis_processor=ImgArrayProcessor(image_size=224),
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

        train_indices, val_indices = train_test_split(
            list(range(len(dataset))),
            test_size=cfg["train"]["val_ratio"],
            shuffle=False,
        )

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
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
            train_stats = train_epoch(
                epoch=epoch,
                iters_per_epoch=iters_per_epoch,
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                log_freq=cfg["train"]["log_freq"],
                cuda_enabled=device.type == "cuda",
                accum_grad_iters=cfg["train"]["accum_grad_iters"],
                output_dir=cfg["run"]["output_dir"]
            )
            print(f"Epoch {epoch} Train Stats: {train_stats}")
    
    # Save the model
    model_name = "model_ws_4_tf_qa_capt"
    torch.save(model.state_dict(), cfg["run"]["output_dir"] + f"/{model_name}.pth")
    print(f"Model saved to {cfg['run']['output_dir']}/{model_name}.pth")
    
    print(f"Total training time: {time.time() - start_time} seconds")
    print("Training complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", required=True)
    args = parser.parse_args()
    cfg_path = args.cfg_path    

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)

    







