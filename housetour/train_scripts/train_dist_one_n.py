import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from socket import gethostname
import argparse
import yaml
import math
import os

from grandtour.models.grandtour import GrandTour
from grandtour.datasets.house_tour_dataset import HouseTour_Dataset
from grandtour.processors.img_array_processors import ImgArrayProcessor
from grandtour.common.optims import LinearWarmupCosineLRScheduler
from grandtour.trainer.trainer import Trainer


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def prepare_dataloader(dataset, batch_size: int):
    return DataLoader(
        dataset,
        collate_fn = dataset.collater,
        batch_size=batch_size,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def load_train_objs(cfg):
    train_dataset = HouseTour_Dataset(
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
    )  # load your dataset
    # Load model
    model = GrandTour.from_config(cfg["model"])

    # Load the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["init_lr"]),
        weight_decay=cfg["train"]["weight_decay"],
    )

    # Load the lr_scheduler
    lr_scheduler = LinearWarmupCosineLRScheduler(
        optimizer = optimizer,
        max_epoch = cfg["run"]["max_epochs"],
        iters_per_epoch = math.ceil(len(train_dataset) / cfg["train"]["batch_size"]),
        min_lr = float(cfg["train"]["min_lr"]),
        init_lr = float(cfg["train"]["init_lr"]),
        warmup_start_lr = float(cfg["train"]["warmup_lr"])
    )

    # Load the scaler
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["use_amp"])
    
    return train_dataset, model, optimizer, lr_scheduler, scaler

def main(rank, world_size, cfg):
    ddp_setup(rank, world_size)

    dataset, model, optimizer, lr_scheduler, scaler = load_train_objs(cfg)
    train_data = prepare_dataloader(dataset, batch_size=cfg["train"]["batch_size"])
    iters_per_epoch = math.ceil(len(train_data) / cfg["train"]["batch_size"])
    trainer = Trainer(
        gpu_id=rank,
        model=model,
        train_data=train_data,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        start_iters=None,
        iters_per_epoch=iters_per_epoch, 
        log_freq=cfg["train"]["log_freq"],
        max_epochs = cfg["run"]["max_epochs"],
        accum_grad_iters=cfg["train"]["accum_grad_iters"],
        save_every= 1,
        snapshot_path=cfg["train"]["snapshot_path"],
        output_dir=cfg["run"]["output_dir"]
    )
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", required=True)
    args = parser.parse_args()
    cfg_path = args.cfg_path    

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, cfg,), nprocs=world_size)