import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import wandb
import logging

from grandtour.common.dist_utils import is_main_process
from grandtour.common.logger import MetricLogger, SmoothedValue

class Trainer:
    def __init__(
        self,
        gpu_id,
        model,
        train_data,
        optimizer,
        lr_scheduler,
        scaler,
        start_iters,
        iters_per_epoch,
        log_freq,
        max_epochs,
        accum_grad_iters,
        output_dir,
        save_every: int = 1,
        snapshot_path: str = "ckpt/model_lora.pth",
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.start_iters = start_iters
        self.iters_per_epoch = iters_per_epoch
        self.log_freq = log_freq
        self.max_epochs = max_epochs
        self.accum_grad_iters = accum_grad_iters
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.output_dir = output_dir
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_epoch(
        self,
        epoch
    ):
        use_amp = self.scaler is not None
        if not hasattr(self.train_data, "__next__"):
            # convert to iterator if not already
            self.train_data = iter(self.train_data)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, self.iters_per_epoch
            )
        )

        header = "Train: data epoch: [{}]".format(epoch)

        if self.start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)
        
        for i in metric_logger.log_every(range(self.iters_per_epoch), self.log_freq, header):            
            print(f"Iteration {i} of {self.iters_per_epoch}...")

            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= self.iters_per_epoch:
                break
            
            samples = next(self.train_data)
            samples = {key: (value.to(self.gpu_id) if isinstance(value, torch.Tensor) else value) for key, value in samples.items()}
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": self.iters_per_epoch,
                    "iters": i,
                }
            )
            self.lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.model(samples)["loss"]

            # after_train_step()
            if use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (i + 1) % self.accum_grad_iters == 0:
                if use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()                     
                else:    
                    self.optimizer.step()
                self.optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            if is_main_process() and wandb.run is not None:
                wandb.log({'train/loss': loss.item(),
                           'train/lr': self.optimizer.param_groups[0]["lr"]},
                          step=epoch * self.iters_per_epoch + i)
            
            # after train_epoch()
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self):
        for epoch in range(self.max_epochs):
            train_stats = self._run_epoch(epoch)

            logging.info(f"Epoch {epoch} Train Stats: {train_stats}")

            if is_main_process() and wandb.run is not None:
                wandb.log(train_stats, step=epoch)

            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
        
        if self.gpu_id == 0:
            # Save the model
            model_name = "model_ws_4_e10"
            torch.save(model.state_dict(), self.output_dir + f"/{model_name}.pth")
            print(f"Model saved to {self.output_dir}/{model_name}.pth")

            print("Training complete")
