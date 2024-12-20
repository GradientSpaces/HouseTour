import os
import yaml
import torch
import torch.optim as optim
from transformers import LlamaTokenizer, BertConfig
from tqdm import tqdm
import wandb

from grandtour.common.optims import LinearWarmupCosineLRScheduler
from grandtour.models.blip2_pretrain import Blip2Qformer
from grandtour.processors.img_array_processors import ImgArrayProcessor
from grandtour.datasets.house_tour_dataset_pretrain import HouseTour_Dataset_Pretrain
from grandtour.datasets.data_utils import prepare_sample


cfg_path = "grandtour/configs/grandtour_pretrain.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Blip2Qformer.from_config(cfg['model']).to(device)

wandb.init(
    project="grandtour_pretrain",
    config=cfg,
    name=f"multi_image_blip_pretrain_8_vitn8",
)

full_dataset = HouseTour_Dataset_Pretrain(
    vis_processor=ImgArrayProcessor(image_size=cfg["model"]["image_size"]),
    vis_root=cfg["data"]["vis_root"],
    ann_root=cfg["data"]["ann_root"],
    num_video_query_token=cfg["data"]["num_video_query_token"],
    max_txt_len=cfg["data"]["max_txt_len"],
)

train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [0.9, 0.1])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=cfg['train']['batch_size'],
    shuffle=True,
    num_workers=cfg['train']['num_workers'],
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=cfg['train']['batch_size'],
    shuffle=False,
    num_workers=cfg['train']['num_workers'],
    pin_memory=True,
    drop_last=False,
)

def evaluate(model, val_loader):
    model.eval()
    
    running_eval_loss = 0.0
    running_eval_loss_itc = 0.0
    running_eval_loss_itm = 0.0
    running_eval_loss_lm = 0.0
    with torch.no_grad():
        for samples in val_loader:
            samples = prepare_sample(samples, cuda_enabled=True)
            blip_output_eval = model(samples)
            running_eval_loss += blip_output_eval.loss.item()
            running_eval_loss_itc += blip_output_eval.loss_itc.item()
            running_eval_loss_itm += blip_output_eval.loss_itm.item()
            running_eval_loss_lm += blip_output_eval.loss_lm.item()
    
    return running_eval_loss / len(val_loader), running_eval_loss_itc / len(val_loader), running_eval_loss_itm / len(val_loader), running_eval_loss_lm / len(val_loader)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(cfg["train"]["init_lr"]),
    # weight_decay=cfg["train"]["weight_decay"],
)

max_epochs = cfg["run"]["max_epochs"]

lr_scheduler = LinearWarmupCosineLRScheduler(
    optimizer = optimizer,
    max_epoch = max_epochs,
    iters_per_epoch = len(train_loader),
    min_lr = float(cfg["train"]["min_lr"]),
    init_lr = float(cfg["train"]["init_lr"]),
    warmup_start_lr = float(cfg["train"]["warmup_lr"])
)

scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["use_amp"])
accum_grad_iters = cfg["train"]["accum_grad_iters"] 

for epoch in range(max_epochs):
    model.train() 

    running_loss = 0.0
    for batch_idx, samples in enumerate(tqdm(train_loader)):
        samples = prepare_sample(samples, cuda_enabled=True)
        
        lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)

        blip_output = model(samples)

        loss = blip_output.loss
        loss_itc = blip_output.loss_itc
        loss_itm = blip_output.loss_itm
        loss_lm = blip_output.loss_lm

        if cfg['train']['use_amp']:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        

        # update gradients every accum_grad_iters iterations
        if (batch_idx + 1) % accum_grad_iters == 0:
            if cfg['train']['use_amp']:
                scaler.step(optimizer)
                scaler.update()                     
            else:    
                optimizer.step()
            optimizer.zero_grad()
        
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_size = param.grad.data.size()
        #         grad_norm = param.grad.data.norm(2)  # Optional: log the norm of the gradient
        #         print(f"Layer: {name}, Gradient size: {grad_size}, Gradient norm: {grad_norm}")
        #     else:
        #         print(f"Layer: {name}, Gradient is None")

        running_loss += loss.item()

        wandb.log({
            "train/loss": loss.item(),
            "train/loss_itc": loss_itc.item(),
            "train/loss_itm": loss_itm.item(),
            "train/loss_lm": loss_lm.item(),
            "train/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch + 1,
            "step": epoch * len(train_loader) + batch_idx,
        })

        if batch_idx % 50 == 0:  # Print every 50 steps
            print(f"Epoch [{epoch+1}/{max_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}")
        
        torch.cuda.empty_cache()

    print(f"Epoch [{epoch+1}/{max_epochs}] finished, Average Training Loss: {running_loss / len(train_loader)}")

    # Evaluation
    eval_loss, eval_loss_itc, eval_loss_itm, eval_loss_lm = evaluate(model, val_loader)
    print(f"Epoch [{epoch+1}/{max_epochs}] finished, Average Eval Loss: {eval_loss}")

    wandb.log({
        "val/loss": eval_loss,
        "val/loss_itc": eval_loss_itc,
        "val/loss_itm": eval_loss_itm,
        "val/loss_lm": eval_loss_lm,
        "epoch": epoch + 1,
    })

    checkpoint_path = os.path.join(cfg['run']['output_dir'], f"pretrain_epoch_{epoch+1}_{eval_loss:.3f}_w8_vit.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

# Optionally, save the final model after all epochs
final_model_path = os.path.join(cfg['run']['output_dir'], f"pretrain_final.pth")
torch.save(model.state_dict(), final_model_path)
print("Final model saved as final_model.pth")
