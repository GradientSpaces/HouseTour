import json
import numpy as np
import os
import pdb
from scipy.spatial.transform import Rotation as R
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time

import diffuser.utils as utils
from diffuser.datasets.cam_traj import CameraTrajectoriesDataset

class Parser(utils.Parser):
    dataset: str = 'residual_diffuser'
    config: str = 'config.residual_diffuser'

def cycle(dl):
    while True:
        for data in dl:
            yield data

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

#---------------------------------- loading ----------------------------------#

path_to_traj = os.path.join(args.logbase, args.dataset, "diffusion","H384_T16")

model_config = utils.load_config(path_to_traj, 'model_config.pkl')
diffusion_config = utils.load_config(path_to_traj, 'diffusion_config.pkl')
trainer_config = utils.load_config(path_to_traj, 'trainer_config.pkl')

dataset = CameraTrajectoriesDataset(args.dataset_path)
model = model_config()
diffusion = diffusion_config(model)
trainer = trainer_config(diffusion, dataset, renderer=None)
trainer.logdir = path_to_traj

epoch = utils.get_latest_epoch(path_to_traj)
print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')
trainer.load(epoch)

trainer.model.eval()
diffusion = trainer.model

#---------------------------------- test ----------------------------------#

# read test set
with open(path_to_traj + '/test_indices.txt', 'r') as f:
    val_indices = f.read()
    val_indices = [int(i) for i in val_indices.split('\n') if i]

val_data = torch.utils.data.Subset(dataset, val_indices)
val_dataloader = cycle(torch.utils.data.DataLoader(
    val_data, batch_size=1, num_workers=1, pin_memory=True, shuffle=False
))

val_losses = []
infos_list = []

lin_int_losses = []
lin_int_infos_list = []

catmull_losses = []
catmull_infos_list = []

times = []

for _ in tqdm(range(len(val_indices))):
    start_time = time()
    batch = next(val_dataloader)
    batch = utils.batch_to_device(batch)

    if batch.scene_id.item() in [227, 202, 933, 223, 1148]:
        continue

    traj = diffusion.forward(batch.conditions, horizon=batch.trajectories.shape[1])
    loss, infos = diffusion.loss_fn(traj, batch.trajectories, batch.conditions, scene_id=batch.scene_id, norm_params=batch.scale)
    
    val_losses.append(loss.item())
    infos_list.append(infos)

    (lin_int_loss, lin_int_infos), lin_int_traj = trainer.linear_interpolation_loss(
        batch.trajectories, batch.conditions, diffusion.loss_fn, scene_id=batch.scene_id, norm_params=batch.scale,
    )
    lin_int_losses.append(lin_int_loss.item())
    lin_int_infos_list.append(lin_int_infos)
    
    (catmull_loss, catmull_infos), catmull_traj = trainer.catmull_rom_loss(
        batch.trajectories, batch.conditions, diffusion.loss_fn, scene_id=batch.scene_id, norm_params=batch.scale,
    )
    catmull_losses.append(catmull_loss.item())
    catmull_infos_list.append(catmull_infos)

    times.append(time() - start_time)

    
avg_val_loss = np.mean(val_losses)
val_infos = {key: np.mean([info[key] for info in infos_list]) for key in infos_list[0].keys()}

avg_lin_int_loss = np.mean(lin_int_losses)
lin_int_infos = {key: np.mean([info[key] for info in lin_int_infos_list]) for key in lin_int_infos_list[0].keys()}

avg_catmull_loss = np.mean(catmull_losses)
catmull_infos = {key: np.mean([info[key] for info in catmull_infos_list]) for key in catmull_infos_list[0].keys()}

infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in val_infos.items()])
lin_int_infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in lin_int_infos.items()])
catmull_infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in catmull_infos.items()])

print(f'Loss: {avg_val_loss:8.4f}')
print(infos_str)

print(f'Linear Interpolation Loss: {avg_lin_int_loss:8.4f}')
print(lin_int_infos_str)

print(f'Catmull Rom Loss: {avg_catmull_loss:8.4f}')
print(catmull_infos_str)

print(f'Average Time: {np.mean(times):8.4f}')
print(f'Standard Deviation Time: {np.std(times):8.4f}')
