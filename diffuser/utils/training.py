import os
import copy
import numpy as np
import torch
import einops
import pdb
from tqdm import tqdm
import wandb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from ..models.helpers import catmull_rom_spline_with_rotation

def cycle(dl):
    while True:
        for data in dl:
            yield data

def assert_no_nan_weights(model):
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN detected in parameter: {name}"

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        dataset_size = len(self.dataset)

        # Read the indices from the .txt file
        with open(os.path.join(results_folder, 'train_indices.txt'), 'r') as f:
            self.train_indices = f.read()
            self.train_indices = [int(i) for i in self.train_indices.split('\n') if i]
        
        with open(os.path.join(results_folder, 'val_indices.txt'), 'r') as f:
            self.val_indices = f.read()
            self.val_indices = [int(i) for i in self.val_indices.split('\n') if i]


        self.train_dataset = torch.utils.data.Subset(self.dataset, self.train_indices)
        self.val_dataset = torch.utils.data.Subset(self.dataset, self.val_indices)
        self.train_dataloader = cycle(torch.utils.data.DataLoader(
            self.train_dataset, batch_size=train_batch_size, num_workers=1, pin_memory=True, shuffle=False
        ))

        self.val_dataloader = cycle(torch.utils.data.DataLoader(
            self.val_dataset, batch_size=train_batch_size, num_workers=1, pin_memory=True, shuffle=False
        ))

        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer

        

        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

        self.log_to_wandb = False

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        # Save the indices as .txt files
        with open(os.path.join(self.logdir, 'train_indices.txt'), 'w') as f:
            for idx in self.train_indices:
                f.write(f"{idx}\n")
        with open(os.path.join(self.logdir, 'val_indices.txt'), 'w') as f:
            for idx in self.val_indices:
                f.write(f"{idx}\n")
        
        timer = Timer()
        torch.autograd.set_detect_anomaly(True)

        # Setup wandb
        if self.log_to_wandb:
            wandb.init(
                project='trajectory-generation', 
                config={'lr': self.optimizer.param_groups[0]['lr'], 'batch_size': self.batch_size, 'gradient_accumulate_every': self.gradient_accumulate_every},
            )

        for step in tqdm(range(n_train_steps)):
            
            mean_train_loss = 0.0
            for i in range(self.gradient_accumulate_every):
                batch = next(self.train_dataloader)
                batch = batch_to_device(batch)
                
                loss, infos = self.model.loss(x=batch.trajectories, cond=batch.conditions)
                loss = loss / self.gradient_accumulate_every
                mean_train_loss += loss.item()
                loss.backward()

            if self.log_to_wandb:
                wandb.log({
                    'step': self.step,
                    'train/loss': mean_train_loss
                })

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.optimizer.zero_grad()

            assert_no_nan_weights(self.model)

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step
                print(f'Saving model at step {self.step}...')
                self.save(label)

            if self.step % self.log_freq == 0:
                val_losses = []
                lin_int_losses = []

                val_infos_list = []
                lin_int_infos_list = []

                catmull_losses = []
                catmull_infos_list = []

                for _ in range(len(self.val_indices)):
                    val_batch = next(self.val_dataloader)
                    val_batch = batch_to_device(val_batch)

                    traj = self.model.forward(val_batch.conditions, horizon=val_batch.trajectories.shape[1])
                    val_loss, val_infos = self.model.loss_fn(traj, val_batch.trajectories, cond=val_batch.conditions)

                    val_losses.append(val_loss.item())
                    val_infos_list.append({key: val for key, val in val_infos.items()})
                    

                    (lin_int_loss, lin_int_infos), lin_int_traj = self.linear_interpolation_loss(
                        val_batch.trajectories, val_batch.conditions, self.model.loss_fn
                    )
                    lin_int_losses.append(lin_int_loss.item())
                    lin_int_infos_list.append({key: val for key, val in lin_int_infos.items()})

                    (catmull_loss, catmull_infos), catmull_traj = self.catmull_rom_loss(
                        val_batch.trajectories, val_batch.conditions, self.model.loss_fn
                    )

                    catmull_losses.append(catmull_loss.item())
                    catmull_infos_list.append(catmull_infos)
                
                avg_val_loss = np.mean(val_losses)
                avg_lin_int_loss = np.mean(lin_int_losses)

                val_infos = {key: np.mean([info[key] for info in val_infos_list]) for key in val_infos_list[0].keys()}
                lin_int_infos = {key: np.mean([info[key] for info in lin_int_infos_list]) for key in lin_int_infos_list[0].keys()}

                avg_catmull_loss = np.mean(catmull_losses)
                catmull_infos = {key: np.mean([info[key] for info in catmull_infos_list]) for key in catmull_infos_list[0].keys()}

                val_infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in val_infos.items()])
                lin_int_infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in lin_int_infos.items()])
                catmull_infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in catmull_infos.items()])
                

                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print("Learning Rate: ", self.optimizer.param_groups[0]['lr'])
                print(f'Step {self.step}: {loss * self.gradient_accumulate_every:8.4f} | {infos_str} | t: {timer():8.4f}')
                print(f'Validation - {self.step}: {avg_val_loss:8.4f} | {val_infos_str} | t: {timer():8.4f}')
                print(f'Linear Interpolation Loss - {self.step}: {avg_lin_int_loss:8.4f} | {lin_int_infos_str} | t: {timer():8.4f}')
                print(f'Catmull Rom Loss - {self.step}: {avg_catmull_loss:8.4f} | {catmull_infos_str} | t: {timer():8.4f}')
                print()

                if self.log_to_wandb:
                    wandb.log({
                        'step': self.step,
                        'val/loss': avg_val_loss,
                        'val/linear_interp/loss': avg_lin_int_loss,
                        'val/linear_interp/quaternion dist.': lin_int_infos['quat. dist.'],
                        'val/linear_interp/euclidean dist.': lin_int_infos['trans. error'],
                        'val/linear_interp/geodesic loss': lin_int_infos['geodesic dist.'],
                        'val/catmull_rom/loss': avg_catmull_loss,
                        'val/catmull_rom/quaternion dist.': catmull_infos['quat. dist.'],
                        'val/catmull_rom/euclidean dist.': catmull_infos['trans. error'],
                        'val/catmull_rom/geodesic loss': catmull_infos['geodesic dist.'],
                        'val/quaternion dist.': val_infos['quat. dist.'],
                        'val/euclidean dist.': val_infos['trans. error'],
                        'val/geodesic loss': val_infos['geodesic dist.'],
                    })

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]


            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)

    def linear_interpolation_loss(self, trajectories, conditions, loss_fc, scene_id=None, norm_params=None):
        batch_size, horizon, transition = trajectories.shape

        # Extract known indices and values
        known_indices = np.array(list(conditions.keys()), dtype=int)
        # candidate_no x batch_size x dim
        known_values = np.stack([c.cpu().numpy() for c in conditions.values()], axis=0)
        known_values = np.moveaxis(known_values, 0, 1)

        # Create time steps for interpolation
        time_steps = np.linspace(0, horizon, num=horizon)

        # Perform interpolation across all dimensions at once
        linear_int_arr = np.array([[
            np.interp(time_steps, known_indices, known_values[b, :, dim])
            for dim in range(transition)] 
            for b in range(batch_size)]
        ).T  # Transpose to match shape (horizon, transition)

        # Convert to tensor and move to the same device as trajectories
        linear_int_arr = np.transpose(linear_int_arr, axes=[2, 0, 1])
        linear_int_tensor = torch.tensor(linear_int_arr, dtype=torch.float64, device=trajectories.device)
        
        return loss_fc(linear_int_tensor, trajectories, cond=conditions, scene_id=scene_id, norm_params=norm_params), linear_int_tensor       


    def catmull_rom_loss(self, trajectories, conditions, loss_fc, scene_id=None, norm_params=None):
        '''
            loss for catmull-rom interpolation
        '''        
    
        batch_size, horizon, transition = trajectories.shape

        # Extract known indices and values
        known_indices = np.array(list(conditions.keys()), dtype=int)
        # candidate_no x batch_size x dim
        known_values = np.stack([c.cpu().numpy() for c in conditions.values()], axis=0)
        known_values = np.moveaxis(known_values, 0, 1)

        # Sort the timepoints
        sorted_indices = np.argsort(known_indices)
        known_indices = known_indices[sorted_indices]
        known_values = known_values[:, sorted_indices]

        spline_points = np.array([catmull_rom_spline_with_rotation(known_values[b], known_indices, horizon) for b in range(batch_size)])
        
        # Convert to tensor and move to the same device as trajectories
        spline_points = torch.tensor(spline_points, dtype=torch.float64, device=trajectories.device)

        assert spline_points.shape == trajectories.shape, f"Shape mismatch: {spline_points.shape} != {trajectories.shape}"

        return loss_fc(spline_points, trajectories, cond=conditions, scene_id=scene_id, norm_params=norm_params), spline_points
        




        



          
        
          
