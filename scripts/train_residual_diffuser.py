import numpy as np
from time import time
import logging

import diffuser.utils as utils
from diffuser.datasets.cam_traj import CameraTrajectoriesDataset

class Parser(utils.Parser):
    dataset: str = 'residual_diffuser'
    config: str = 'config.residual_diffuser'

args = Parser().parse_args('diffusion')
dataset = CameraTrajectoriesDataset(args.dataset_path)

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=7,
    cond_dim=8,
    dim_mults=args.dim_mults,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=7,
    action_dim=8,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    scales_path=args.dataset_scales_path,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)
model = model_config()
diffusion = diffusion_config(model)
trainer = trainer_config(diffusion, dataset, renderer=None)

logging.info('Testing forward...')
batch = utils.batchify(dataset[2])
loss, _ = diffusion.loss(x=batch.trajectories, cond=batch.conditions)
loss.backward()
logging.info('✓')

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    logging.info(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)


if __name__ == '__main__':
    main()