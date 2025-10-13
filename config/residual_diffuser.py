import socket
import os

from diffuser.utils import watch

#------------------------ base ------------------------#

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
]

DATA_DIR = 'data/'
DATASET_PATH = 'trajectories.jsonl'
DATASET_SCALES_PATH = 'scene_scale.json'

base = {

    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 512,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': True,
        'dim_mults': (1, 4, 8),
        'renderer': None,

        ## dataset
        'data_dir': DATA_DIR,
        'dataset_path' : os.path.join(DATA_DIR, DATASET_PATH),
        'dataset_scales_path' : os.path.join(DATA_DIR, DATASET_SCALES_PATH),
        'loader': None,
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'training_checkpoints',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 60000,
        'loss_type': 'spline',
        'n_train_steps': 6e4,
        'batch_size': 1,
        'learning_rate': 5e-6,
        'gradient_accumulate_every': 8,
        'ema_decay': 0.995,
        'save_freq': 2000,
        'sample_freq': 1000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda',

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## dataset
        'data_dir': DATA_DIR,
        'dataset_path' : os.path.join(DATA_DIR, DATASET_PATH),
        'dataset_scales_path' : os.path.join(DATA_DIR, DATASET_SCALES_PATH),

        ## serialization
        'vis_freq': 10,
        'logbase': 'training_checkpoints',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',
        'conditional': False,

        ## loading
        # 'diffusion_loadpath': 'f:{logbase}/residual_diffuser/diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_loadpath': 'f:checkpoints/residual-diffuser/',
        'diffusion_epoch': 'latest',
    },

}

#------------------------ overrides ------------------------#

residual_diffuser = {
    'diffusion': {
        'horizon': 384,
        'n_diffusion_steps': 16,
    },
    'plan': {
        'horizon': 384,
        'n_diffusion_steps': 16,
    },
}