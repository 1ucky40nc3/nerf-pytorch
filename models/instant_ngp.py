import os
import sys
import json
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

import imageio

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import wandb

from run_nerf_helpers import to8b
from run_nerf_helpers import mse2psnr
from run_nerf_helpers import img2mse
from run_nerf_helpers import get_embedder
from run_nerf_helpers import NeRF
from run_nerf_helpers import get_rays
from run_nerf_helpers import get_rays_np
from run_nerf_helpers import ndc_rays
from run_nerf_helpers import sample_pdf

from datasets.blender import load_data as load_blender_data
from datasets.deepvoxels import load_data as load_dv_data
from datasets.LINEMOD import load_data as load_LINEMOD_data
from datasets.llff import load_data as load_llff_data

from utils import config_parser
from utils import args2dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def create_instantngp(args):
    """Instantiate instant-ngp components."""
    model_config = {
        "encoding_config": {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 2.0
        },
        "network_config": {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 6
        }
    }
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,
        n_output_dims=4,
        **model_config).to(device)
    grad_vars = list(model.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(
        inputs, viewdirs, network_fn,
        embed_fn=lambda x: x,
        embeddirs_fn=lambda x: x,
        netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f) 
            for f in sorted(
                os.listdir(os.path.join(basedir, expname))) 
            if 'tar' in f
        ]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : 0,
        'network_fine' : None,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : False,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer