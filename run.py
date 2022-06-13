import os
import sys
import json
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from render import volumetric_rendering, render_path
import models

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

import datasets
from datasets.blender import load_data as load_blender_data
from datasets.deepvoxels import load_data as load_dv_data
from datasets.LINEMOD import load_data as load_LINEMOD_data
from datasets.llff import load_data as load_llff_data

from utils import config_parser
from utils import args2dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def main():
    parser = config_parser()
    args = parser.parse_args()

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    exp_dir = os.path.join(basedir, expname)
    os.makedirs(exp_dir, exist_ok=True)

    args_file = os.path.join(exp_dir, 'args.txt')
    with open(args_file, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        config_file = os.path.join(exp_dir, 'config.txt')
        with open(config_file, 'w') as file:
            file.write(open(args.config, 'r').read())

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = models.load(args)
    global_step = start

    bds_dict = {'near': near, 'far': far}
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    config = args2dict(args)
    config["render_train"] = render_kwargs_train
    config["render_test"] = render_kwargs_test

    wandb.init(
        project="nerf-pytorch",
        dir=args.basedir,
        tags=args.expname.split("_"),
        config=config)

    images, poses, render_poses, rays_rgb, hwf, near, far, K, i_train, i_val, i_test = datasets.load(args)
    H, W, focal = hwf

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(
                basedir, 
                expname, 
                'renderonly_{}_{:06d}'.format(
                    'test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(
                render_poses, 
                hwf, 
                K, 
                args.chunk, 
                render_kwargs_test, 
                gt_imgs=images, 
                savedir=testsavedir, 
                render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            test_video = os.path.join(testsavedir, 'video.mp4')
            imageio.mimwrite(test_video, to8b(rgbs), fps=30, quality=8)

            return

    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        batch_rays, target_s, i_batch = datasets.get(
            args, images, poses, rays_rgb, hwf, K, i_train, 
            i_batch=i_batch, i=i, start=start)
        
        #####  Core optimization loop  #####
        rgb, disp, acc, extras = volumetric_rendering(
            H, W, K, 
            chunk=args.chunk, 
            rays=batch_rays,
            verbose=i < 10, retraw=True,
            **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate**(global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        ################################

        dt = time.time() - time0

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            network_fn_state_dict = render_kwargs_train['network_fn'].state_dict()
            network_fine_state_dict = render_kwargs_train['network_fine'].state_dict() if render_kwargs_train['network_fine'] else None
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': network_fn_state_dict,
                'network_fine_state_dict': network_fine_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            with torch.no_grad():
                rgbs, disps = render_path(
                    render_poses, 
                    hwf, 
                    K, 
                    args.chunk, 
                    render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            rgb_video, disp_video = moviebase + 'rgb.mp4', moviebase + 'disp.mp4'
            imageio.mimwrite(rgb_video, to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(disp_video, to8b(disps / np.max(disps)), fps=30, quality=8)
            wandb.log({"rgb_video": wandb.Video(rgb_video), "disp_video": wandb.Video(disp_video)}, step=i)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(
                    torch.Tensor(poses[i_test]).to(device), 
                    hwf, 
                    K, 
                    args.chunk, 
                    render_kwargs_test, 
                    gt_imgs=images[i_test], 
                    savedir=testsavedir,
                    step=i)
            print('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            wandb.log(dict(train_loss=loss.item(), train_psnr=psnr.item()), step=i)

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type(
        'torch.cuda.FloatTensor')

    main()