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
from run_nerf import create_instantngp, create_nerf, create_nerfwithhash

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


from datasets import blender
from datasets import llff
from datasets import deepvoxels
from datasets import LINEMOD

from utils import config_parser
from utils import args2dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def load_blender(args):
    images, poses, render_poses, hwf, i_split = blender.load_data(
        args.datadir, args.half_res, args.testskip)
    print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_test = i_split

    near = 2.
    far = 6.

    if args.white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    else:
        images = images[..., :3]
    
    return images, poses, render_poses, hwf, near, far, None, i_train, i_val, i_test


def load_deepvoxels(args):
    images, poses, render_poses, hwf, i_split = deepvoxels.load_data(
        scene=args.shape, basedir=args.datadir, testskip=args.testskip)

    print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_test = i_split

    hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
    near = hemi_R - 1.
    far = hemi_R + 1.

    return images, poses, render_poses, hwf, near, far, None, i_train, i_val, i_test


def load_LINEMOD(args):
    images, poses, render_poses, hwf, K, i_split, near, far = LINEMOD.load_data(
        args.datadir, args.half_res, args.testskip)

    print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
    print(f'[CHECK HERE] near: {near}, far: {far}.')
    i_train, i_val, i_test = i_split

    if args.white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    else:
        images = images[..., :3]

    return images, poses, render_poses, hwf, near, far, K, i_train, i_val, i_test



def load_llff(args):
    images, poses, bds, render_poses, i_test = llff.load_data(
        args.datadir, args.factor, recenter=True, bd_factor=.75, spherify=args.spherify)

    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if args.llffhold > 0:
        print('Auto LLFF holdout,', args.llffhold)
        i_test = np.arange(images.shape[0])[::args.llffhold]

    i_val = i_test
    i_train = np.array(
        [i for i in np.arange(int(images.shape[0])) 
            if (i not in i_test and i not in i_val)])

    print('DEFINING BOUNDS')
    if args.no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
    else:
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)
    return images, poses, render_poses, hwf, near, far, None, i_train, i_val, i_test


def load(args):
    if args.dataset_type == 'llff':
        load_fn = load_llff
    elif args.dataset_type == 'blender':
        load_fn = load_blender
    elif args.dataset_type == 'LINEMOD':
        load_fn = load_LINEMOD
    elif args.dataset_type == 'deepvoxels':
        load_fn = load_deepvoxels
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    images, poses, render_poses, hwf, near, far, K, i_train, i_val, i_test = load_fn(args)
    
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])
    render_poses = torch.Tensor(render_poses).to(device)

        # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)                # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:, None]], 1)                                  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])                                     # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)                                 # train images only
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])                                            # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    return images, poses, render_poses, rays_rgb, hwf, near, far, K, i_train, i_val, i_test


def get(args, images, poses, rays_rgb, hwf, K, i_train, i_batch=0, i=0, start=0):
    N_rand = args.N_rand
    use_batching = not args.no_batching
    H, W, _ = hwf

    # Sample random ray batch
    if use_batching:
        # Random over all images
        batch = rays_rgb[i_batch:i_batch+N_rand]                                           # [B, 2+1, 3*?]
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]

        i_batch += N_rand
        if i_batch >= rays_rgb.shape[0]:
            print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            i_batch = 0
    else:
        # Random from one image
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3,:4]

        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))                         # (H, W, 3), (H, W, 3)

            if i < args.precrop_iters:
                dH = int(H // 2 * args.precrop_frac)
                dW = int(W // 2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH), 
                        torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2 * dH} x {2 * dW} "
                            "is enabled until iter {args.precrop_iters}")                
            else:
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, H-1, H), 
                        torch.linspace(0, W-1, W)
                ), dim=-1)                                                                 # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])                                         # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()                                     # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]                      # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]                      # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]                    # (N_rand, 3)

    return batch_rays, target_s, i_batch