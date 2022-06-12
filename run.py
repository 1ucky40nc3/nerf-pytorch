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


from data.blender import load_blender_data
from data.deepvoxels import load_dv_data
from data.LINEMOD import load_LINEMOD_data
from data.llff import load_llff_data

from utils import config_parser
from utils import args2dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def main():
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(
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

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(
            args.datadir, args.half_res, args.testskip)

        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':
        images, poses, render_poses, hwf, i_split = load_dv_data(
            scene=args.shape, basedir=args.datadir, testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

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

    # Create nerf model
    create_method = create_nerf
    if args.hashenc:
        create_method = create_nerfwithhash
    elif args.instant_ngp:
        create_method = create_instantngp
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_method(args)
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

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

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


    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

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