import os
import json

import numpy as np

import torch
import torch.nn.functional as F

import cv2
import imageio 


def trans_t(t):
    return torch.Tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]]).float()


def rot_phi(phi):
    sp = np.sin(phi)
    cp = np.cos(phi)
    rot = torch.Tensor([
        [1,  0,   0, 0],
        [0, cp, -sp, 0],
        [0, sp,  cp, 0],
        [0,  0,   0, 1]]).float()
    return rot


def rot_theta(th):
    st = np.sin(th)
    ct = np.cos(th)
    rot = torch.Tensor([
        [ct, 0, -st, 0],
        [ 0, 1,   0, 0],
        [ct, 0,  ct, 0],
        [ 0, 0,   0, 1]]).float()
    return rot


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([
            [-1, 0, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        path = os.path.join(
            basedir, 
            'transforms_{}.json'.format(s))
        with open(path, 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs, all_poses = [], []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs, poses = [], []

        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            img = imageio.imread(fname)
            pose = np.array(frame['transform_matrix'])
            imgs.append(img)
            poses.append(pose)

        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [
        np.arange(counts[i], counts[i+1]) 
        for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    angles = np.linspace(-180, 180, 40 + 1)[:-1]
    render_poses = torch.stack([
        pose_spherical(angle, -30.0, 4.0) 
        for angle in angles
    ], dim=0)
    
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        tmp = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            tmp[i] = cv2.resize(
                img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = tmp
        
    return imgs, poses, render_poses, [H, W, focal], i_split


