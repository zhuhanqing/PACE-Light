'''
 # @ Author: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Create Time: 2024-10-20 15:19:47
 # @ Modified by: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Modified time: 2024-10-29 12:40:26
 # @ Description:
 '''

import logging
from typing import Dict, Tuple
import numpy as np
import os
import torch
import torch.nn as nn
from pyutils.datasets import get_dataset
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.types import Device
from pyutils.config import configs
from pyutils.optimizer.sam import SAM
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.loss import AdaptiveLossSoft, KLLossMixed
from .utils import (
    ComplexL1Loss,
    ComplexMSELoss,
    ComplexTVLoss,
    # CurlLoss,
    DivergenceLoss,
    MatReader,
    PoyntingLoss,
    UnitGaussianNormalizer,
)

from core.models import *
from core.datasets import (
    MNISTDataset,
    FashionMNISTDataset,
    CIFAR10Dataset,
    SVHNDataset,
    MMIDataset,
)

__all__ = [
    "make_dataloader",
    "make_model",
    "make_weight_optimizer",
    "make_arch_optimizer",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]

from torch.nn.functional import pad

def custom_collate_fn(batch, field_pad_value=0.0, eps_pad_value=1.0):
    """
    Generate a collate_fn that pads the fields to the same size and creates a mask for the original data area.
    """
    raise NotImplementedError("This function is not implemented yet")
    wavelengths, grid_steps, eps, input_modes, fields, mask = zip(*batch)
    max_height = max(tensor.shape[2] for tensor in fields)
    max_width = max(tensor.shape[3] for tensor in fields)
    
    # Round up to the nearest multiple of 16
    n_ports = wavelengths[0].shape[0]
    max_height = ((max_height + 15) // 16) * 16
    max_width = ((max_width + 15) // 16) * 16
    
    # max_height = 160
    # max_width = 848
    # Initialize lists for padded tensors
    eps_padded, input_modes_padded, fields_padded = [], [], []
    masks = []  # Single mask list for all
    
    for eps_item, input_mode_item, field_item in zip(eps, input_modes, fields):
        # logging.error('check ori shape %s %s %s', eps_item.shape, input_mode_item.shape, field_item.shape)
        padding = (max_height - field_item.shape[2], max_width - field_item.shape[3])
        lower, upper = padding[0] // 2, padding[0] - padding[0] // 2
        left, right = padding[1] // 2, padding[1] - padding[1] // 2
        padding = (left, right, lower, upper)
        # print(padding)
        # Pad tensors
        eps_padded.append(pad(eps_item, padding, value=eps_pad_value))
        input_modes_padded.append(pad(input_mode_item, padding, value=field_pad_value))
        fields_padded.append(pad(field_item, padding, value=field_pad_value))
        
        # Create a mask with the same shape as the padded tensors
        mask = torch.zeros(size=(n_ports, 1, max_height, max_width), dtype=torch.bool)
        # print(mask.sum())
        # print(lower, upper, left, right)
        mask[:, :, lower:max_height+lower, left:max_width+left].fill_(True)  # Fill the original data area with ones
        
        if mask.sum() == 0:
            logging.error("capture error")
            logging.error('check ori shape %s %s %s', eps_item.shape, input_mode_item.shape, field_item.shape)
            logging.error('check mask %s %s %s', max_height, max_width, n_ports)
            logging.error('check padding %s %s %s %s', lower, upper, left, right)
            logging.error('check mask sum %s', mask.sum(dim=(1,2,3)))
            raise ValueError("Mask is all zeros")
        # print(mask.sum())
        # logging.error('check padding %s %s %s %s', lower, upper, left, right)
        # logging.error('check mask sum %s', mask.sum(dim=(1,2,3)))
        masks.append(mask)
        
    # Stack tensors and the mask
    wavelengths = torch.stack(wavelengths)
    grid_steps = torch.stack(grid_steps)
    eps = torch.stack(eps_padded)
    input_modes = torch.stack(input_modes_padded)
    fields = torch.stack(fields_padded)
    mask = torch.stack(masks)  # Single mask for all
    
    epsilon_min = 1.0
    epsilon_max = 12.3
    eps = (eps - epsilon_min) / (epsilon_max - epsilon_min)

    return wavelengths, grid_steps, eps, input_modes, fields, mask


def custom_collate_fn2(batch):
    """
    Generate a collate_fn that pads the fields to the same size and creates a mask for the original data area.
    """
    raise NotImplementedError("This function is not implemented yet")
    wavelengths, grid_steps, eps, input_modes, fields, mask = zip(*batch)

    # normalize the eps to adapt to Jiaqi's codebase
    epsilon_min = 1.0
    epsilon_max = 12.3
    
    eps_norm = []
    for eps_ori in (eps):
        eps_ori = (eps_ori - epsilon_min) / (epsilon_max - epsilon_min)
        
        eps_norm.append(eps_ori)
    
    eps = torch.stack(eps_norm)
    
    wavelengths = torch.stack(wavelengths)
    grid_steps = torch.stack(grid_steps)
    input_modes = torch.stack(input_modes)
    fields = torch.stack(fields)
    mask = torch.stack(mask)  # Single mask for all

    return wavelengths, grid_steps, eps, input_modes, fields, mask


def make_dataloader(name: str = None, splits=["train", "valid", "test"]) -> Tuple[DataLoader, DataLoader]:
    name = (name or configs.dataset.name).lower()
    if name == "fashionmnist":
        train_dataset, validation_dataset, test_dataset = (
            FashionMNISTDataset(
                root=configs.dataset.root,
                split=split,
                train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                center_crop=configs.dataset.center_crop,
                resize=configs.dataset.img_height,
                resize_mode=configs.dataset.resize_mode,
                binarize=False,
                n_test_samples=configs.dataset.n_test_samples,
                n_valid_samples=configs.dataset.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "cifar10":
        train_dataset, validation_dataset, test_dataset = (
            CIFAR10Dataset(
                root=configs.dataset.root,
                split=split,
                train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                center_crop=configs.dataset.center_crop,
                resize=configs.dataset.img_height,
                resize_mode=configs.dataset.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=configs.dataset.n_test_samples,
                n_valid_samples=configs.dataset.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "svhn":
        train_dataset, validation_dataset, test_dataset = (
            SVHNDataset(
                root=configs.dataset.root,
                split=split,
                train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                center_crop=configs.dataset.center_crop,
                resize=configs.dataset.img_height,
                resize_mode=configs.dataset.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=configs.dataset.n_test_samples,
                n_valid_samples=configs.dataset.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "mmi":
        train_dataset, validation_dataset, test_dataset = (
            MMIDataset(
                root=configs.dataset.root,
                split=split,
                test_ratio=configs.dataset.test_ratio,
                train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                pol_list=configs.dataset.pol_list,
                processed_dir=configs.dataset.processed_dir,
                # resize params and normalize
                device_list=configs.dataset.device_list,
                normalize= configs.dataset.normalize,
                resize=configs.dataset.resize,
                resize_size=[configs.dataset.img_height, configs.dataset.img_width],
                resize_mode=configs.dataset.resize_mode,
                # resize_style=configs.dataset.resize_style,
                data_ratio=configs.dataset.data_ratio,
            )
            if split in splits else None
            for split in ["train", "valid", "test"]
        )
        # print("check dataset size: train %s" % (len(train_dataset)))
        # print("check dataset size: valid %s" % (len(validation_dataset)))
        # print("check dataset size: test %s" % (len(test_dataset)))
        
    elif name == "ns_2d":
        ntrain, ntest = 10, 10
        r = 5
        h = int(((256 - 1)/r) + 1)
        s = h
        reader = MatReader(os.path.join(configs.dataset.root, "navier_stokes/ns_data.mat"))
        x_train = reader.read_field('a')[:ntrain,::r,::r][:,:s,:s]
        y_train = reader.read_field('u')[:ntrain,::r,::r][:,:s,:s]

        x_test = reader.read_field('a')[-ntest:,::r,::r][:,:s,:s]
        y_test = reader.read_field('u')[-ntest:,::r,::r][:,:s,:s]

        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)
        y_train = y_train[..., -1][:,np.newaxis,...]
        y_test = y_test[..., -1][:,np.newaxis,...]
        x_train = x_train.reshape(ntrain,1,s,s)
        x_test = x_test.reshape(ntest,1,s,s)

        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        validation_dataset = None
    else:
        train_dataset, test_dataset = get_dataset(
            name,
            configs.dataset.img_height,
            configs.dataset.img_width,
            dataset_dir=configs.dataset.root,
            transform=configs.dataset.transform,
        )
        validation_dataset = None
    if train_dataset and validation_dataset and test_dataset:
        print(f"check dataset size: train {len(train_dataset)}, valid {len(validation_dataset)}, test {len(test_dataset)}")
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.run.batch_size,
        shuffle=int(configs.dataset.shuffle),
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
        collate_fn=custom_collate_fn if not configs.dataset.resize else None,
    ) if train_dataset is not None else None

    validation_loader = (
        torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=configs.run.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=configs.dataset.num_workers,
            collate_fn=custom_collate_fn if not configs.dataset.resize else None,
        )
        if validation_dataset is not None
        else None
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=configs.run.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
        collate_fn=custom_collate_fn if not configs.dataset.resize else None,
    ) if test_dataset is not None else None

    return train_loader, validation_loader, test_loader


def make_model(device: Device, model_cfg=None, random_state: int = None, **kwargs) -> nn.Module:

    if "pace" in model_cfg.name.lower():
        model = eval(model_cfg.name)(
            in_channels=configs.dataset.in_channels,
            out_channels=model_cfg.out_channels,
            domain_size=model_cfg.domain_size,
            grid_step=model_cfg.grid_step,
            buffer_width=model_cfg.buffer_width,
            pos_encoding=model_cfg.pos_encoding,
            with_cp=False,
            pace_config=model_cfg.pace_config,
            aux_pace=model_cfg.aux_pace,
            aux_pace_learn_residual=model_cfg.aux_pace_learn_residual,
            aux_pace_aug_input=model_cfg.aux_pace_aug_input,
            aux_pace_aug_feature=model_cfg.aux_pace_aug_feature,
            aux_pace_aug_feature_enhance=model_cfg.aux_pace_aug_feature_enhance,
            aux_pace_config=model_cfg.aux_pace_config,
            device=device,
            **kwargs,
        ).to(device)
        model.reset_parameters(random_state)
    elif "neurolight2d" in model_cfg.name.lower():
        model = eval(model_cfg.name)(
            in_channels=configs.dataset.in_channels,
            out_channels=model_cfg.out_channels,
            dim=model_cfg.dim,
            kernel_list=model_cfg.kernel_list,
            kernel_size_list=model_cfg.kernel_size_list,
            padding_list=model_cfg.padding_list,
            mode_list=model_cfg.mode_list,
            # act_func=model_cfg.act_func,
            # norm_func=model_cfg.norm_func,
            domain_size=model_cfg.domain_size,
            grid_step=model_cfg.grid_step,
            buffer_width=model_cfg.buffer_width,
            dropout_rate=model_cfg.dropout_rate,
            drop_path_rate=model_cfg.drop_path_rate,
            aux_head=model_cfg.aux_head,
            aux_head_idx=model_cfg.aux_head_idx,
            pos_encoding=model_cfg.pos_encoding,
            with_cp=model_cfg.with_cp,
            device=device,
            conv_stem=model_cfg.conv_stem,
            aug_path=model_cfg.aug_path,
            ffn=model_cfg.ffn,
            ffn_dwconv=model_cfg.ffn_dwconv,
            **kwargs,
        ).to(device)
        model.reset_parameters(random_state)
    else:
        model = None
        raise NotImplementedError(f"Not supported model name: {model_cfg.name}")

    return model



def make_refiner(device: Device, model_cfg=None,  random_state: int = None, **kwargs) -> nn.Module:
    model_cfg = model_cfg or configs.model
    if "pace" in model_cfg.name.lower():
        model = eval(model_cfg.name)(
            in_channels=configs.dataset.in_channels,
            out_channels=model_cfg.out_channels,
            dim=model_cfg.dim,
            domain_size=model_cfg.domain_size,
            grid_step=model_cfg.grid_step,
            buffer_width=model_cfg.buffer_width,
            pos_encoding=model_cfg.pos_encoding,
            with_cp=False,
            pace_config=model_cfg.pace_config,
            aux_pace=model_cfg.aux_pace,
            aux_pace_learn_residual=model_cfg.aux_pace_learn_residual,
            aux_pace_aug_input=model_cfg.aux_pace_aug_input,
            aux_pace_aug_feature=model_cfg.aux_pace_aug_feature,
            aux_pace_aug_feature_enhance=model_cfg.aux_pace_aug_feature_enhance,
            aux_pace_config=model_cfg.aux_pace_config,
            device=device,
            **kwargs,
        ).to(device)
        # TODO(hanqing): i didn't reset parameters
        model.reset_parameters(random_state)
    elif "ffno" in model_cfg.name.lower():
        model = eval(model_cfg.name)(
            in_channels=configs.dataset.in_channels,
            out_channels=model_cfg.out_channels,
            dim=model_cfg.dim,
            kernel_list=model_cfg.kernel_list,
            kernel_size_list=model_cfg.kernel_size_list,
            padding_list=model_cfg.padding_list,
            mode_list=model_cfg.mode_list,
            act_func=model_cfg.act_func,
            norm_func=model_cfg.norm_func,
            domain_size=model_cfg.domain_size,
            grid_step=model_cfg.grid_step,
            buffer_width=model_cfg.buffer_width,
            dropout_rate=model_cfg.dropout_rate,
            drop_path_rate=model_cfg.drop_path_rate,
            aux_head=model_cfg.aux_head,
            aux_head_idx=model_cfg.aux_head_idx,
            pos_encoding=model_cfg.pos_encoding,
            with_cp=model_cfg.with_cp,
            device=device,
            conv_stem=model_cfg.conv_stem,
            aug_path=model_cfg.aug_path,
            ffn=model_cfg.ffn,
            ffn_dwconv=model_cfg.ffn_dwconv,
            fno_bias=model_cfg.fno_bias,
            **kwargs,
        ).to(device)
        model.reset_parameters(random_state)
    else:
        model = None
        raise NotImplementedError(f"Not supported model name: {model_cfg.name}")

    return model

def make_optimizer(params, name: str = None, configs=None) -> Optimizer:
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=configs.lr,
            momentum=configs.momentum,
            weight_decay=configs.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            betas=getattr(configs, "betas", (0.9, 0.999)),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.5),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.001),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(optimizer: Optimizer, name: str = None) -> Scheduler:
    name = (name or configs.scheduler.name).lower()
    if name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(configs.run.n_epochs), eta_min=float(configs.scheduler.lr_min)
        )
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=configs.run.n_epochs,
            max_lr=configs.optimizer.lr,
            min_lr=configs.scheduler.lr_min,
            warmup_steps=int(configs.scheduler.warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=configs.scheduler.lr_gamma)
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None, cfg=None) -> nn.Module:
    name = (name or configs.criterion.name).lower()
    cfg = cfg or configs.criterion
    if name == "nll":
        criterion = nn.NLLLoss()
    elif name == "mse":
        criterion = nn.MSELoss()
    elif name == "cmse":
        criterion = ComplexMSELoss(norm=cfg.norm)
    elif name == "cmae":
        criterion = ComplexL1Loss(norm=cfg.norm)
    elif name == "mae":
        criterion = nn.L1Loss()
    elif name == "ce":
        criterion = nn.CrossEntropyLoss()
    elif name == "adaptive":
        criterion = AdaptiveLossSoft(alpha_min=-1.0, alpha_max=1.0)
    elif name == "mixed_kl":
        criterion = KLLossMixed(
            T=getattr(cfg, "T", 3),
            alpha=getattr(cfg, "alpha", 0.9),
        )
    elif name == "tv_loss":
        criterion = ComplexTVLoss(norm=cfg.norm)
    elif name == "div_loss":
        criterion = DivergenceLoss()
    elif name == "poynting_loss":
        criterion = PoyntingLoss(configs.model.grid_step, configs.model.wavelength)
    else:
        raise NotImplementedError(name)
    return criterion

