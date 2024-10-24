'''
 # @ Author: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Create Time: 2024-03-17 16:01:15
 # @ Modified by: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Modified time: 2024-10-23 22:28:43
 # @ Description: Define device shape at here with cutsomized parameters
'''
 
import cv2

from typing import Optional, Tuple
import numpy as np
from angler.structures import get_grid
import torch
import torch.nn.functional as F
from angler import Simulation

import matplotlib.pyplot as plt

import time

# global permitivity params for si and sio2
eps_sio2 = 1.44**2 # 2.0736
eps_si = 3.48**2 
verbose=False
__all__ = ["mmi_3x3_L_random_slots", "mmi_5x5_L_random_slots"]



def apply_regions(reg_list, xs, ys, eps_r_list, eps_bg):
    # feed this function a list of regions and some coordinates and it spits out a permittivity
    if isinstance(eps_r_list, (int, float)):
        eps_r_list = [eps_r_list] * len(reg_list)
    # if it's not a list, make it one
    if not isinstance(reg_list, (list, tuple)):
        reg_list = [reg_list]

    # initialize permittivity
    eps_r = np.zeros(xs.shape) + eps_bg

    # loop through lambdas and apply masks
    for e, reg in zip(eps_r_list, reg_list):
        reg_vec = np.vectorize(reg)
        material_mask = reg_vec(xs, ys)
        eps_r[material_mask] = e

    return eps_r


def gaussian_blurring(x):
    # return x
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()
    size = 3
    std = 0.4
    ax = torch.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    kernel = torch.exp(-0.5 / std ** 2 * (xx ** 2 + yy ** 2))
    kernel = kernel.div(kernel.sum()).unsqueeze(0).unsqueeze(0).float()
    return torch.nn.functional.conv2d(x, kernel, padding=size // 2).squeeze(0).squeeze(0).numpy()


class MMI_NxM(object):
    def __init__(
        self,
        num_in_ports: int,
        num_out_ports: int,
        box_size: Tuple[float, float],  # box [length, width], um
        wg_width: Tuple[float, float] = (0.4, 0.4),  # in/out wavelength width, um
        port_diff: Tuple[float, float] = (4, 4),  # distance between in/out waveguides. um
        port_len: float = 10,  # length of in/out waveguide from PML to box. um
        taper_width: float = 0.0,  # taper width near the multi-mode region. um. Default to 0
        taper_len: float = 0.0,  # taper length. um. Default to 0
        border_width: float = 3,  # space between box and PML. um
        grid_step: float = 0.1,  # isotropic grid step um
        NPML: Tuple[int, int] = (20, 20),  # PML pixel width. pixel
        eps_r: float = eps_si,  # relative refractive index
        eps_bg: float = eps_sio2,  # background refractive index
    ):
        super().__init__()
        self.num_in_ports = num_in_ports
        self.num_out_ports = num_out_ports
        self.box_size = box_size
        self.wg_width = wg_width
        self.port_diff = port_diff
        self.port_len = port_len
        
        # remove invalid taper
        if taper_width < 1e-5 or taper_len < 1e-5:
            taper_width = taper_len = 0
        self.taper_width = taper_width

        assert (
            max(taper_width, wg_width[0]) * num_in_ports <= box_size[1]
        ), "The input ports cannot fit the multimode region"
        assert (
            max(taper_width, wg_width[1]) * num_out_ports <= box_size[1]
        ), "The output ports cannot fit the multimode region"
        
        if taper_width > 1e-5:
            assert taper_width >= wg_width[0], "Taper width cannot be smaller than input waveguide width"
            assert taper_width >= wg_width[1], "Taper width cannot be smaller than output waveguide width"
        
        self.taper_len = taper_len

        self.border_width = border_width
        self.grid_step = grid_step
        self.NPML = list(NPML)
        self.eps_r = eps_r
        self.eps_bg = eps_bg

        ## geometric parameters
        Nx = 2 * NPML[0] + int(
            round((port_len * 2 + taper_len * 2 + box_size[0]) / grid_step)
        )  # num. grids in horizontal
        Ny = 2 * NPML[1] + int(round((box_size[1] + 2 * border_width) / grid_step))  # num. grids in vertical

        self.shape = (Nx, Ny)  # shape of domain (in num. grids)

        y_mid = 0
        # self.wg_width_px = [int(round(i / grid_step)) for i in wg_width]

        # x and y coordinate arrays
        self.xs, self.ys = get_grid(self.shape, grid_step)

        # define the multimode regions
        box = lambda x, y: (np.abs(x) < box_size[0] / 2) * (np.abs(y - y_mid) < box_size[1] / 2)

        in_ports = []
        out_ports = []
        # define the input ports
        for i in range(num_in_ports):
            y_i = (i - (num_in_ports - 1) / 2) * port_diff[0]
            # wg_i = lambda x, y, y_i=y_i: (x < 0) * (abs(y - y_i) < grid_step * self.wg_width_px[0] / 2)
            wg_i = lambda x, y, y_i=y_i: (x < 0) * (abs(y - y_i) < self.wg_width[0] / 2)
            if verbose:
                print(f"port {i} - {wg_i}")
            in_ports.append(wg_i)

        # define the input tapers
        # NOTE(hqzhu): input taper to minmize signal loss
        # define a linear taper here based on y = kx + b
        if taper_width > 1e-5 and taper_len > 1e-5:
            k = (taper_width - wg_width[0]) / (2 * taper_len)
            b = taper_width / 2 + box_size[0] * (taper_width - wg_width[0]) / (4 * taper_len)
            # NOTE(hqzhu): change to <=
            for i in range(num_in_ports):
                y_i = (i - (num_in_ports - 1) / 2) * port_diff[0]
                taper_i = lambda x, y, y_i=y_i, k=k, b=b: (-box_size[0] / 2 - taper_len <= x <= -box_size[0] / 2) * (
                    abs(y - y_i) < k * x + b
                )
                in_ports.append(taper_i)


        # degine the output ports
        for i in range(num_out_ports):
            y_i = (i - (num_out_ports - 1) / 2.0) * port_diff[1]
            wg_i = lambda x, y, y_i=y_i: (x > 0) * (abs(y - y_i) < self.wg_width[1] / 2)
            out_ports.append(wg_i)

        # define the output tapers:
        if taper_width > 1e-5 and taper_len > 1e-5:
            k = -(taper_width - wg_width[1]) / (2 * taper_len)
            b = taper_width / 2 + box_size[0] * (taper_width - wg_width[1]) / (4 * taper_len)
            for i in range(num_out_ports):
                y_i = (i - (num_in_ports - 1) / 2) * port_diff[1]
                taper_i = lambda x, y, y_i=y_i, k=k, b=b: (box_size[0] / 2 <= x <= box_size[0] / 2 + taper_len) * (
                    abs(y - y_i) < k * x + b
                )
                out_ports.append(taper_i)

        reg_list = [box] + in_ports + out_ports
        
        if verbose:
            print("reg_list")

        self.epsilon_map = apply_regions(reg_list, self.xs, self.ys, eps_r_list=eps_r, eps_bg=eps_bg)
        if verbose:
            print(f'check epsilon_map: {self.epsilon_map.size}')
        # NOTE(hqzhu): a defualt way to set the permitivity for design region
        self.design_region = apply_regions([box], self.xs, self.ys, eps_r_list=1, eps_bg=0)
        self.pad_regions = None

        #NOTE(hqzhu): x and y coordinates
        self.in_port_centers = [
            (-box_size[0] / 2 - 0.98 * port_len, (i - (num_in_ports - 1) / 2) * port_diff[0])
            for i in range(num_in_ports)
        ]  # centers

        cut = self.epsilon_map[0] > eps_bg
        # print(cut)
        edges = cut[1:] ^ cut[:-1]
        indices = np.nonzero(edges)[0]
        self.in_port_width_px = (indices[1::2] - indices[::2]).tolist()
        # print(indices)
        # print(self.in_port_width_px)
        centers = (indices[::2] + indices[1::2]) // 2 + 1
        self.in_port_centers_px = [
            (NPML[0] + int(round(box_size[0] / 2 + port_len - np.abs(x))), y)
            for (x, _), y in zip(self.in_port_centers, centers)
        ]

        cut = self.epsilon_map[-1] > eps_bg
        # print(cut)
        edges = cut[1:] ^ cut[:-1]
        indices = np.nonzero(edges)[0]
        self.out_port_width_px = (indices[1::2] - indices[::2]).tolist()
        # print(indices)
        # print(self.out_port_width_px)
        centers = (indices[::2] + indices[1::2]) // 2 + 1
        self.out_port_centers = [
            (box_size[0] / 2 + 0.98 * port_len, (float(i) - float(num_out_ports - 1) / 2.0) * port_diff[1])
            for i in range(num_out_ports)
        ]  # centers
        self.out_port_centers_px = self.out_port_pixel_centers = [
            (
                Nx - 1 - NPML[0] - int(round(box_size[0] / 2 + port_len - np.abs(x))),
                # NPML[1] + int(round((border_width + box_size[1] / 2 + y) / grid_step)),
                y
            )
            for (x, _), y in zip(self.out_port_centers, centers)
        ]


        # # fix the symmetric issue, especially on tapers
        # # NOTE(hqzhu): cannot understand, ask jiaqi
        if taper_width > 1e-5 and taper_len > 1e-5:
            for i in range(self.num_in_ports):
                left, right = NPML[0] + int(round(port_len / grid_step)), NPML[0] + int(
                    round((port_len + taper_len) / grid_step)
                )
                # left = 0
                # right = -1
                upper, lower = (
                    self.in_port_centers_px[i][1] + int(round(taper_width / 2 / grid_step)) + 2,
                    self.in_port_centers_px[i][1] - int(round(taper_width / 2 / grid_step)) - 2,
                )
                self.epsilon_map[left:right, lower:upper] = (
                    self.epsilon_map[left:right, lower:upper] + np.fliplr(self.epsilon_map[left:right, lower:upper])
                ) / 2

            for i in range(self.num_out_ports):
                left, right = -NPML[0] - int(round(port_len + taper_len / grid_step)), -NPML[0] - int(
                    round((port_len) / grid_step)
                )
                upper, lower = (
                    self.out_port_centers_px[i][1] + int(round(taper_width / 2 / grid_step)) + 2,
                    self.out_port_centers_px[i][1] - int(round(taper_width / 2 / grid_step)) - 2,
                )
                self.epsilon_map[left:right, lower:upper] = (
                    self.epsilon_map[left:right, lower:upper] + np.fliplr(self.epsilon_map[left:right, lower:upper])
                ) / 2
            
        self.epsilon_map = gaussian_blurring(self.epsilon_map)

    def set_pad_region(self, pad_regions):
        # pad_regions = [[xl, xh, yl, yh], [xl, xh, yl, yh], ...] rectanglar pads bounding box
        # (0,0) is the center of the entire region
        # default argument in lambda can avoid lazy evaluation in python!
        self.pad_regions = [
            lambda x, y, xl=xl, xh=xh, yl=yl, yh=yh: (xl < x < xh) and (yl < y < yh)
            for xl, xh, yl, yh in pad_regions
        ]
        self.pad_region_mask = apply_regions(
            self.pad_regions, self.xs, self.ys, eps_r_list=1, eps_bg=0
        ).astype(np.bool)
    
    def save_pad_info(self, center_x, center_y, size_x, size_y):
        self.pad_info_center_x = center_x
        self.pad_info_center_y = center_y
        self.pad_info_size_x = size_x
        self.pad_info_size_y = size_y

    def set_pad_eps(self, pad_eps) -> np.ndarray:
        assert self.pad_regions is not None and len(pad_eps) == len(self.pad_regions)
        epsilon_map = apply_regions(self.pad_regions, self.xs, self.ys, eps_r_list=pad_eps, eps_bg=0)
        return np.where(self.pad_region_mask, epsilon_map, self.epsilon_map)

    def trim_pml(self, epsilon_map: Optional[np.ndarray] = None):
        epsilon_map = epsilon_map if epsilon_map is not None else self.epsilon_map
        return epsilon_map[
            ...,
            self.NPML[0] : epsilon_map.shape[-2] - self.NPML[-2],
            self.NPML[1] : epsilon_map.shape[-1] - self.NPML[-1],
        ]

    def resize(self, x, size, mode="bilinear"):
        if not isinstance(x, torch.Tensor):
            y = torch.from_numpy(x)
        else:
            y = x
        y = y.view(-1, 1, x.shape[-2], x.shape[-1])
        old_grid_step = (self.grid_step, self.grid_step)
        old_size = y.shape[-2:]
        new_grid_step = [old_size[0] / size[0] * old_grid_step[0], old_size[1] / size[1] * old_grid_step[1]]
        if y.is_complex():
            y = torch.complex(
                torch.nn.functional.interpolate(y.real, size=size, mode=mode),
                torch.nn.functional.interpolate(y.imag, size=size, mode=mode),
            )
        else:
            y = torch.nn.functional.interpolate(y, size=size, mode=mode)
        y = y.view(list(x.shape[:-2]) + list(size))
        if isinstance(x, np.ndarray):
            y = y.numpy()
        return y, new_grid_step

    def __repr__(self) -> str:
        str = f"MMI{self.num_in_ports}x{self.num_out_ports}("
        str += f"size = {self.box_size[0]} um x {self.box_size[1]} um)"
        return str

def mmi_3x3_L_random_slots(size=(30,7), port_len=3, grid_step=0.05, NPWL=(30, 30), n_sampled_slots_range=(0.05, 0.1), taper_size=(3, 1.1)):
    ## random rectangular SiO2 slots
    N = 3
    
    size = (np.random.uniform(20, 30), np.random.uniform(5.5, 7))
    # print(size)
    # size = (np.random.uniform(40, 50), np.random.uniform(5.5, 7))
    taper_len = taper_size[0]
    taper_width = taper_size[1]
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        taper_len=taper_len, # length of taper, which is required to avoid light being out
        taper_width=taper_width, # width of tapper, which is required to minimize the loss
        border_width=0.25,  # space between box and PML. um
        grid_step=grid_step,  # isotropic grid step um
        NPML=NPWL,  # PML pixel width. pixel
    )

    n_slots = (30, 7)
    total_slots = n_slots[0] * n_slots[1]
    assert n_sampled_slots_range[0] <= n_sampled_slots_range[-1]
    n_sampled_slots = int(np.random.uniform(n_sampled_slots_range[0], n_sampled_slots_range[-1]) * total_slots)
    # n_sampled_slots = int(np.random.uniform(0.1) * total_slots)
    
    #NOTE(hqzhu): pad generation, here we generate the pad based on the size of device, so the pad is invariant with device size
    w, h = size[0] / n_slots[0] * 0.8, size[1] / n_slots[1] * 0.8  # do not remove materials on the boundary
    slot_centers_x = np.linspace(-(n_slots[0] / 2 - 0.5) * w, (n_slots[0] / 2 - 0.5) * w, n_slots[0])
    # slot_centers_y = np.linspace(-(n_slots[1] / 2 - 0.5) * h, (n_slots[1] / 2 - 0.5) * h, n_slots[1])
    slot_centers_y = np.linspace(-(n_slots[1] / 2 - 0.5) * h, (n_slots[1] / 2 - 0.5) * h, n_slots[1])
    

    centers_x = np.random.choice(slot_centers_x, size=n_sampled_slots, replace=True)
    centers_y = slot_centers_y[
        (np.round(np.random.choice(len(slot_centers_y), size=n_sampled_slots, replace=True) / 2) * 2).astype(np.int32)
    ]  # a trick to generate slots along the prop direction
    
    pad_centers = np.stack([centers_x, centers_y], axis=-1)
    # pad_centers = np.array(list(product(slot_centers_x, slot_centers_y)))[np.random.choice(total_slots, size=n_sampled_slots, replace=False)]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    mmi.save_pad_info(centers_x, centers_y, w, h)
    
    return mmi


def mmi_5x5_L_random_slots(port_len=3, grid_step=0.05, NPWL=(30, 30), n_sampled_slots_range=(0.05, 0.1), taper_size=(3, 1.1)):
    ## random rectangular SiO2 slots
    N = 5
    size = (np.random.uniform(25, 35), np.random.uniform(7.5, 9))
    
    taper_len = taper_size[0]
    taper_width = taper_size[1]
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        taper_len=taper_len,
        taper_width=taper_width,
        border_width=0.25,  # space between box and PML. um
        grid_step=grid_step,  # isotropic grid step um
        NPML=NPWL,  # PML pixel width. pixel
    )
    
    n_slots = (35, 9)
    total_slots = n_slots[0] * n_slots[1]
    assert n_sampled_slots_range[0] <= n_sampled_slots_range[-1]
    n_sampled_slots = int(np.random.uniform(n_sampled_slots_range[0], n_sampled_slots_range[-1]) * total_slots)

    #NOTE(hqzhu): pad generation, here we generate the pad based on the size of device, so the pad is invariant with device size
    w, h = size[0] / n_slots[0] * 0.8, size[1] / n_slots[1] * 0.8  # do not remove materials on the boundary
    slot_centers_x = np.linspace(-(n_slots[0] / 2 - 0.5) * w, (n_slots[0] / 2 - 0.5) * w, n_slots[0])
    slot_centers_y = np.linspace(-(n_slots[1] / 2 - 0.5) * h, (n_slots[1] / 2 - 0.5) * h, n_slots[1])

    centers_x = np.random.choice(slot_centers_x, size=n_sampled_slots, replace=True)
    centers_y = slot_centers_y[
        (np.round(np.random.choice(len(slot_centers_y), size=n_sampled_slots, replace=True) / 2) * 2).astype(np.int32)
    ]  # a trick to generate slots along the prop direction
    pad_centers = np.stack([centers_x, centers_y], axis=-1)
    # pad_centers = np.array(list(product(slot_centers_x, slot_centers_y)))[np.random.choice(total_slots, size=n_sampled_slots, replace=False)]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    mmi.save_pad_info(centers_x, centers_y, w, h)
    
    return mmi