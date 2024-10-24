'''
 # @ Author: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Create Time: 2024-10-20 15:19:47
 # @ Modified by: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Modified time: 2024-10-21 00:50:54
 # @ Description:
 '''

from typing import Optional, Tuple
import torch
from torch import Tensor
import numpy as np
import random
import math
import scipy.io
import h5py
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device
from math import exp
import operator
from functools import lru_cache, reduce
from functools import partial
from pyutils.compute import get_complex_energy
from angler import Simulation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft
#################################################
#
# Utilities
#
#################################################
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MaskerImage(nn.Module):
    """Performs masking on datasets with regular meshes.
    """
    def __init__(self, 
                drop_type="zeros", 
                max_block=0.7, 
                drop_pix=0.3, 
                channel_per=0.5, 
                channel_drop_per=0.2, 
                device="cpu",
                min_block=10,
                additive_noise=False,
                additive_noise_type='gaussian',
                additive_noise_std=0.1,
                block_wise=False,
                block_size=8,
                ):
        super(MaskerImage, self).__init__()
        
        self.drop_type = drop_type
        self.max_block = max_block
        self.drop_pix = drop_pix
        self.channel_per = channel_per
        self.channel_drop_per = channel_drop_per
        self.device = device
        self.min_block = min_block
        
        self.additive_noise = additive_noise
        self.additive_noise_type = additive_noise_type
        assert additive_noise_type in ['gaussian', 'uniform']
        self.additive_noise_std = additive_noise_std

        self.block_wise = block_wise
        self.block_size = block_size
    
    def set_drop_pix(self, drop_pix):
        self.drop_pix = drop_pix
    
    def forward(self, x, random_state=None):
        # https://github.com/facebookresearch/mae/blob/main/models_mae.py
        # NOTE(hqzhu): only perform masking on the field part
        bs, c, h, w = x.shape # batch size, channel, height, width
        
        if random_state is not None:
            old_seed = torch.random.get_rng_state()
            torch.random.set_rng_state(random_state)
        
        if self.drop_pix > 0:
            if not self.block_wise:
                # preform random mask on the last two dimensions
                len_keep = int(h * w * (1 - self.drop_pix)) # noise in [0, 1]
                
                ## todo(hqzhu): only drop the non-input part
                # Create a random mask for each image in the batch
                mask = torch.zeros(bs, h * w, device=x.device)
                for i in range(bs):
                    # Randomly choose indices to keep
                    keep_indices = torch.randperm(h * w)[:len_keep]
                    mask[i, keep_indices] = 1
                    
                # Reshape mask to match the input dimensions
                mask = mask.view(bs, 1, h, w)
            else:
                # block-wise masking
                mask = torch.ones(bs, 1, h, w, device=x.device)
                num_blocks = int((h * w * self.drop_pix) / (self.block_size ** 2))
                # Generate blocks for each image in the batch
                for i in range(bs):
                    for _ in range(num_blocks):
                        block_size = torch.randint(self.block_size, 2 * self.block_size, (1,)).item()
                        # Random start coordinates for the block
                        x_block = torch.randint(0, h - block_size + 1, (1,))
                        y_block = torch.randint(0, w - block_size + 1, (1,))
                        
                        x_block_r = max(x_block + block_size, h)
                        y_block_r = max(y_block + block_size, w)
                        
                        mask[i, :, x_block:x_block_r, y_block:y_block_r] = 0
            # Apply the mask to the input
            if self.drop_type == 'zeros':
                x = x * mask
            elif self.drop_type == 'fill':
                fill_value = x.mean() if self.drop_type == 'fill' else 0
                x = x * mask + (1 - mask) * fill_value
            else:
                raise ValueError(f"Unsupported drop_type {self.drop_type}")
        else:
            mask = None

        # Add Gaussian noise to whole image
        # the noise should multiply with the image
        if self.additive_noise:
            if self.additive_noise_type == 'gaussian':
                if x.is_complex():
                    # noise_real = torch.randn_like(x.real) * self.additive_noise_std + 1# 0.5
                    # noise_imag = torch.randn_like(x.imag) * self.additive_noise_std + 1# 0.5
                    # x = torch.complex(noise_real, noise_imag) * x
                    # noise = torch.complex(noise_real, noise_imag)
                    # lower_bound = torch.randint(1, 3, (1,)).item() # 10%~40% noise
                    # noise_real = torch.randn_like(x.real) * self.additive_noise_std # sample a noise in guassian noise
                    # noise_imag = torch.randn_like(x.imag) * self.additive_noise_std # sample a noise in guassian noise
                    # # Ensure the absolute value of noise is greater than 0.5
                    # noise_real = torch.sign(noise_real) * (torch.abs(noise_real) + lower_bound/10)
                    # noise_imag = torch.sign(noise_imag) * (torch.abs(noise_imag) + lower_bound/10)
                    # x += torch.complex(noise_real, noise_imag) * x
                    # random additive noise
                    noise_real = torch.randn_like(x.real) * self.additive_noise_std
                    noise_imag = torch.randn_like(x.imag) * self.additive_noise_std
                    lower_bound = torch.randint(1, 4, (1,)).item() # 10%~40% noise
                    noise_real = torch.sign(noise_real) * (torch.abs(noise_real) + lower_bound/10)
                    noise_imag = torch.sign(noise_imag) * (torch.abs(noise_imag) + lower_bound/10)
                    
                    noise_imag *= ~hint_mask
                    noise_real *= ~hint_mask
                    x += torch.complex(noise_real, noise_imag)
                else:
                    noise = torch.randn_like(x) * self.additive_noise_std + 1
                    x = x * noise
                # x = x + noise
            else:
                raise ValueError(f"Unsupported noise type {self.additive_noise_type}")
        if random_state is not None:
            torch.random.set_rng_state(old_seed)
    
        return x, mask

class MaskerUniform:
    """Performs masking on datasets with regular meshes.

    For masking with data points from data sets with irregular meshes,
    use ``MaskerNonuniformMesh`` (below).
    """

    def __init__(
        self,
        drop_type="zeros",
        max_block=0.7,
        drop_pix=0.3,
        channel_per=0.5,
        channel_drop_per=0.2,
        device="cpu",
        min_block=10,
    ):
        self.drop_type = drop_type
        self.max_block = max_block
        self.drop_pix = drop_pix
        self.channel_per = channel_per
        self.channel_drop_per = channel_drop_per
        self.device = device
        self.min_block = min_block

    def __call__(self, size):
        """Returns a mask to be multiplied into a data tensor.

        Generates a binary mask of 0s and 1s to be point-wise multiplied into a
        data tensor to create a masked sample. By training on masked data, we
        expect the model to be resilient to missing data.

        TODO arg ``max_block_sz`` is outdated
        Parameters
        ---
        max_block_sz: percentage of the maximum block to be dropped
        """

        # NOTE(hqzhu): remove this to ensure reproductivity
        # np.random.seed() 
        C, H, W = size
        mask = torch.ones(size, device=self.device)
        drop_t = self.drop_type
        augmented_channels = np.random.choice(C, math.ceil(C * self.channel_per))

        drop_len = int(self.channel_drop_per * math.ceil(C * self.channel_per))
        mask[augmented_channels[:drop_len], :, :] = 0.0
        for i in augmented_channels[drop_len:]:

            n_drop_pix = self.drop_pix * H * W
            mx_blk_height = int(H * self.max_block)
            mx_blk_width = int(W * self.max_block)

            while n_drop_pix > 0:
                rnd_r = random.randint(0, H - 2)
                rnd_c = random.randint(0, W - 2)

                rnd_h = min(random.randint(self.min_block, mx_blk_height), H - rnd_r)
                rnd_w = min(random.randint(self.min_block, mx_blk_width), W - rnd_c)
                mask[i, rnd_r : rnd_r + rnd_h, rnd_c : rnd_c + rnd_w] = 0
                n_drop_pix -= rnd_h * rnd_c
        return None, mask


def save_model(*args, **kwargs):
    torch.save(*args, **kwargs)

# Define the function to compute the spectrum
def spectrum_2d(signal, n_observations, normalize=True):
    """This function computes the spectrum of a 2D signal using the Fast Fourier Transform (FFT).

    Paramaters
    ----------
    signal : a tensor of shape (T * n_observations * n_observations)
        A 2D discretized signal represented as a 1D tensor with shape
        (T * n_observations * n_observations), where T is the number of time
        steps and n_observations is the spatial size of the signal.

        T can be any number of channels that we reshape into and
        n_observations * n_observations is the spatial resolution.
    n_observations: an integer
        Number of discretized points. Basically the resolution of the signal.

    Returns
    --------
    spectrum: a tensor
        A 1D tensor of shape (s,) representing the computed spectrum.
    """
    T = signal.shape[0]
    signal = signal.view(T, n_observations, n_observations)

    if normalize:
        signal = torch.fft.fft2(signal)
    else:
        signal = torch.fft.rfft2(
            signal, s=(n_observations, n_observations), normalized=False
        )

    # 2d wavenumbers following PyTorch fft convention
    k_max = n_observations // 2
    wavenumers = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1),
            torch.arange(start=-k_max, end=0, step=1),
        ),
        0,
    ).repeat(n_observations, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers

    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)
    sum_k = sum_k

    # Remove symmetric components from wavenumbers
    index = -1.0 * torch.ones((n_observations, n_observations))
    k_max1 = k_max + 1
    index[0:k_max1, 0:k_max1] = sum_k[0:k_max1, 0:k_max1]

    spectrum = torch.zeros((T, n_observations))
    for j in range(1, n_observations + 1):
        ind = torch.where(index == j)
        spectrum[:, j - 1] = (signal[:, ind[0], ind[1]].sum(dim=1)).abs() ** 2

    spectrum = spectrum.mean(dim=0)
    return spectrum

def filter_spectrum_2d(signal, n_observations, normalize=True, low=0, mid=5, high=-1):
    """This function try to visualize the frequency info by masking the spectrum out. """
    pass


# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
# equal to H1loss used in HT-net by setting d=2. p=2, k=1, a=2*math*pi
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()
        # Dimension and Lp-norm type are postive
        # p: norm
        # k: order of derivatives to consider in loss function
        # d: spatial dimension
        
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [
                1,
            ] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = (
            torch.cat(
                (torch.arange(start=0, end=nx // 2, step=1), torch.arange(start=-nx // 2, end=0, step=1)), 0
            )
            .reshape(nx, 1)
            .repeat(1, ny)
        )
        k_y = (
            torch.cat(
                (torch.arange(start=0, end=ny // 2, step=1), torch.arange(start=-ny // 2, end=0, step=1)), 0
            )
            .reshape(1, ny)
            .repeat(nx, 1)
        )
        k_x = torch.abs(k_x).reshape(1, nx, ny, 1).to(x.device)
        k_y = torch.abs(k_y).reshape(1, nx, ny, 1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced == False:
            weight = 1
            if k >= 1:
                weight += a[0] ** 2 * (k_x ** 2 + k_y ** 2)
            if k >= 2:
                weight += a[1] ** 2 * (k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 + k_y ** 4)
            weight = torch.sqrt(weight)
            loss = self.rel(x * weight, y * weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x ** 2 + k_y ** 2)
                loss += self.rel(x * weight, y * weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 + k_y ** 4)
                loss += self.rel(x * weight, y * weight)
            loss = loss / (k + 1)

        return loss


# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))
    return c


class Laplacian_2d(nn.Module):
    def __init__(self, grid_step: float):
        """Spatial discrete Laplacian operator in the 2D space. Assume square grid size.

        Args:
            grid_step (float): The step size of the grid in the unit of um
        """
        super().__init__()
        # 4-point stencil, only good for smooth function
        kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float) / grid_step ** 2

        # 2nd-order accurate 9-point stencil, also good for non-smooth function.
        # https://journals.ametsoc.org/view/journals/mwre/145/11/mwr-d-17-0015.1.xml
        # kernel = (
        #     torch.tensor([[[[0.25, 0.5, 0.25], [0.5, -3, 0.5], [0.25, 0.5, 0.25]]]], dtype=torch.float)
        #     / grid_step ** 2
        # )
        # kernel = (
        #     torch.tensor([[[[1, 4, 1], [4, -20, 4], [1, 4, 1]]]], dtype=torch.float)
        #     / 6 / grid_step ** 2
        # )

        ## 18th-order accurate and compact Laplacian in 2D uniform cartesian mesh
        ## https://hal.archives-ouvertes.fr/hal-01998201/document
        a0 = -2838953 / 817278
        a1 = 889264 / 1225917
        a2 = 549184 / 3677751
        a01 = -18881 / 2451834
        a11 = 392 / 525393
        a02 = -233 / 2674728

        kernel = (
            torch.tensor(
                [
                    [
                        [
                            [a01, a11, a01, a11, a02],
                            [a11, a2, a1, a2, a11],
                            [a01, a1, a0, a1, a01],
                            [a11, a2, a1, a2, a11],
                            [a02, a11, a01, a11, a02],
                        ]
                    ]
                ],
                dtype=torch.float,
            )
            / grid_step ** 2
        )
        self.register_buffer("kernel", kernel)

    def forward(self, x: Tensor) -> Tensor:
        ## x [bs, inc, h, w] is padded field if there is any buffer
        if self.kernel.device != x.device:
            self.kernel = self.kernel.to(x.device)
        inc, h, w = x.shape[1:]
        x = x.flatten(0, 1).unsqueeze(1)  # [bs*inc, 1, h, w]
        # TODO: fix the boundary
        if x.is_complex():
            # Laplacian in the complex plane is just Laplacian of a vector function, i.e., independently on Real and Imag
            x = torch.view_as_real(x).permute(0, 4, 1, 2, 3).flatten(0, 1)  # [bs*inc*2, 1, h, w] real
            x = torch.conv2d(x, self.kernel, padding=self.kernel.size(-1) // 2)  # [bs*inc*2, 1, h, w] real
            x = torch.view_as_complex(
                x.reshape(-1, inc, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()
            )  # [bs, inc, h, w] complex
        else:
            x = torch.conv2d(x, self.kernel, padding=self.kernel.size(-1) // 2)  # [bs*inc, 1, h, w]
            x = x.view(-1, inc, h, w)

        return x

# Maxwell spatial residual based on Faraday's equation
class FaradayResidue2d(torch.nn.modules.loss._Loss):
    def __init__(self, grid_step: float, wavelength: float) -> None:
        """Frequency-domain electric field residue based on Faraday-Maxwell's Equation.\\
            Assume no source or current.
        curl(curl(E)) - omega^2 * mu_0 * epsilon_0 * epsilon_r * E = J \\
        Simplified to: Laplacian(E) + omega^2 * mu_0 * epsilon_0 * epsilon_r * E + J = 0

        Args:
            grid_step (float): The step size of the grid in the unit of um
            wavelength (float): Wavelength lambda in the unit of um
        """
        super().__init__()
        self.grid_step = grid_step
        self.vacuum_permeability = 1.25663706e-6
        self.vacuum_permittivity = 8.85418782e-12
        self.c = 1 / (self.vacuum_permeability * self.vacuum_permittivity) ** 0.5
        self.angular_freq = 2 * np.pi * self.c / (wavelength / 1e6)

        ## square of phase propagation factor in vacuum: coeff = beta^2 = omega^2 * mu_0 * epsilon_0
        self.beta_sq = 4 * np.pi ** 2 / (wavelength ** 2)
        self.laplacian_2d_op = Laplacian_2d(grid_step)

    def _forward_impl_real(
        self, x: Tensor, epsilon_r: Tensor, source_field: Optional[Tensor] = None, reduction: str = "mse"
    ) -> Tensor:
        """
        Real field and real permittivity in a lossless system\\
        Laplacian(E) + omega^2 * mu_0 * epsilon_0 * epsilon_r * E + J = 0\\
        Laplacian(E) + beta^2 * E + J = 0

        Args:
            x: (Tensor): Padded frequency-domain electric field \\
            epsilon_r (Tensor): Padded relative electric permittivity field \\
            source_field (Complex Tensor): Padded input light source field
        Returns:
            Tensor: Residual loss from Faraday equation
        """
        x_lap = self.laplacian_2d_op(x)
        x = torch.addcmul(x_lap, x, epsilon_r, value=self.beta_sq)
        if x is not None:
            x = x.add(source_field)
        if reduction == "mse":
            return x.square().mean()
        elif reduction == "mae":
            return x.abs().mean()
        else:
            return x

    def _forward_impl_complex(
        self, x: Tensor, epsilon_r: Tensor, source_field: Optional[Tensor] = None, reduction: str = "mse"
    ) -> Tensor:
        """
        Complex field and complex permittivity in a lossy system
        Laplacian(E) + omega^2 * mu_0 * epsilon_0 * epsilon_r * E + J = 0, where epsilon_r = epsilon' - j* epsilon'' is complex-valued\\
        Laplacian(E) - gamma^2 * E + J = 0

        Args:
            x: (Tensor): Padded frequency-domain electric field \\
            epsilon_r (Tensor): Padded relative electric permittivity field \\
            source_field (Complex Tensor): Padded input light source field

        Returns:
            Tensor: Residual loss from Faraday equation
        """
        x_lap = self.laplacian_2d_op(x)
        # print(x_lap.device, x.device, epsilon_r.device, source_field.device)
        x = x_lap.add(x.mul(epsilon_r.mul(self.beta_sq)))
        if source_field is not None:
            x = x.add(source_field)
        if reduction == "mse":
            return torch.view_as_real(x).square().mean()
        elif reduction == "mae":
            return torch.view_as_real(x).abs().mean()
        else:
            return x

    def forward(
        self, x: Tensor, epsilon_r: Tensor, source_field: Optional[Tensor] = None, reduction: str = "mse"
    ) -> Tensor:
        """
        Args:
            x: (Real/Complex Tensor): Padded frequency-domain electric field \\
            epsilon_r (Real/Complex Tensor): Padded relative electric permittivity field \\
            source_field (Complex Tensor): Padded input light source field

        Return:
            y: (Tensor): Residual loss from Faraday equation
        """
        if x.is_complex():
            return self._forward_impl_complex(x, epsilon_r, source_field, reduction=reduction)
        else:
            return self._forward_impl_real(x, epsilon_r, source_field, reduction=reduction)


class IntensityMSELoss(torch.nn.MSELoss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """MSE between the intensity of predicted electric field and target field

        Args:
            x (Tensor): Predicted complex-valued frequency-domain full electric field
            target (Tensor): Ground truth real-valued electric field intensity

        Returns:
            Tensor: loss
        """
        return F.mse_loss(get_complex_energy(torch.view_as_real(x)), target)


class IntensityL1Loss(torch.nn.MSELoss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """L1 between the intensity of predicted electric field and target field

        Args:
            x (Tensor): Predicted complex-valued frequency-domain full electric field
            target (Tensor): Ground truth real-valued electric field intensity

        Returns:
            Tensor: loss
        """
        return F.l1_loss(get_complex_energy(torch.view_as_real(x)), target)


class ComplexMSELoss(torch.nn.MSELoss):
    def __init__(self, norm=False, left_factor=0.5, right_factor=2) -> None:
        super().__init__()
        self.norm = norm
        self.left_factor = left_factor
        self.right_factor = right_factor

    def forward(self, x: Tensor, target: Tensor, mask_scaler: Tensor=None, apply_scaling: bool=False) -> Tensor:
        """Complex MSE between the predicted electric field and target field

        Args:
            x (Tensor): Predicted complex-valued frequency-domain full electric field
            target (Tensor): Ground truth complex-valued electric field intensity

        Returns:
            Tensor: loss
        """
        # factors = torch.linspace(0.1, 1, x.size(-1), device=x.device).view(1,1,1,-1)
        # return F.mse_loss(torch.view_as_real(x.mul(factors)), torch.view_as_real(target.mul(factors)))
        
        if apply_scaling:
            width_midpoint = x.size(-1) // 2
            factors = torch.ones_like(torch.view_as_real(x))
            factors[..., :width_midpoint, :] *= self.left_factor
            factors[..., width_midpoint:, :] *= self.right_factor
        else:
            factors = None

        if self.norm:
            diff = torch.view_as_real(x - target)
            if factors is not None:
                diff *= factors
            if mask_scaler is not None:
                return (
                    diff.square()
                    .sum(dim=[1, 2, 3, 4])
                    .div(torch.view_as_real(target).square().sum(dim=[1, 2, 3, 4]))
                    .mul(mask_scaler)
                    .mean()
                )
            else:
                return (
                    diff.square()
                    .sum(dim=[1, 2, 3, 4])
                    .div(torch.view_as_real(target).square().sum(dim=[1, 2, 3, 4]))
                    .mean()
                )
        else:
            if mask_scaler is not None:
                return F.mse_loss(torch.view_as_real(x), torch.view_as_real(target), reduction="none").mul(mask_scaler).mean()
            return F.mse_loss(torch.view_as_real(x), torch.view_as_real(target))


class ComplexL1Loss(torch.nn.MSELoss):
    def __init__(self, norm=False, left_factor=1, right_factor=2) -> None:
        super().__init__()
        self.norm = norm
        self.left_factor = left_factor
        self.right_factor = right_factor

    def forward(self, x: Tensor, target: Tensor, mask_scaler: Tensor=None, apply_scaling: bool=False) -> Tensor:
        """Complex L1 loss between the predicted electric field and target field

        Args:
            x (Tensor): Predicted complex-valued frequency-domain full electric field
            target (Tensor): Ground truth complex-valued electric field intensity

        Returns:
            Tensor: loss
        """
        
        if apply_scaling:
            width_midpoint = x.size(-1) // 2
            factors = torch.ones_like(torch.view_as_real(x))
            factors[..., :width_midpoint, :] *= self.left_factor
            factors[..., width_midpoint:, :] *= self.right_factor
        else:
            factors = None
        
        if self.norm:
            diff = torch.view_as_real(x - target)
            if factors is not None:
                diff *= factors
            if mask_scaler is not None:
                return (
                    diff.norm(p=1, dim=[1, 2, 3, 4])
                    .div(torch.view_as_real(target).norm(p=1, dim=[1, 2, 3, 4]))
                    .mul(mask_scaler)
                    .mean()
                )
            else:
                return (
                    diff.norm(p=1, dim=[1, 2, 3, 4])
                    .div(torch.view_as_real(target).norm(p=1, dim=[1, 2, 3, 4]))
                    .mean()
                )
        else:
            if mask_scaler is not None:
                return F.l1_loss(torch.view_as_real(x), torch.view_as_real(target), reduction="none").mul(mask_scaler).mean()
            else:
                return F.l1_loss(torch.view_as_real(x), torch.view_as_real(target))


class ComplexTVLoss(torch.nn.MSELoss):
    def __init__(self, norm=False) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """TV loss between the predicted electric field and target field

        Args:
            x (Tensor): Predicted complex-valued frequency-domain full electric field
            target (Tensor): Ground truth complex-valued electric field intensity

        Returns:
            Tensor: loss
        """
        target_deriv_x = torch.view_as_real(target[..., 1:, :] - target[..., :-1, :])
        target_deriv_y = torch.view_as_real(target[..., 1:] - target[..., :-1])
        pred_deriv_x = torch.view_as_real(x[..., 1:, :] - x[..., :-1, :])
        pred_deriv_y = torch.view_as_real(x[..., 1:] - x[..., :-1])
        if self.norm:
            return (
                (pred_deriv_x - target_deriv_x)
                .square()
                .sum(dim=[1, 2, 3, 4])
                .add((pred_deriv_y - target_deriv_y).square().sum(dim=[1, 2, 3, 4]))
                .div(
                    target_deriv_x.square()
                    .sum(dim=[1, 2, 3, 4])
                    .add(target_deriv_y.square().sum(dim=[1, 2, 3, 4]))
                )
                .mean()
            )
        return F.mse_loss(pred_deriv_x, target_deriv_x) + F.mse_loss(pred_deriv_y, target_deriv_y)


class PhaseLoss(torch.nn.MSELoss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """Phase loss between the predicted electric field and target field

        Args:
            x (Tensor): Predicted complex-valued frequency-domain full electric field
            target (Tensor): Ground truth complex-valued electric field intensity

        Returns:
            Tensor: loss
        """
        return F.mse_loss(x.angle(), target.angle())


class ComplexAdptiveFourierLoss(torch.nn.MSELoss):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, alpha=1.0, ave_spectrum=False, log_matrix=False, batch_matrix=False, norm=True):
        super(ComplexAdptiveFourierLoss, self).__init__()
        self.alpha = alpha
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.norm = norm

    def tensor2freq(self, x):
        # crop image patches
        # patch_factor = self.patch_factor
        # _, _, h, w = x.shape
        # assert h % patch_factor == 0 and w % patch_factor == 0, (
        #     'Patch factor should be divisible by image height and width')
        # patch_list = []
        # patch_h = h // patch_factor
        # patch_w = w // patch_factor
        # for i in range(patch_factor):
        #     for j in range(patch_factor):
        #         patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # # stack to patch tensor
        # y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        # perform 2D DFT (complex-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(x, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(x, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        """Compute the spectral loss on the fly"""
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            # real and imaginary parts with a norm alpha
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1] # bs, c, h, w

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance # bs, c, h, w
        
        if self.norm:
            loss = (
                loss.sum(dim=[1, 2, 3])
                .div(recon_freq.square().sum(dim=[1, 2, 3, 4]))
            ).mean()
        else:
            loss = torch.mean(loss)

        # print('loss shape', loss.shape)
        # print('loss', loss)
        # raise NotImplementedError
        return loss

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        # print('pred shape', pred.shape)
        # print('target shape', target.shape)
        
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        
        # print('pred shape', pred_freq.shape)
        # print('target shape', target_freq.shape)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix)


class DivergenceLoss(torch.nn.MSELoss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, permittivity: Tensor) -> Tensor:
        """Phase loss between the predicted electric field and target field

        Args:
            x (Tensor): Predicted complex-valued frequency-domain full electric field
            target (Tensor): Ground truth complex-valued electric field intensity

        Returns:
            Tensor: loss
        """
        x = x * permittivity
        return torch.view_as_real(x[..., 1:, :] - x[..., :-1, :]).square().mean()


class TMPropagateFieldLoss(torch.nn.modules.loss._Loss):
    def __init__(self, grid_step: float, wavelength: float) -> None:
        """2-D Frequency-domain magnatic field residue based on Maxwell's Equation. z: prop, x: cross, y: vertical\\
            Assume no source or current.
        dxx_Hy + omega^2/c0^2 * epsilon * Hy = 1/epsilon*dx_epsilon * dx_Hy \\

        Args:
            grid_step (float): The step size of the grid in the unit of um
            wavelength (float): Wavelength lambda in the unit of um
        """
        super().__init__()
        self.grid_step = grid_step
        self.vacuum_permeability = 1.25663706e-6
        self.vacuum_permittivity = 8.85418782e-12
        self.c = 1 / (self.vacuum_permeability * self.vacuum_permittivity) ** 0.5
        self.angular_freq = 2 * np.pi * self.c / (wavelength / 1e6)
        self.wavelength = wavelength
        self.scale = 1e15
        ## square of phase propagation factor in vacuum: coeff = beta^2 = omega^2 * mu_0 * epsilon_0
        self.beta_sq = 4 * np.pi ** 2 / (wavelength ** 2)
        self.dx_kernel = torch.tensor([-1, 0, 1]).view(1, 1, 3, 1) / (
            2 * self.grid_step
        )  # partial derivative along cross direction
        self.dxx_kernel = torch.tensor([1, -2, 1]).view(1, 1, 3, 1) / (
            self.grid_step ** 2
        )  # second-order partial derivative along cross direction

    def _forward_impl_real(
        self, x: Tensor, epsilon_r: Tensor, source_field: Optional[Tensor] = None, reduction: str = "mse"
    ) -> Tensor:
        """
        Real field and real permittivity in a lossless system\\
        Laplacian(E) + omega^2 * mu_0 * epsilon_0 * epsilon_r * E + J = 0\\
        Laplacian(E) + beta^2 * E + J = 0

        Args:
            x: (Tensor): Padded frequency-domain electric field \\
            epsilon_r (Tensor): Padded relative electric permittivity field \\
            source_field (Complex Tensor): Padded input light source field
        Returns:
            Tensor: Residual loss from Faraday equation
        """
        Hy = x  # for TM mode, we solve Hy (vertical H field)
        if self.dxx_kernel.device != x.device:
            self.dxx_kernel = self.dxx_kernel.to(x.device)
        dxx_Hy = F.conv2d(Hy, weight=self.dxx_kernel, padding=(1, 0))
        if self.dx_kernel.device != x.device:
            self.dx_kernel = self.dx_kernel.to(x.device)
        dx_Hy = F.conv2d(Hy, weight=self.dx_kernel, padding=(1, 0))
        dx_epsilon_r = F.conv2d(epsilon_r, weight=self.dx_kernel, padding=(1, 0))
        coeff = self.beta_sq.mul(epsilon_r) - self.beta_sq  # (k2-beta^2)
        x = dxx_Hy + coeff * Hy - 1 / epsilon_r * dx_epsilon_r * dx_Hy
        if source_field is not None:
            x = x.add(source_field)
        if reduction == "mse":
            return x.square().mean()
        elif reduction == "mae":
            return x.abs().mean()
        else:
            return x

    def _forward_impl_complex(
        self, x: Tensor, epsilon_r: Tensor, source_field: Optional[Tensor] = None, reduction: str = "mse"
    ) -> Tensor:
        """
        Complex field and complex permittivity in a lossy system
        Laplacian(E) + omega^2 * mu_0 * epsilon_0 * epsilon_r * E + J = 0, where epsilon_r = epsilon' - j* epsilon'' is complex-valued\\
        Laplacian(E) - gamma^2 * E + J = 0

        Args:
            x: (Tensor): Padded frequency-domain electric field \\
            epsilon_r (Tensor): Padded relative electric permittivity field \\
            source_field (Complex Tensor): Padded input light source field

        Returns:
            Tensor: Residual loss from Faraday equation
        """
        Hy = x  # for TM mode, we solve Hy (vertical H field)
        if self.dxx_kernel.device != x.device:
            self.dxx_kernel = self.dxx_kernel.to(x.device)
        dxx_Hy = torch.complex(
            F.conv2d(
                Hy.real,
                weight=self.dxx_kernel.expand(Hy.size(1), -1, -1, -1),
                padding=(1, 0),
                groups=Hy.size(1),
            ),
            F.conv2d(
                Hy.imag,
                weight=self.dxx_kernel.expand(Hy.size(1), -1, -1, -1),
                padding=(1, 0),
                groups=Hy.size(1),
            ),
        )
        if self.dx_kernel.device != x.device:
            self.dx_kernel = self.dx_kernel.to(x.device)
        dx_Hy = torch.complex(
            F.conv2d(
                Hy.real,
                weight=self.dx_kernel.expand(Hy.size(1), -1, -1, -1),
                padding=(1, 0),
                groups=Hy.size(1),
            ),
            F.conv2d(
                Hy.imag,
                weight=self.dx_kernel.expand(Hy.size(1), -1, -1, -1),
                padding=(1, 0),
                groups=Hy.size(1),
            ),
        )
        dx_epsilon_r = torch.complex(
            F.conv2d(epsilon_r.real, weight=self.dx_kernel, padding=(1, 0)),
            F.conv2d(epsilon_r.imag, weight=self.dx_kernel, padding=(1, 0)),
        )
        coeff = self.beta_sq * epsilon_r - self.beta_sq  # (k2-beta^2)
        x = dxx_Hy + coeff * Hy - 1 / epsilon_r * dx_epsilon_r * dx_Hy
        if source_field is not None:
            x = x.add(source_field)
        if reduction == "mse":
            return torch.view_as_real(x).square().mean()
        elif reduction == "mae":
            return torch.view_as_real(x).abs().mean()
        else:
            return x

    def forward(
        self, x: Tensor, epsilon_r: Tensor, source_field: Optional[Tensor] = None, reduction: str = "mse"
    ) -> Tensor:
        """
        Args:
            x: (Real/Complex Tensor): Padded frequency-domain electric field \\
            epsilon_r (Real/Complex Tensor): Padded relative electric permittivity field \\
            source_field (Complex Tensor): Padded input light source field

        Return:
            y: (Tensor): Residual loss from Faraday equation
        """
        if x.is_complex():
            return self._forward_impl_complex(x, epsilon_r, source_field, reduction=reduction)
        else:
            return self._forward_impl_real(x, epsilon_r, source_field, reduction=reduction)

    def Hy_to_E(self, Hy: Tensor, epsilon_r: Tensor) -> Tuple[Tensor, Tensor]:
        # this function uses standard coordinate
        # from vertical H to prop E
        if self.dx_kernel.device != Hy.device:
            self.dx_kernel = self.dx_kernel.to(Hy.device)
        dx_Hy = torch.complex(
            F.conv2d(Hy.real, weight=self.dx_kernel, padding=(1, 0)),
            F.conv2d(Hy.imag, weight=self.dx_kernel, padding=(1, 0)),
        )
        omega_epsilon = self.angular_freq * self.vacuum_permittivity * epsilon_r
        Ez = -1 / (1j * omega_epsilon) * dx_Hy * 1e6  # standard unit

        # from vertical H to cross E
        Ex = 2 * np.pi / (self.wavelength / 1e6) / (omega_epsilon) * Hy
        # return prop E and cross E
        return Ez, Ex


class CurlLoss(torch.nn.modules.loss._Loss):
    def __init__(self, grid_step: float, wavelength: float) -> None:
        """2-D Frequency-domain field residue based on Maxwell's Equation. z: prop, x: cross, y: vertical\\
            Assume no source or current. \\
        x->y->z: clock-wise: \\
            dz_Ex - dx_Ez = j * omega * mu0 * Hy \\
            -dz_Hy = -j * omega * eps * Ex \\
            dx_Hy = -j * omega * eps * Ez \\
        x'->y->z: counter-clockwise: negate the cross-axis x\\
            dz_Ex - dx_Ez = -j* omega * mu0 * Hy\\
            -dz_Hy = j * omega * eps * Ex \\
            -dx_Hy = -j * omega * eps * Ez \\
        Args:
            grid_step (float): The step size of the grid in the unit of um
            wavelength (float): Wavelength lambda in the unit of um
        """
        super().__init__()
        self.grid_step = grid_step
        self.vacuum_permeability = 1.25663706e-6
        self.vacuum_permittivity = 8.85418782e-12
        self.c = 1 / (self.vacuum_permeability * self.vacuum_permittivity) ** 0.5
        self.angular_freq = 2 * np.pi * self.c / (wavelength / 1e6)
        self.wavelength = wavelength
        self.scale = 1e15
        ## square of phase propagation factor in vacuum: coeff = beta^2 = omega^2 * mu_0 * epsilon_0
        self.beta_sq = 4 * np.pi ** 2 / (wavelength ** 2)
        # from top to bottom is postive x direction
        self.dx_kernel = torch.tensor([1, -8, 0, 8, -1]).view(1, 1, 5, 1) / (
            12 * self.grid_step
        )  # partial derivative along cross direction, 5-point
        # from left to right is postive z direction
        self.dz_kernel = torch.tensor([1, -8, 0, 8, -1]).view(1, 1, 1, 5) / (
            12 * self.grid_step
        )  # partial derivative along prop direction, 5-point

        self.dx_kernel = torch.tensor([-1, 0, 1]).view(1, 1, 3, 1) / (
            2 * self.grid_step
        )  # partial derivative along cross direction, 5-point
        # from left to right is postive z direction
        self.dz_kernel = torch.tensor([-1, 0, 1]).view(1, 1, 1, 3) / (
            2 * self.grid_step
        )  # partial derivative along prop direction, 5-point
        # the positive vertical direction y is out of the surface

    def _forward_impl_real(self, x: Tensor, epsilon: Tensor, reduction: str = "mse") -> Tensor:
        """
        Real field and real permittivity in a lossless system\\
        Laplacian(E) + omega^2 * mu_0 * epsilon_0 * epsilon_r * E + J = 0\\
        Laplacian(E) + beta^2 * E + J = 0

        Args:
            x: (Tensor): Padded frequency-domain TM-mode [Ez, Ex, Hy] field \\
        Returns:
            Tensor: Residual loss from Faraday equation
        """
        Ez, Ex, Hy = x.chunk(3, dim=1)  # for TM mode, we have Ez: prop, Ex: cross, Hy: vertical
        if self.dz_kernel.device != x.device:
            self.dz_kernel = self.dz_kernel.to(x.device)
        dz_Ex = F.conv2d(Ex, weight=self.dz_kernel, padding=(0, 2))
        if self.dx_kernel.device != x.device:
            self.dx_kernel = self.dx_kernel.to(x.device)
        dx_Ez = F.conv2d(Ez, weight=self.dx_kernel, padding=(2, 0))

        x = dz_Ex - dx_Ez + 1j * self.angular_freq * self.vacuum_permeability * Hy / 1e6  # um

        if reduction == "mse":
            return x.square().mean()
        elif reduction == "mae":
            return x.abs().mean()
        else:
            return x

    def _forward_impl_complex(self, x: Tensor, epsilon: Tensor, reduction: str = "mse") -> Tensor:
        """
        Complex field and complex permittivity in a lossless system
        x'->y->z: counter-clockwise: negate the cross-axis x\\
            dz_Ex - dx_Ez = -j* omega * mu0 * Hy\\
            -dz_Hy = j * omega * eps * Ex \\
            -dx_Hy = -j * omega * eps * Ez \\

        Args:
            x: (Tensor): Padded frequency-domain electric field \\
            epsilon_r (Tensor): Padded relative electric permittivity field

        Returns:
            Tensor: Residual loss from Faraday equation
        """
        Ez, Ex, Hy = x.chunk(3, dim=1)  # for TM mode, we have Ez: prop, Ex: cross, Hy: vertical
        """dz_Ex - dx_Ez = -j* omega * mu0 * Hy: counter-clockwise"""
        if self.dz_kernel.device != x.device:
            self.dz_kernel = self.dz_kernel.to(x.device)
        dz_Ex = torch.complex(
            F.conv2d(Ex.real, weight=self.dz_kernel, padding=(0, self.dz_kernel.size(3) // 2)),
            F.conv2d(Ex.imag, weight=self.dz_kernel, padding=(0, self.dz_kernel.size(3) // 2)),
        )

        if self.dx_kernel.device != x.device:
            self.dx_kernel = self.dx_kernel.to(x.device)
        dx_Ez = torch.complex(
            F.conv2d(Ez.real, weight=self.dx_kernel, padding=(self.dx_kernel.size(2) // 2, 0)),
            F.conv2d(Ez.imag, weight=self.dx_kernel, padding=(self.dx_kernel.size(2) // 2, 0)),
        )
        # print(Ez.abs().mean(), Ex.abs().mean(), dx_Ez.abs().mean(), dz_Ex.abs().mean(), Hy.abs().mean())
        res1 = dz_Ex - dx_Ez + 1j * self.angular_freq * self.vacuum_permeability * Hy / 1e6  # um

        """-dz_Hy = +j * omega * eps * Ex: counter-clockwise"""
        dz_Hy = torch.complex(
            F.conv2d(Hy.real, weight=self.dz_kernel, padding=(0, self.dz_kernel.size(3) // 2)),
            F.conv2d(Hy.imag, weight=self.dz_kernel, padding=(0, self.dz_kernel.size(3) // 2)),
        )
        permittivity_by_omega = epsilon * (1j * self.vacuum_permittivity * self.angular_freq / 1e6)
        res2 = dz_Hy + permittivity_by_omega * Ex  # um

        """-dx_Hy = -j * omega * eps * Ez: counter-clockwise"""
        dx_Hy = torch.complex(
            F.conv2d(Hy.real, weight=self.dx_kernel, padding=(self.dx_kernel.size(2) // 2, 0)),
            F.conv2d(Hy.imag, weight=self.dx_kernel, padding=(self.dx_kernel.size(2) // 2, 0)),
        )
        res3 = dx_Hy - permittivity_by_omega * Ez  # um
        # print(res1.abs().mean(), res2.abs().mean(), res3.abs().mean())
        x = res1 + 100 * res2 + 300 * res3
        # exit(0)
        if reduction == "mse":
            return torch.view_as_real(x).square().mean()
        elif reduction == "mae":
            return torch.view_as_real(x).abs().mean()
        else:
            return x

    def forward(self, x: Tensor, epsilon: Tensor, reduction: str = "mse") -> Tensor:
        """
        Args:
            x: (Real/Complex Tensor): Padded frequency-domain electric field \\
            epsilon_r (Real/Complex Tensor): Padded relative electric permittivity field \\
            source_field (Complex Tensor): Padded input light source field

        Return:
            y: (Tensor): Residual loss from Faraday equation
        """
        if x.is_complex():
            return self._forward_impl_complex(x, epsilon, reduction=reduction)
        else:
            return self._forward_impl_real(x, epsilon, reduction=reduction)


class PoyntingLoss(torch.nn.modules.loss._Loss):
    def __init__(self, grid_step: float, wavelength: float) -> None:
        """2-D Frequency-domain field residue based on Maxwell's Equation. z: prop, x: cross, y: vertical\\
            Assume no source or current. \\
        x->y->z: clock-wise: \\
            dz_Ex - dx_Ez = j * omega * mu0 * Hy \\
            -dz_Hy = -j * omega * eps * Ex \\
            dx_Hy = -j * omega * eps * Ez \\
        x'->y->z: counter-clockwise: negate the cross-axis x\\
            dz_Ex - dx_Ez = -j* omega * mu0 * Hy\\
            -dz_Hy = j * omega * eps * Ex \\
            -dx_Hy = -j * omega * eps * Ez \\
        Args:
            grid_step (float): The step size of the grid in the unit of um
            wavelength (float): Wavelength lambda in the unit of um
        """
        super().__init__()
        self.grid_step = grid_step
        self.vacuum_permeability = 1.25663706e-6
        self.vacuum_permittivity = 8.85418782e-12
        self.c = 1 / (self.vacuum_permeability * self.vacuum_permittivity) ** 0.5
        self.angular_freq = 2 * np.pi * self.c / (wavelength / 1e6)
        self.wavelength = wavelength
        self.scale = 1e15

    def flux_prob(self, x: Tensor) -> Tensor:
        _, Ex, Hy = x.chunk(3, dim=1)  # for TM mode, we have Ez: prop, Ex: cross, Hy: vertical
        Hy_z = (Hy[..., :-1] + Hy[..., 1:]) / 2
        Sx = Ex[..., :-1].mul(Hy_z.conj_()).real  # power density at each point
        flux = self.grid_step * Sx.sum(dim=2) / 2  # [bs, w] # accumulated power density over the profile
        return flux

    def poynting_vector_prob(self, x: Tensor) -> Tensor:
        _, Ex, Hy = x.chunk(3, dim=1)  # for TM mode, we have Ez: prop, Ex: cross, Hy: vertical
        Hy_z = (Hy[..., :-1] + Hy[..., 1:]) / 2
        Sx = Ex[..., :-1].mul(Hy_z.conj_()).real / 2  # power density at each point
        return Sx  # [bs, 1, h, w] real

    def _forward_impl_complex(self, x: Tensor, target: Tensor, reduction: str = "mse") -> Tensor:
        """
        Complex field and complex permittivity in a lossless system
        x'->y->z: counter-clockwise: negate the cross-axis x\\
            dz_Ex - dx_Ez = -j* omega * mu0 * Hy\\
            -dz_Hy = j * omega * eps * Ex \\
            -dx_Hy = -j * omega * eps * Ez \\

        Args:
            x: (Tensor): Padded frequency-domain electric field \\
            epsilon_r (Tensor): Padded relative electric permittivity field

        Returns:
            Tensor: Residual loss from Faraday equation
        """
        # x = self.flux_prob(x) - self.flux_prob(target)
        x = self.poynting_vector_prob(x) - self.poynting_vector_prob(target)

        # exit(0)
        if reduction == "mse":
            return x.square().mean()
        elif reduction == "mae":
            return x.abs().mean()
        else:
            return x

    def forward(self, x: Tensor, target: Tensor, reduction: str = "mse") -> Tensor:
        """
        Args:
            x: (Real/Complex Tensor): Padded frequency-domain electric field \\
            epsilon_r (Real/Complex Tensor): Padded relative electric permittivity field \\
            source_field (Complex Tensor): Padded input light source field

        Return:
            y: (Tensor): Residual loss from Faraday equation
        """
        if x.is_complex():
            return self._forward_impl_complex(x, target, reduction=reduction)
        else:
            return 0


def normalize(x):
    if isinstance(x, np.ndarray):
        x_min, x_max = np.percentile(x, 5), np.percentile(x, 95)
        x = np.clip(x, x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    else:
        x_min, x_max = torch.quantile(x, 0.05), torch.quantile(x, 0.95)
        x = x.clamp(x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    return x


def plot_compare_tmp(
    wavelength: Tensor,
    grid_step: Tensor,
    epsilon: Tensor,
    pred_fields: Tensor,
    target_fields: Tensor,
    filepath: str,
    pol: str = "Hz",
    norm: bool = True,
) -> None:
    if epsilon.is_complex():
        epsilon = epsilon.real
    # simulation = Simulation(omega, eps_r=epsilon.data.cpu().numpy(), dl=grid_step, NPML=[1,1], pol='Hz')
    field_val = pred_fields.data.cpu().numpy()
    target_field_val = target_fields.data.cpu().numpy()
    # eps_r = simulation.eps_r
    eps_r = epsilon.data.cpu().numpy()
    err_field_val = field_val - target_field_val
    # field_val = np.abs(field_val)
    field_val = field_val.real
    # target_field_val = np.abs(target_field_val)
    target_field_val = target_field_val.real
    err_field_val = np.abs(err_field_val)
    outline_val = np.abs(eps_r)

    # vmax = field_val.max()
    vmin = 0.0
    b = field_val.shape[0]
    h, w = field_val.shape[-2:]
    if w < 160:
        figsize = (3 * b, 9)
    else:
        figsize = (5 * b, 3.1)
    fig, axes = plt.subplots(3, b, constrained_layout=True, figsize=figsize)
    if b == 1:
        axes = axes[..., np.newaxis]
    cmap = "magma"
    # print(field_val.shape, target_field_val.shape, outline_val.shape)
    for i in range(b):
        vmax = np.max(target_field_val[i])
        if norm:
            h1 = axes[0, i].imshow(normalize(field_val[i]), cmap=cmap, origin="lower")
            h2 = axes[1, i].imshow(normalize(target_field_val[i]), cmap=cmap, origin="lower")
        else:
            h1 = axes[0, i].imshow(field_val[i], cmap=cmap, vmin=0, vmax=vmax, origin="lower")
            h2 = axes[1, i].imshow(target_field_val[i], cmap=cmap, vmin=0, vmax=vmax, origin="lower")
        h3 = axes[2, i].imshow(err_field_val[i], cmap=cmap, vmin=0, vmax=vmax, origin="lower")
        
        if w > 160:
            for j in range(3):
                divider = make_axes_locatable(axes[j, i])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar([h1, h2, h3][j], label=pol, ax=axes[j, i], cax=cax)
            axes[0, i].title.set_text(
                f"{wavelength[i,0].item():.2f} um, dh=({grid_step[i,0].item()*1000:.1f} nm x {grid_step[i,1].item()*1000:.1f} nm)"
            )
        else:
            pass
            # for j in range(3):
            #     divider = make_axes_locatable(axes[j, i])
            #     cax = divider.append_axes("right", size="10%", pad=0.05)
            #     fig.colorbar([h1, h2, h3][j], label=pol, ax=axes[j, i], cax=cax)
        # fig.colorbar(h2, label=pol, ax=axes[1,i])
        # fig.colorbar(h3, label=pol, ax=axes[2,i])

    # Do black and white so we can see on both magma and RdBu
    for ax in axes.flatten():
        ax.contour(outline_val[0], levels=2, linewidths=1.0, colors="w")
        ax.contour(outline_val[0], levels=2, linewidths=0.5, colors="k")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(filepath, dpi=150)
    plt.close()

def plot_compare(
    wavelength: Tensor,
    grid_step: Tensor,
    epsilon: Tensor,
    pred_fields: Tensor,
    target_fields: Tensor,
    filepath: str,
    pol: str = "Hz",
    norm: bool = True,
) -> None:
    if epsilon.is_complex():
        epsilon = epsilon.real
    # simulation = Simulation(omega, eps_r=epsilon.data.cpu().numpy(), dl=grid_step, NPML=[1,1], pol='Hz')
    field_val = pred_fields.data.cpu().numpy()
    target_field_val = target_fields.data.cpu().numpy()
    # eps_r = simulation.eps_r
    eps_r = epsilon.data.cpu().numpy()
    err_field_val = field_val - target_field_val
    # field_val = np.abs(field_val)
    field_val = field_val.real
    # target_field_val = np.abs(target_field_val)
    target_field_val = target_field_val.real
    err_field_val = np.abs(err_field_val)
    outline_val = np.abs(eps_r)
    
    # vmax = field_val.max()
    vmin = 0.0
    b = field_val.shape[0]
    h, w = field_val.shape[-2:]
    if w < 160:
        figsize = (3 * b, 9)
    else:
        figsize = (5 * b, 3.1)
    fig, axes = plt.subplots(3, b, constrained_layout=True, figsize=figsize)
    if b == 1:
        axes = axes[..., np.newaxis]
    cmap = "magma"
    # print(field_val.shape, target_field_val.shape, outline_val.shape)
    
    err_field_val = np.log10(err_field_val)
    
    
    for i in range(b):
        vmax = np.max(target_field_val[i])
        if norm:
            h1 = axes[0, i].imshow(normalize(field_val[i]), cmap=cmap, origin="lower")
            h2 = axes[1, i].imshow(normalize(target_field_val[i]), cmap=cmap, origin="lower")
        else:
            h1 = axes[0, i].imshow(field_val[i], cmap=cmap, vmin=0, vmax=vmax, origin="lower")
            h2 = axes[1, i].imshow(target_field_val[i], cmap=cmap, vmin=0, vmax=vmax, origin="lower")
        vmax = np.log10(10)
        vmin = np.log10(1e-1)
        
        h3 = axes[2, i].imshow(err_field_val[i], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        
        if w > 160:
            for j in range(3):
                divider = make_axes_locatable(axes[j, i])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar([h1, h2, h3][j], label=pol, ax=axes[j, i], cax=cax)
            axes[0, i].title.set_text(
                f"{wavelength[i,0].item():.2f} um, dh=({grid_step[i,0].item()*1000:.1f} nm x {grid_step[i,1].item()*1000:.1f} nm)"
            )
        else:
            pass
            # for j in range(3):
            #     divider = make_axes_locatable(axes[j, i])
            #     cax = divider.append_axes("right", size="10%", pad=0.05)
            #     fig.colorbar([h1, h2, h3][j], label=pol, ax=axes[j, i], cax=cax)
        # fig.colorbar(h2, label=pol, ax=axes[1,i])
        # fig.colorbar(h3, label=pol, ax=axes[2,i])

    # Do black and white so we can see on both magma and RdBu
    for ax in axes.flatten():
        ax.contour(outline_val[0], levels=2, linewidths=1.0, colors="w")
        ax.contour(outline_val[0], levels=2, linewidths=0.5, colors="k")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_dynamics(
    wavelength: Tensor,
    grid_step: Tensor,
    epsilon: Tensor,
    pred_fields: Tensor,
    target_fields: Tensor,
    filepath: str,
    eps_list,
    eps_text_loc_list,
    region_list,
    box_id,
    ref_eps,
    norm: bool = True,
    wl_text_pos=None,
    time=None,
    fps=None,
) -> None:
    if epsilon.is_complex():
        epsilon = epsilon.real
        ref_eps = ref_eps.real
    # simulation = Simulation(omega, eps_r=epsilon.data.cpu().numpy(), dl=grid_step, NPML=[1,1], pol='Hz')
    field_val = pred_fields.data.cpu().numpy()
    target_field_val = target_fields.data.cpu().numpy()
    # eps_r = simulation.eps_r
    eps_r = epsilon.data.cpu().numpy()
    err_field_val = field_val - target_field_val
    # field_val = np.abs(field_val)
    field_val = field_val.real
    # target_field_val = np.abs(target_field_val)
    target_field_val = target_field_val.real
    err_field_val = np.abs(err_field_val)
    outline_val = np.abs(ref_eps.data.cpu().numpy())

    # vmax = field_val.max()
    vmin = 0.0
    b = field_val.shape[0]
    fig, ax = plt.subplots(1, b, constrained_layout=True, figsize=(5.1, 1.15))
    # fig, ax = plt.subplots(1, b, constrained_layout=True, figsize=(5.2, 1))
    cmap = "magma"
    # print(field_val.shape, target_field_val.shape, outline_val.shape)
    i = 0
    vmax = np.max(target_field_val[i])
    if norm:
        h1 = ax.imshow(normalize(field_val[i]), cmap=cmap, origin="lower")
    else:
        h1 = ax.imshow(field_val[i], cmap=cmap, vmin=0, vmax=vmax, origin="lower")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h1, label="Mag.", ax=ax, cax=cax)
    for i, (eps, eps_pos, box) in enumerate(zip(eps_list, eps_text_loc_list, region_list)):
        xl, xh, yl, yh = box
        if i == box_id:
            color = "yellow"
        else:
            color = "white"
        ax.annotate(r"$\epsilon_r$"+f" = {eps:.3f}", xy=eps_pos, xytext=eps_pos, color=color)
        ax.plot((xl, xh), (yl, yl), linewidth=0.5, color=color)
        ax.plot((xl, xh), (yh, yh), linewidth=0.5, color=color)
        ax.plot((xl, xl), (yl, yh), linewidth=0.5, color=color)
        ax.plot((xh, xh), (yl, yh), linewidth=0.5, color=color)
    if wl_text_pos is not None:
        if box_id == len(region_list):
            color = "yellow"
        else:
            color = "white"
        ax.annotate(r"$\lambda$" + f" = {wavelength.item():.3f}", xy=wl_text_pos, xytext=wl_text_pos, color=color)
        # fig.colorbar(h2, label=pol, ax=axes[1,i])
        # fig.colorbar(h3, label=pol, ax=axes[2,i])
    # ax.annotate(f"Runtime = {time:.3f} s", xy=(field_val.shape[-1]//2-30, field_val.shape[-2]), xytext=(field_val.shape[-1]//2-40, field_val.shape[-2]+1), color="black", annotation_clip=False)
    if time is not None:
        ax.annotate(f"Runtime = {time:.3f} s, FPS = {fps:.1f}", xy=(field_val.shape[-1]//2-110, field_val.shape[-2]+3), xytext=(field_val.shape[-1]//2-110, field_val.shape[-2]+3), color="black", annotation_clip=False)

    if box_id == len(region_list) + 1:
        color = "blue"
    else:
        color = "black"
    ax.annotate(r"$l_z$" + f" = {grid_step[..., 0].item()*field_val.shape[-1]:.2f} " + r"$\mu m$", xy=(field_val.shape[-1]//2-30, -15), xytext=(field_val.shape[-1]//2-30, -15), color=color, annotation_clip=False)
    ax.annotate(r"$l_x$" + f" = {grid_step[..., 1].item()*field_val.shape[-2]:.2f} " + r"$\mu m$", xy=(-18, field_val.shape[-2]//2-44), xytext=(-18, field_val.shape[-2]//2-44), color=color, annotation_clip=False, rotation=90)


    # Do black and white so we can see on both magma and RdBu

    ax.contour(outline_val[0], levels=1, linewidths=1.0, colors="w")
    ax.contour(outline_val[0], levels=1, linewidths=0.5, colors="k")
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


@lru_cache(maxsize=64, typed=True)
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(
            img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range
        )

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).to(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


class HOGLayer(nn.Module):
    def __init__(self, nbins=10, pool=8, max_angle=np.pi, stride=1, padding=1, dilation=1):
        super(HOGLayer, self).__init__()
        self.nbins = nbins
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool = pool
        self.max_angle = max_angle
        mat = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.register_buffer("weight", mat[:, None, :, :])
        self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)

    def forward(self, x):
        with torch.no_grad():
            gxy = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, 1)
            # 2. Mag/ Phase
            mag = gxy.norm(dim=1)
            norm = mag[:, None, :, :]
            phase = torch.atan2(gxy[:, 0, :, :], gxy[:, 1, :, :])

            # 3. Binning Mag with linear interpolation
            phase_int = phase / self.max_angle * self.nbins
            phase_int = phase_int[:, None, :, :]

            n, c, h, w = gxy.shape
            out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
            out.scatter_(1, phase_int.floor().long() % self.nbins, norm)
            out.scatter_add_(1, phase_int.ceil().long() % self.nbins, 1 - norm)

            return self.pooler(out)


class ComplexHOGLayer(nn.Module):
    def __init__(self, nbins=10, pool=8, max_angle=np.pi, stride=1, padding=1, dilation=1):
        super(HOGLayer, self).__init__()
        self.nbins = nbins
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool = pool
        self.max_angle = max_angle
        mat = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.register_buffer("weight", mat[:, None, :, :])
        self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)

    def forward(self, x):
        with torch.no_grad():
            gxy = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, 1)
            # 2. Mag/ Phase
            mag = gxy.norm(dim=1)
            norm = mag[:, None, :, :]
            phase = torch.atan2(gxy[:, 0, :, :], gxy[:, 1, :, :])

            # 3. Binning Mag with linear interpolation
            phase_int = phase / self.max_angle * self.nbins
            phase_int = phase_int[:, None, :, :]

            n, c, h, w = gxy.shape
            out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
            out.scatter_(1, phase_int.floor().long() % self.nbins, norm)
            out.scatter_add_(1, phase_int.ceil().long() % self.nbins, 1 - norm)

            return self.pooler(out)
