'''
 # @ Author: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Create Time: 2024-03-01 15:32:39
 # @ Modified by: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Modified time: 2024-10-21 00:49:54
 # @ Description: dataset and dataloader support for h5py datasets
 '''
import h5py
import os
import glob
import numpy as np
import torch

from torch import Tensor
from torchvision import transforms
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.transforms import InterpolationMode
from torch.nn.functional import pad


resize_modes = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

__all__ = ["MMI", "MMIDataset"]

# load data and process using h5py format
# support two options: one is resize, the other is return raw data, which will be resize and padded in the loader as a collate function

class MMI(VisionDataset):
    """Basic dataset for mmi data, support load multiple devices"""
    ### process and load data
    url = None
    train_filename = "training"
    test_filename = "test"
    folder = "mmi"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_ratio: float = 0.7,
        resize: bool = False,
        resize_size: List[int] = [80, 384], # height x width
        resize_mode: str = "bilinear",
        # resize_style: str = "trim", # trim or full size
        normalize: bool = True,
        data_ratio: float = 1, # last idx we choose to extract from the data
        pol_list: List[str] = ["Hz"],
        device_list: List[str] = ["mmi_3x3_L_random_slots"],
        processed_dir: str = "processed",
        download: bool = False,
    ) -> None:
        self.processed_dir = processed_dir
        root = os.path.join(os.path.expanduser(root), self.folder)
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self.normalize = normalize
        self.resize = resize
        self.resize_mode = resize_mode
        self.resize_size = resize_size
        # self.resize_style = resize_style
        
        self.train = train  # training set or test set
        self.train_ratio = train_ratio
        self.pol_list = sorted(pol_list)
        # self.filenames = [f"{pol}{self.filename_suffix}" for pol in self.pol_list] # files we need to load
        self.data_ratio = data_ratio
        self.device_list = device_list
        
        self.filenames = []
        
        if self.resize:
            resize_note = f"{self.resize_size[0]}x{self.resize_size[1]}_{self.resize_mode}_{self.data_ratio}"
        else:
            resize_note = f'raw_{self.data_ratio}'
        self.train_filename = f'{self.train_filename}_{resize_note}'
        self.test_filename = f'{self.test_filename}_{resize_note}'

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        self.process_raw_data() # save raw data into processed pt files for next step
        
        self.wavelength: Any = []
        self.grid_step: Any = []
        self.eps: Any = []
        self.input_mode: Any = []
        self.fields: Any = []
        self.mask: Any = []
        
        self.eps_min = torch.tensor(1.0)
        self.eps_max = torch.tensor(12.3)
        self.wavelength, self.grid_step, self.eps, self.input_mode, self.fields, self.mask = self.load(train=train)
        print(f"Loaded {len(self.eps)} files")

    def process_raw_data(self) -> None:
        processed_dir = os.path.join(self.root, self.processed_dir)
        processed_training_file = os.path.join(processed_dir, f"{self.train_filename}.pt")
        processed_test_file = os.path.join(processed_dir, f"{self.test_filename}.pt")
        
        print(f"processed_dir: {processed_dir}")
        print(f"processed_training_file: {processed_training_file}")
        print(f"processed_test_file: {processed_test_file}")

        if os.path.exists(processed_training_file) and os.path.exists(processed_test_file):
            print("Data already processed")
            return
        else:
            print("Start processing data")

        wavelength, grid_step, eps, input_mode, fields, mask = self._load_dataset()
        (
            wavelength_train,
            wavelength_test,
            grid_step_train,
            grid_step_test,
            eps_train,
            eps_test,
            input_mode_train,
            input_mode_test,
            fields_train,
            fields_test,
            mask_train,
            mask_test,
        ) = self._split_dataset(wavelength, grid_step, eps, input_mode, fields, mask)
        
        self._save_dataset(
            wavelength_train,
            grid_step_train,
            eps_train,
            input_mode_train,
            fields_train,
            mask_train,
            wavelength_test,
            grid_step_test,
            eps_test,
            input_mode_test,
            fields_test,
            mask_test,
            processed_dir,
            self.train_filename,
            self.test_filename,
        )


    def resize_helper_fn(self, x: Tensor, grid_step: Tensor, size: List[int], mode: str) -> Tuple[Tensor, Tensor]:
        """Resize function to resize the data to the desired size"""
        # torch.nn.functional.interpolate(data, size=size, mode=mode)
        # x: bs, n, 1, h, w or bs, 1, h, w
        # grid_step: bs, n, 2 or bs, 2
        if not isinstance(x, torch.Tensor):
            y = torch.from_numpy(x)
        else:
            y = x
        
        y = y.view(-1, 1, x.shape[-2], x.shape[-1]) # bs, 1, h, w
        
        old_size = y.shape[-2:]
        new_size = size
        grid_step_scaling = torch.tensor([old_size[0]/new_size[0], old_size[1]/new_size[1]])
        new_grid_step = grid_step * grid_step_scaling # hadamard product to scale the grid step
        
        if y.is_complex():
            y = torch.complex(
                torch.nn.functional.interpolate(y.real, size=size, mode=mode),
                torch.nn.functional.interpolate(y.imag, size=size, mode=mode),
            )
        else:
            y = torch.nn.functional.interpolate(y, size=size, mode=mode)
            
        y = y.view(list(x.shape[:-2]) + list(size))
        
        return y, new_grid_step
    
    def _load_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        for device in self.device_list:
            ### support for multiple devices
            for pol in self.pol_list:
                print(f'{device}-{pol}')
                if device in pol:
                    all_samples = [f'{pol}/{os.path.basename(i)}'for i in glob.glob(os.path.join(self.root, "raw", pol, f"{device}*.h5"))]
                    all_samples = sorted(all_samples)[:int(len(all_samples) * self.data_ratio)]
                    self.filenames.extend(all_samples)
        
        eps_list =[]
        fields_list = []
        wavelength_list = []
        grid_step_list = []
        input_len_list = []
        input_mode_list = []
        mask_list = []
        for device_file in self.filenames:
            with h5py.File(os.path.join(self.root, "raw", device_file), "r") as f:
                eps = torch.from_numpy(f["epsilon"][()]).float()# [N, 1, h, w]
                fields = torch.from_numpy(f["fields"][()]).to(torch.complex64)  # [N, 1, h, w]
                wavelength = torch.from_numpy(f["wavelength"][()]).to(torch.float16) # [N, 1]
                grid_step = torch.from_numpy(f["grid_step"][()]).to(torch.float16) # [N, 2]
                input_len = torch.from_numpy(f["input_len"][()]).to(torch.float16) # [N, 1]
                if self.resize:
                    eps, new_grid_step = self.resize_helper_fn(eps, grid_step, self.resize_size, self.resize_mode)
                    fields, _ = self.resize_helper_fn(fields, grid_step, self.resize_size, self.resize_mode)
                    if self.normalize:
                        mag = fields.abs() # N, 1, h, w
                        mag_mean = mag.mean(dim=(0, 2, 3))  # No need for keepdim=True if we process only one channel
                        if mag_mean > 1e-18:
                            mag_std = mag.std(dim=(0, 2, 3))
                            fields /= mag_std * 2
                        epsilon_min = 1.0
                        epsilon_max = 12.3
                        eps = (eps - epsilon_min) / (epsilon_max - epsilon_min) # bg is set to 1, okay for this logic
                    
                    input_len = (input_len * grid_step[0, 1] / new_grid_step[0, 1]).int()
                    grid_step = new_grid_step

                grid_step = grid_step[:, [1, 0]] # change the order to adpat to wave prior
                input_mode = fields.detach().clone()
                input_mask = torch.zeros(
                    input_mode.shape[0], 1, 1, input_mode.shape[-1]
                ) # [bs, N, 1, 1, 1, w]
                input_mask[..., :int(input_len[0, 0])].fill_(1)
                input_mode.mul_(input_mask) # [N, 1, h, w]

                mask = torch.ones_like(eps).to(torch.bool) # [N, 1, h, w]

                eps_list.append(eps)
                fields_list.append(fields)
                wavelength_list.append(wavelength)
                grid_step_list.append(grid_step)
                input_len_list.append(input_len)
                input_mode_list.append(input_mode)
                mask_list.append(mask)
        
        eps = torch.stack(eps_list)
        fields = torch.stack(fields_list)
        wavelength = torch.stack(wavelength_list)
        grid_step = torch.stack(grid_step_list)
        input_len = torch.stack(input_len_list)
        input_mode = torch.stack(input_mode_list)
        mask = torch.stack(mask_list)
        
        return wavelength, grid_step, eps, input_mode, fields, mask
   
    def _split_dataset(
        self, wavelength, grid_step, eps, input_mode, fields, mask
    ) -> Tuple[Tensor, ...]:
        from sklearn.model_selection import train_test_split

        (
            wavelength_train,
            wavelength_test,
            grid_step_train,
            grid_step_test,
            eps_train,
            eps_test,
            input_mode_train,
            input_mode_test,
            fields_train,
            fields_test,
            mask_train,
            mask_test,
        ) = train_test_split(
            wavelength, grid_step, eps, input_mode, fields, mask, train_size=self.train_ratio, random_state=42
        )

        print(
            f"training: {wavelength_train.shape[0]} device examples, "
            f"test: {wavelength_test.shape[0]} device examples"
        )
        return (
            wavelength_train,
            wavelength_test,
            grid_step_train,
            grid_step_test,
            eps_train,
            eps_test,
            input_mode_train,
            input_mode_test,
            fields_train,
            fields_test,
            mask_train,
            mask_test,
        )
    def _preprocess_dataset(self, data_train: Tensor, data_test: Tensor) -> Tuple[Tensor, Tensor]:
        return data_train, data_test

    @staticmethod
    def _save_dataset(
        # data_train: List,
        # data_test: List,
        wavelength_train: Tensor,
        grid_step_train: Tensor,
        eps_train: Tensor,
        input_mode_train: Tensor,
        fields_train: Tensor,
        mask_train: Tensor,
        wavelength_test: Tensor,
        grid_step_test: Tensor,
        eps_test: Tensor,
        input_mode_test: Tensor,
        fields_test: Tensor,
        mask_test: Tensor,
        processed_dir: str,
        train_filename: str = "training",
        test_filename: str = "test",
    ) -> None:
        os.makedirs(processed_dir, exist_ok=True)
        processed_training_file = os.path.join(processed_dir, f"{train_filename}.pt")
        processed_test_file = os.path.join(processed_dir, f"{test_filename}.pt")
        
        with open(processed_training_file, "wb") as f:
            torch.save((wavelength_train, grid_step_train, eps_train, input_mode_train, fields_train, mask_train), f)
        
        with open(processed_test_file, "wb") as f:
            torch.save((wavelength_test, grid_step_test, eps_test, input_mode_test, fields_test, mask_test), f)
        
        print(f"Processed data filenames + start_index saved")

    def load(self, train: bool = True):
        filename = f"{self.train_filename}.pt" if train else f"{self.test_filename}.pt"
        print(f"loading {filename}")
        
        with open(os.path.join(self.root, self.processed_dir, filename), "rb") as f:
            wavelength, grid_step, eps, input_mode, fields, mask= torch.load(f)
            if isinstance(eps, np.ndarray):
                raise NotImplementedError
                eps = torch.from_numpy(eps)
            if isinstance(grid_step, np.ndarray):
                raise NotImplementedError
                grid_step = torch.from_numpy(grid_step)
            if isinstance(wavelength, np.ndarray):
                raise NotImplementedError
                wavelength = torch.from_numpy(wavelength)
            if isinstance(input_mode, np.ndarray):
                raise NotImplementedError
                input_mode = torch.from_numpy(input_mode)
            if isinstance(mask, np.ndarray):
                raise NotImplementedError
                mask = torch.from_numpy(input_mode)
            if isinstance(fields, np.ndarray):
                raise NotImplementedError
                fields = torch.from_numpy(fields)
        return wavelength, grid_step, eps, input_mode, fields, mask
        

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

    def _check_integrity(self) -> bool:
        return all([os.path.exists(os.path.join(self.root, "raw", filename)) for filename in self.filenames])

    def __len__(self):
        return len(self.eps)
    
    def __getitem__(self, item):
        return self.wavelength[item], self.grid_step[item], self.eps[item], self.input_mode[item],  self.fields[item], self.mask[item]

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class MMIDataset:
    def __init__(
        self,
        root: str,
        split: str,
        test_ratio: float,
        train_valid_split_ratio: List[float],
        device_list: List[str],
        normalize: bool = True,
        resize: bool = True,
        resize_size: List[int] = [80, 384], # height x width
        resize_mode: str = "bilinear",
        # resize_style: str = "trim", # trim or full size
        pol_list: List[str] = None,
        data_ratio: float = 1, # last idx we choose to extract from the data
        processed_dir: str = "processed",
    ):
        self.root = root
        self.split = split
        self.test_ratio = test_ratio
        assert 0 < test_ratio < 1, print(f"Only support test_ratio from (0, 1), but got {test_ratio}")
        self.train_valid_split_ratio = train_valid_split_ratio
        self.resize = resize
        self.resize_size = resize_size
        self.resize_mode = resize_mode
        self.normalize = normalize
        self.data = None
        self.eps_min = None
        self.eps_max = None
        self.device_list = sorted(device_list)
        self.pol_list = sorted(pol_list)
        self.processed_dir = processed_dir
        self.data_ratio = data_ratio

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        tran = [
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(tran)

        if self.split == "train" or self.split == "valid":
            train_valid = MMI(
                self.root,
                train=True,
                download=True,
                transform=transform,
                train_ratio=1 - self.test_ratio,
                pol_list=self.pol_list,
                processed_dir=self.processed_dir,
                device_list=self.device_list,
                # adapt to NeuroOLight
                resize=self.resize,
                resize_size=self.resize_size,
                resize_mode=self.resize_mode,
                normalize=self.normalize,
                data_ratio=self.data_ratio,
            )
            self.eps_min = train_valid.eps_min
            self.eps_max = train_valid.eps_max

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            if self.train_valid_split_ratio[0] + self.train_valid_split_ratio[1] > 0.99999:
                valid_len = len(train_valid) - train_len
            else:
                valid_len = int(self.train_valid_split_ratio[1] * len(train_valid))
                
                train_valid.input_mode = train_valid.input_mode[:train_len+valid_len]
                train_valid.eps = train_valid.eps[:train_len+valid_len]
                train_valid.mask = train_valid.mask[:train_len+valid_len]
                train_valid.wavelength = train_valid.wavelength[:train_len+valid_len]
                train_valid.grid_step = train_valid.grid_step[:train_len+valid_len]
                train_valid.fields = train_valid.fields[:train_len+valid_len]

            split = [train_len, valid_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1)
            )

            if self.split == "train":
                self.data = train_subset
            else:
                self.data = valid_subset

        else:
            test = MMI(
                self.root,
                train=False,
                download=True,
                transform=transform,
                train_ratio=1 - self.test_ratio,
                pol_list=self.pol_list,
                processed_dir=self.processed_dir,
                device_list=self.device_list,
                # adapt to NeuroOLight
                resize=self.resize,
                resize_size=self.resize_size,
                resize_mode=self.resize_mode,
                normalize=self.normalize,
                data_ratio=self.data_ratio,
            )

            self.data = test
            self.eps_min = test.eps_min
            self.eps_max = test.eps_max

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __call__(self, index: int) -> Dict[str, Tensor]:
        return self.__getitem__(index)