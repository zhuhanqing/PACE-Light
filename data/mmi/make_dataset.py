'''
 # @ Author: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Create Time: 2024-10-20 15:19:47
 # @ Modified by: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Modified time: 2024-10-23 22:40:16
 # @ Description: Generate simulation data for MMI devices
 '''

# add angler to path (not necessary if pip installed)
import sys

import matplotlib.pylab as plt
from pyutils.config import Config
import numpy as np
import torch
from tqdm import tqdm
import time
import h5py

from device_shape import (
    mmi_3x3_L_random_slots,
    mmi_5x5_L_random_slots,
)

sys.path.append("..")

from itertools import product

# import the main simulation and optimization classes
from angler import Optimization, Simulation


class SimulationConfig(Config):
    def __init__(self):
        super().__init__()
        self.update(
            dict(
                device=dict(
                    type="",
                    cfg=dict(),
                ),
                simulation=dict(),
            )
        )
        
def generate_slot_mmi_random_raw_data_h5py(configs, name, data_dir):
    #NOTE(hqzhu): generate original resolution image
    # save epsilon, field, grid_step, wavelength, input_len, padding
    # each epsilon combination randomly sample an MMI box size, treat them as unified permittivies distribution
    c0 = 299792458  # speed of light in vacuum (m/s)
    source_amp = 1e-9  # amplitude of modal source (make around 1 for nonlinear effects)
    neff_si = 3.48

    import os

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    for idx, config in enumerate(configs):
        print(f"Generating data with config:\n\t{config} ({idx:4d}/{len(configs):4d})")
        pol, device_fn, eps_val, n_points, wavelengths, size, random_state, grid_step, npwl, n_sampled_slots_range, taper_size, port_len, normalize_flag = config

        device_id = 0 # start with 0
        # device_id = 800
        for idx_points in tqdm(range(device_id, n_points)): 
            np.random.seed(random_state + device_id)
            device = device_fn(grid_step=grid_step, NPWL=(npwl, npwl), n_sampled_slots_range=n_sampled_slots_range, taper_size=taper_size, port_len=port_len)  # re-sample device shape
            device_id += 1
            for wavelength in wavelengths:
                # save one point one wavelength into one file with n_ports data
                file_name = f"./{data_dir}/{name}_{idx_points}.h5"
                
                lambda0 = wavelength / 1e6  # free space wavelength (m)
                omega = 2 * np.pi * c0 / lambda0  # angular frequency (2pi/s)
                eps_map = device.set_pad_eps(np.zeros([len(device.pad_regions)]) + eps_val)
                
                epsilon_list_tmp = []
                field_list_tmp = []
                grid_step_list_tmp = []
                wavelength_list_tmp = []
                input_len_list_tmp = []
                for i in range(device.num_in_ports):
                    simulation = Simulation(omega, eps_map, device.grid_step, device.NPML, pol)
                    simulation.add_mode(
                        neff=neff_si,
                        direction_normal="x",
                        center=device.in_port_centers_px[i],
                        width=int(2 * device.in_port_width_px[i]),
                        scale=source_amp,
                    )
                    simulation.setup_modes()
                    (Ex, Ey, Hz) = simulation.solve_fields()

                    if pol == "Hz":
                        # 3, h, w
                        field = np.stack(
                            [
                                # simulation.fields["Ex"],
                                # simulation.fields["Ey"],
                                simulation.fields["Hz"],
                            ],
                            axis=0,
                        )
                    else:
                        field = np.stack(
                            [
                                # simulation.fields["Hx"],
                                # simulation.fields["Hy"],
                                simulation.fields["Ez"],
                            ],
                            axis=0,
                        )
                    #NOTE(hqzhu): here we trim down the image and save the original image
                    eps_map_trim = device.trim_pml(eps_map)
                    field_trim = device.trim_pml(field)
                    
                    assert (eps_map_trim.shape[-2:] == field_trim.shape[-2:])

                    epsilon_list_tmp.append(np.stack([
                        eps_map_trim
                        ], axis=0)) # keep the dim as 1, h, w
                    field_list_tmp.append(field_trim)
                    
                    wavelength_list_tmp.append(np.array([wavelength]))
                    #NOTE(hqzhu): here we set the unified grid step 
                    grid_step_list_tmp.append(np.array([grid_step, grid_step]))
                    input_len_list_tmp.append(np.array([int((device.port_len+device.taper_len) / grid_step)]))

                # NOTE(hqzhu): save a simulation config here
                device_cfg = dict(
                    num_in_ports=device.num_in_ports,
                    num_out_ports=device.num_out_ports,
                    box_size=device.box_size,
                    wg_width=device.wg_width,
                    port_diff=device.port_diff,
                    port_len=device.port_len,
                    taper_width=device.taper_width,
                    taper_len=device.taper_len,
                    eps_r=device.eps_r,
                    eps_bg=device.eps_bg,
                    NPML=device.NPML,
                )
                
                simulation_cfg = dict(
                    pol=pol,
                    wavelength=wavelength,
                    grid_step=grid_step,
                    input_len=int((device.port_len+device.taper_len) / grid_step),
                )
                
                cfg = SimulationConfig()
                cfg.device.type = (device.__class__.__name__)
                cfg.device.update(dict(cfg=device_cfg))
                cfg.update(dict(simulation=simulation_cfg))

                # NOTE(hqzhu): add normalize logic here to avoid processing in dataset
                
                fields = np.stack(field_list_tmp, axis=0)
                
                if normalize_flag:
                    # Compute the magnitude of the complex numbers
                    mag = np.abs(fields)
                    mag_mean = mag.mean(axis=(0, 2, 3))
                    if mag_mean > 1e-18:
                        mag_std = mag.std(axis=(0, 2, 3))
                        fields /= mag_std * 2
                
                with h5py.File(file_name, 'w') as f:
                    f.create_dataset('epsilon', data=np.stack(epsilon_list_tmp, axis=0).astype(np.float16).transpose(0, 1, 3, 2)) # [N, 1, h, w]
                    f.create_dataset('fields', data=fields.astype(np.complex128).transpose(0, 1, 3, 2)) # [N, 3, h, w]
                    
                    f.create_dataset('wavelength', data=np.stack(wavelength_list_tmp, axis=0)) # [N, 1]
                    f.create_dataset('grid_step', data=np.stack(grid_step_list_tmp, axis=0)) # [N, 2]
                    f.create_dataset('input_len', data=np.stack(input_len_list_tmp, axis=0)) # [N, 1]
                    
                
                cfg.dump_to_yml(f'{os.path.splitext(file_name)[0]}.yml')

def launch_slot_rHz_raw_data_mmi3x3_generation(grid_step=0.05):
    """Generate raw data of simulation. No postprocessing such as normalization. Only padding to a given size.
    Sweep over wavelength from in the range of (1.53, 1.565, 0.002)
    """
    pol = "Hz"
    device_list = [mmi_3x3_L_random_slots]
    points_per_port = [1024]
    eps_val = 1.44 ** 2
    
    npwl=30# should scale with grid_step
    n_sampled_slots_range=(0.05, 0.1)
    tapper_size=(4.5, 1.3)
    port_len=1.5
    size = (848, 160) # size we saved to
    normalize_flag = False
    note = f'_{grid_step}um'
    
    import os
    
    machine=os.uname()[1]
    
    # we only use 5 wavelengths for training
    wavelengths = np.arange(1.53, 1.571, 0.01).tolist() # 5 points
    tasks = list(enumerate(wavelengths, start=0))
    
    tasks = {
        "eda06": tasks[0], # 5
        "eda03": tasks[1], # 6
        "eda07": tasks[2], # 7
        "eda09": tasks[3], # 8
        "eda10": tasks[4], # 9
    }

    start_time = time.time()
    if True:
        # print(i, wavelength)
        i, wavelength = tasks[machine]
        name = f"slot_mmi3x3_rHz_{i}_fields_epsilon_size_{size[0]}x{size[1]}"
        configs = [
            (pol, device, eps_val, n_points, [wavelength], size, int(10000 + i * 2000), grid_step, npwl, n_sampled_slots_range, tapper_size, port_len, normalize_flag)
            for device, n_points in zip(device_list, points_per_port)
        ]
        generate_slot_mmi_random_raw_data_h5py(configs, name=name, data_dir=f'./raw/{name}_normalize_{int(normalize_flag)}_grid_{grid_step}um/')
        
    # End the timer
    end_time = time.time()
    # Calculate the total execution time
    total_time_hours = (end_time - start_time) / 3600  # Convert seconds to hours

    # Display the execution time
    print(f"Total execution time: {total_time_hours:.2f} hours")

if __name__ == "__main__":
    import sys

    mode = sys.argv[1]

    if mode == "mmi3x3_etched":
        launch_slot_rHz_raw_data_mmi3x3_generation(grid_step=0.05)
    elif mode == "mmi5x5_etched":
        raise ValueError("Not implemented")
        # launch_slot_rHz_raw_data_mmi5x5_generation(grid_step=0.05)
    else:
        print(f"Not supported mode: {mode}")
