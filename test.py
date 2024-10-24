# coding=UTF-8
import argparse
import os
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from pyutils.config import configs
from pyutils.general import AverageMeter, logger as lg
from pyutils.torch_train import (
    count_parameters,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader

from core import builder
from core.datasets.mixup import MixupAll
from core.utils import plot_compare, plot_dynamics


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = True,
    masked: bool = False,
    apply_mask_scaler: bool = True,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    with torch.no_grad():
        for i, (wavelength, grid_step, eps, input_mode, target, mask) in enumerate(validation_loader):
            wavelength = wavelength.to(device, non_blocking=True)
            grid_step = grid_step.to(device, non_blocking=True)
            eps = eps.to(device, non_blocking=True)
            input_mode = input_mode.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True) # a mask to concentrate on the useful part of the images
            target = target.to(device, non_blocking=True)
            data = torch.cat([eps, input_mode], dim=2) # bs, n, 2, h, w complex64
            if mixup_fn is not None:
                data, target = mixup_fn(data, target, random_state=i, vflip=False)

            wavelength, grid_step, data, target, mask = [
                x.flatten(0, 1) for x in [wavelength, grid_step, data, target, mask]
            ]
            batch_h = mask.shape[-2]
            batch_w = mask.shape[-1]
            
            with amp.autocast(enabled=False):
                output = model(data, wavelength, grid_step)
                
                if type(output) == tuple:
                    output, aux_output = output
                else:
                    aux_output = None
            
            mask_scaler = None
            
            if masked:
                raise NotImplementedError
                output = output * mask
                target = target * mask

            val_loss = criterion(output, target, mask_scaler)
            mse_meter.update(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\nValidation set: Average loss: {:.4e}\n".format(mse_meter.avg))
    mlflow.log_metrics({"val_loss": mse_meter.avg}, step=epoch)

    if plot and (epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1) or epoch == 1:
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_valid.png")
        plot_compare(
            wavelength[0:3],
            grid_step=grid_step[0:3],
            epsilon=data[0:3, 0],
            pred_fields=output[0:3, -1],
            target_fields=target[0:3, -1],
            filepath=filepath,
            pol="Hz",
            norm=False,
        )

def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = False,
    masked: bool = False,
    apply_mask_scaler: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    with torch.no_grad():
        for i, (wavelength, grid_step, eps, input_mode, target, mask) in enumerate(test_loader):
            wavelength = wavelength.to(device, non_blocking=True)
            grid_step = grid_step.to(device, non_blocking=True)
            eps = eps.to(device, non_blocking=True)
            input_mode = input_mode.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True) # a mask to concentrate on the useful part of the images
            target = target.to(device, non_blocking=True)
            data = torch.cat([eps, input_mode], dim=2) # bs, n, 2, h, w complex64
            if mixup_fn is not None:
                data, target = mixup_fn(data, target, random_state=i + 10000, vflip=False)

            wavelength, grid_step, data, target, mask = [
                x.flatten(0, 1) for x in [wavelength, grid_step, data, target, mask]
            ]
            
            batch_h = mask.shape[-2]
            batch_w = mask.shape[-1]
            
            with amp.autocast(enabled=False):
                output = model(data, wavelength, grid_step)
            # print(output.shape)
                if type(output) == tuple:
                    output, aux_output = output
                else:
                    aux_output = None
            
            
            if masked:
                output = output * mask
                target = target * mask
                
            mask_scaler = None

            val_loss = criterion(output, target, mask_scaler)
            mse_meter.update(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\nTest set: Average loss: {:.4e}\n".format(mse_meter.avg))
    mlflow.log_metrics({"test_loss": mse_meter.avg}, step=epoch)

    if plot and (epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1) or epoch == 1:
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_test.png")
        plot_compare(
            wavelength[0:3],
            grid_step=grid_step[0:3],
            epsilon=data[0:3, 0],
            pred_fields=output[0:3, -1],
            target_fields=target[0:3, -1],
            filepath=filepath,
            pol="Hz",
            norm=False,
        )


def multiport_train_multiport_test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = False,
    test_split: str = "test",
    test_random_state: int = 0,
    apply_mask_scaler: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    mse_vec = []
    with torch.no_grad():
         for i, (wavelength, grid_step, eps, input_mode, target, mask) in enumerate(
            test_loader
        ):
            wavelength = wavelength.to(device, non_blocking=True)
            grid_step = grid_step.to(device, non_blocking=True)
            eps = eps.to(device, non_blocking=True)
            input_mode = input_mode.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            data = torch.cat([eps, input_mode], dim=2)  # bs, n, 2, h, w complex64
            if mixup_fn is not None:
                data, target = mixup_fn(
                    data, target, random_state=i + test_random_state, vflip=False
                )

            wavelength, grid_step, data, target, mask = [
                x.flatten(0, 1) for x in [wavelength, grid_step, data, target, mask]
            ]
            output = model(data, wavelength, grid_step)
            
            if type(output) == tuple:
                output, aux_output = output
            else:
                aux_output = None

            mask_scaler = None

            val_loss = criterion(output, target, mask_scaler)
            mse_meter.update(val_loss.item())
            mse_vec.append(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info(
        "\n(MM) Test set: Average loss: {:.4e} std: {:.4e}\n".format(
            mse_meter.avg, np.std(mse_vec)
        )
    )
    mlflow.log_metrics({"test_loss": mse_meter.avg}, step=epoch)

    if plot and (epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"mm_{test_split}.png")
        plot_compare(
            wavelength[0:3],
            grid_step=grid_step[0:3],
            epsilon=data[0:3, 0],
            pred_fields=output[0:3, -1],
            target_fields=target[0:3, -1],
            filepath=filepath,
            pol="Hz",
            norm=False,
        )
        
def plot_multiport_train_multiport_test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = True,
    test_split: str = "test",
    test_random_state: int = 0,
    apply_mask_scaler: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    mse_vec = []
    with torch.no_grad():
        for i, (wavelength, grid_step, eps, input_mode, target, mask) in enumerate(
            test_loader
        ):
            wavelength = wavelength.to(device, non_blocking=True)
            grid_step = grid_step.to(device, non_blocking=True)
            eps = eps.to(device, non_blocking=True)
            input_mode = input_mode.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            data = torch.cat([eps, input_mode], dim=2)  # bs, n, 2, h, w complex64
            if mixup_fn is not None:
                data, target = mixup_fn(
                    data, target, random_state=i + test_random_state, vflip=False
                )

            wavelength, grid_step, data, target, mask = [
                x.flatten(0, 1) for x in [wavelength, grid_step, data, target, mask]
            ]
            output = model(data, wavelength, grid_step)
            
            if type(output) == tuple:
                output, aux_output = output
            else:
                aux_output = None

            mask_scaler = None

            val_loss = criterion(output, target, mask_scaler)
            mse_meter.update(val_loss.item())
            mse_vec.append(val_loss.item())
            
            if plot and (
                epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1) and i % 100 == 0:
                dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
                os.makedirs(dir_path, exist_ok=True)
                filepath = os.path.join(dir_path, f"mm_{test_split}_{i}.png")
                plot_compare(
                    wavelength[0:3],
                    grid_step=grid_step[0:3],
                    epsilon=data[0:3, 0],
                    pred_fields=output[0:3, -1],
                    target_fields=target[0:3, -1],
                    filepath=filepath,
                    pol="Hz",
                    norm=False,
                )

    loss_vector.append(mse_meter.avg)

    lg.info(
        "\n(MM) Test set: Average loss: {:.4e} std: {:.4e}\n".format(
            mse_meter.avg, np.std(mse_vec)
        )
    )
    mlflow.log_metrics({"test_loss": mse_meter.avg}, step=epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic()

    test_split = getattr(configs.run, "test_split", "test")
    if test_split == "test":
        _, _, test_loader = builder.make_dataloader(splits=["test"])
    elif test_split == "valid":
        _, test_loader, _ = builder.make_dataloader(splits=["valid"])
    elif test_split == "train":
        test_loader, _, _ = builder.make_dataloader(splits=["train"])
    model = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
        eps_min=test_loader.dataset.eps_min.item(),
        eps_max=test_loader.dataset.eps_max.item(),
    )
    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(device)
    aux_criterions = {
        name: [builder.make_criterion(name, cfg=config), float(config.weight)]
        for name, config in configs.aux_criterion.items()
        if float(config.weight) > 0
    }
    print(aux_criterions)

    test_mixup_fn = MixupAll(**configs.dataset.test_augment)

    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)

    # run_id_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=model_name)
    mlflow.log_params(
        {
            "exp_name": configs.run.experiment,
            "exp_id": experiment.experiment_id,
            "run_id": mlflow.active_run().info.run_id,
            "init_lr": configs.optimizer.lr,
            "checkpoint": checkpoint,
            "restore_checkpoint": configs.checkpoint.restore_checkpoint,
            "pid": os.getpid(),
        }
    )

    lossv = [0]
    try:
        lg.info(
            f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )
        lg.info(configs)
        if int(configs.checkpoint.resume) and len(configs.checkpoint.restore_checkpoint) > 0:
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )
            
            if int(configs.model.aux_head):
                model.set_aux_head_mode(True) # set we pass through aux_head
            
            lg.info("Validate resumed model on test dataset...")
            
            test(
                model,
                test_loader,
                0,
                criterion,
                [],
                [],
                device,
                mixup_fn=test_mixup_fn,
                plot=configs.plot.valid,
                apply_mask_scaler=False,
                )

            test_mode = getattr(configs.run, "test_mode", "mm")
            if test_mode == "mm":
                multiport_train_multiport_test(
                    model,
                    test_loader,
                    0,
                    criterion,
                    lossv,
                    device,
                    test_mixup_fn,
                    plot=True,
                    test_split=test_split,
                    test_random_state=configs.run.test_random_state,
                )
            elif test_mode == "plot":
                plot_multiport_train_multiport_test(
                    model,
                    test_loader,
                    0,
                    criterion,
                    lossv,
                    device,
                    test_mixup_fn,
                    plot=True,
                    test_split=test_split,
                    test_random_state=configs.run.test_random_state,
                )
            else:
                raise NotImplementedError(f"Test mode {test_mode} not implemented yet")

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
