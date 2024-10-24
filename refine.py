'''
 # @ Author: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Create Time: 2024-03-25 16:58:04
 # @ Modified by: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Modified time: 2024-10-21 02:16:28
 # @ Description: main train logic
 '''

#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Callable, Dict, Iterable

import mlflow
import torch
import torch.fft
import torch.cuda.amp as amp

import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import AverageMeter, logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler

from core import builder
from core.datasets.mixup import MixupAll
from core.utils import plot_compare, save_model


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    aux_criterions: Dict,
    mixup_fn: Callable = None,
    plot: bool = False,
    device: torch.device = torch.device("cuda:0"),
    grad_scaler: amp.GradScaler = None,
    disable_aux_output_loss: bool = False,
) -> None:
    model.train()
    step = epoch * len(train_loader)

    mse_meter = AverageMeter("mse")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}
    aux_output_weight = getattr(configs.criterion, "aux_output_weight", 0)
    accum_iter = getattr(configs.run, "grad_accum_step", 1)

    # poynting_loss = PoyntingLoss(configs.model.grid_step, wavelength=1.55)
    data_counter = 0
    mask = None
    total_data = len(train_loader.dataset)
    for batch_idx, (wavelength, grid_step, eps, input_mode, target, mask) in enumerate(train_loader):
        wavelength = wavelength.to(device, non_blocking=True)
        grid_step = grid_step.to(device, non_blocking=True)
        eps = eps.to(device, non_blocking=True)
        input_mode = input_mode.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True) # a mask to concentrate on the useful part of the images
        data_counter += eps.shape[0]
        target = target.to(device, non_blocking=True)
        data = torch.cat([eps, input_mode], dim=2) # bs, n, 2, h, w complex64

        if mixup_fn is not None:
            data, target = mixup_fn(data, target)
        

        wavelength, grid_step, data, target, mask = [x.flatten(0, 1) for x in [wavelength, grid_step, data, target, mask]]

        with amp.autocast(enabled=grad_scaler._enabled):
            output = model(data, wavelength, grid_step)

            if type(output) == tuple:
                output, aux_output = output
            else:
                aux_output = None
            mask_scaler = None
            
            regression_loss = criterion(output, target, mask_scaler)
            mse_meter.update(regression_loss.item())
            loss = regression_loss
            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                if name == "tv_loss":
                    aux_loss = weight * aux_criterion(output, target)

                loss = loss + aux_loss
                aux_meters[name].update(aux_loss.item())

            if aux_output is not None and aux_output_weight > 0 and not disable_aux_output_loss:
                aux_output_loss = aux_output_weight * criterion(aux_output, target, mask_scaler)
                loss = loss + aux_output_loss
            else:
                aux_output_loss = None
            
            loss = loss / accum_iter
        
        grad_scaler.scale(loss).backward()

        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
            grad_scaler.unscale_(optimizer)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            log = "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4e} Regression Loss: {:.4e}".format(
                epoch,
                data_counter,
                total_data,
                100.0 * data_counter / total_data,
                loss.data.item(),
                regression_loss.data.item(),
            )
            for name, aux_meter in aux_meters.items():
                log += f" {name}: {aux_meter.val:.4e}"
            if aux_output_loss is not None:
                log += f" aux_output_loss: {aux_output_loss.item()}"
            lg.info(log)

            mlflow.log_metrics({"train_loss": loss.item()}, step=step)
        
    scheduler.step()
    avg_regression_loss = mse_meter.avg
    lg.info(f"Train Regression Loss: {avg_regression_loss:.4e}")
    mlflow.log_metrics(
        {"train_regression": avg_regression_loss, "lr": get_learning_rate(optimizer)}, step=epoch
    )
    if plot and (epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1) or epoch == 1:
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        print(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_train.png")
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
            
            with amp.autocast(enabled=False):
                output = model(data, wavelength, grid_step)
                
                if type(output) == tuple:
                    output, aux_output = output
                else:
                    aux_output = None
                
            
            mask_scaler = None

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
                
            if type(output) == tuple:
                output, aux_output = output
            else:
                aux_output = None
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
        set_torch_deterministic(int(configs.run.random_state))

    train_loader, validation_loader, test_loader = builder.make_dataloader()
    #### load the model with a refiner module, aux_pace (set feature passing to False)
    model = builder.make_model(
        device,
        model_cfg=configs.model,
        random_state=int(configs.run.random_state) if int(configs.run.deterministic) else None,
        eps_min=train_loader.dataset.eps_min.item(),
        eps_max=train_loader.dataset.eps_max.item(),
    )
    lg.info("Checking model architecture")
    lg.info(model)

    # optimizer for all params
    ft_optimizer = builder.make_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )
    ft_scheduler = builder.make_scheduler(ft_optimizer)
    
    # build optimizer for aux head and aug feature
    lp_optimizer = builder.make_optimizer(
        [p for p in model.aux_pace_model.parameters() if p.requires_grad] + ([p for p in model.aug_feature.parameters() if p.requires_grad] if model.aug_feature is not None else []),
        name=configs.lp_optimizer.name,
        configs=configs.lp_optimizer,
    )
    lp_scheduler = builder.make_scheduler(lp_optimizer)
    
    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(device)
    aux_criterions = {
        name: [builder.make_criterion(name, cfg=config), float(config.weight)]
        for name, config in configs.aux_criterion.items()
        if float(config.weight) > 0
    }
    
    lg.info("Checking training aux loss")
    lg.info(aux_criterions)
    
    mixup_config = configs.dataset.augment
    mixup_fn = MixupAll(**mixup_config)
    test_mixup_fn = MixupAll(**configs.dataset.test_augment)
    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=False,
        truncate=4,
        metric_name="err",
        format="{:.4f}",
    )
    
    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = (
        f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"
    )

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

    lossv, accv = [0], [0]
    epoch = 0
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
            lg.info("Validate resumed model without head...")
            model.set_aux_pace_mode(False) # test under the normal mode
            
            lg.info("Validate resumed model on validation...")
            test(model, 
                validation_loader, 
                0, 
                criterion, 
                [], 
                [], 
                device,
                mixup_fn=test_mixup_fn,
                plot=configs.plot.valid,
            )
            lg.info("Validate resumed model on test...")
            test(
                model,
                test_loader,
                0,
                criterion,
                [],
                [],
                device,
                mixup_fn=test_mixup_fn,
                plot=configs.plot.valid
                )
            
            # load the aux_pace
            if configs.model.aux_pace_model and len(configs.checkpoint.aux_pace_restore_checkpoint) > 0:
                load_model(
                    model.aux_pace_model,
                    configs.checkpoint.aux_pace_restore_checkpoint,
                    ignore_size_mismatch=int(configs.checkpoint.no_linear),
                )
                lg.info("Validate resumed model with pretrained head...")
                model.set_aux_pace_mode(True) # test under the normal mode
                
                lg.info("Validate resumed model on validation...")
                test(model, 
                    validation_loader, 
                    0, 
                    criterion, 
                    [], 
                    [], 
                    device,
                    mixup_fn=test_mixup_fn,
                    plot=configs.plot.valid,
                )
                lg.info("Validate resumed model on test...")
                test(
                    model,
                    test_loader,
                    0,
                    criterion,
                    [],
                    [],
                    device,
                    mixup_fn=test_mixup_fn,
                    plot=configs.plot.test,
                    )

        model.reset_aux_head() # reset the head of aux_head
        model.set_aux_head_mode(True) # set we pass through aux_head
        model.set_aux_head_linear_probing_mode(False) # linear probing mode for aux_head (only train head)

        for epoch in range(1, int(configs.run.n_epochs) + 1):
            # train with aux head
            if configs.criterion.aux_output_weight > 0:
                if epoch <= 100 and len(configs.checkpoint.restore_checkpoint) > 0:
                    lg.info("Continue training for the 2nd-stage model...")
                    #NOTE(hanqing): continuing training with aux head
                    model.set_aux_head_linear_probing_mode(False) # linear probing mode for aux_pace (only train head if we load prettrained model)
                    # only free aux_pace params
                    model.disable_all_params()
                    model.enable_aux_head_params()
                    optimizer = lp_optimizer
                    scheduler = lp_scheduler
                    disable_aux_output_loss = True
                else:
                    #NOTE(hanqing): train two-stage model jointly with aux head
                    lg.info("Training with 2-stage loss...")
                    # fine-tuning with 2-stage loss
                    model.set_aux_head_linear_probing_mode(False) # linear probing mode for aux_pace (only train head)
                    # free all params
                    model.enable_all_params()
                    model.enable_aux_head_params()
                    optimizer = ft_optimizer
                    scheduler = ft_scheduler
                    disable_aux_output_loss = False
            else:
                raise NotImplementedError("Not implemented for the current version")
            train(
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                criterion,
                aux_criterions,
                mixup_fn,
                device=device,
                plot=configs.plot.train,
                grad_scaler=grad_scaler,
                disable_aux_output_loss=disable_aux_output_loss,
            )

            if validation_loader is not None:
                validate(
                    model,
                    validation_loader,
                    epoch,
                    criterion,
                    lossv,
                    accv,
                    device,
                    mixup_fn=test_mixup_fn,
                    plot=configs.plot.valid,
                )
            if epoch > (configs.run.n_epochs - 30):
                test(
                    model,
                    test_loader,
                    epoch,
                    criterion,
                    [],
                    [],
                    device,
                    mixup_fn=test_mixup_fn,
                    plot=configs.plot.test,
                )
                saver.save_model(
                    model, lossv[-1], epoch=epoch, path=checkpoint, save_model=False, print_msg=True
                )
        # save the model for last checkpoint
        save_model({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
            'scaler': grad_scaler.state_dict(),
            "args": args,
        }, checkpoint)
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
