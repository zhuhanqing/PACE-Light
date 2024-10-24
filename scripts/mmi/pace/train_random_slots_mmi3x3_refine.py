'''
 # @ Author: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Create Time: 2024-04-20 20:29:13
 # @ Modified by: Hanqing Zhu(hqzhu@utexas.edu)
 # @ Modified time: 2024-10-21 02:38:08
 # @ Description: add the refine module to train_random_slots_mmi3x3_refine.py
 '''

import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

test_flag = True

dataset = "mmi"
model = "pace"
device = "mmi3x3"
exp_name = "mmi3x3_etched"
script = 'refine.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/refine.yml'
configs.load(config_file, recursive=True)

exp_name = f"{exp_name}_refine"
root = f"log/{dataset}/{model}/{exp_name}"

mlflow_name = f"{device}_{model}"


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    mixup, loss, loss_norm, aux_loss, aux_loss_w, aux_loss_norm, lr, enc, n_data, n_layer, id, bs, n_epochs, data_ratio, ckpt, aux_head_ckpt = args
    # data_list = [f"slot_rHz_{i}" for i in range(n_data)]
    
    device_list = ["slot_mmi3x3"]
    data_list = [f"slot_mmi3x3_rHz_{i}_fields_epsilon_normalize_0_grid_0.05um"  for i in range(n_data)]
    grid_step = 0.05
    processed_dir = f"random_size{n_data}_slot_{device}_normalize0_data"
    
    ### dataset args
    resize = True
    resize_mode = 'bilinear'
    resize_style = 'trim'
    normalize = True
    test_ratio = 1 /  (n_data * data_ratio) #
    
    ### optimizr args
    optimizer_name = 'adamw'
    weight_decay = 1e-5
    
    lp_optimizer_name = 'adamw'
    lp_weight_decay = 0 if len(aux_head_ckpt) > 0 else weight_decay # only set the weight decay to 0 once we have pretrained model
    lp_lr =0.001 if len(aux_head_ckpt) > 0 else lr # only set the lr to 0.001 when we have aux head
    
   ### normalization function
    norm_fn = 'bn' # bn, in not work for resize
    drop_path_rate =  0.1 if n_layer < 15 else 0.15# ori: 0.15 
    
    ### pace design
    layer_skip = True
    block_skip = True
    kk =3
    block_type = 'ffno'
    dim=64 # make nerologht go to 3.23M
    mode_list = (40, 70)
    kernel_size_list = [kk]*n_layer # kernel size for depthwise conv
    mode_list = [mode_list]*n_layer
    padding_list = [1]*n_layer
    kernel_list = [dim]*n_layer
    pre_norm_fno = True
    module_type = 'pace_4x'
    pos = []
    hidden_list=[128]
    if n_layer == 12:
        pos = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # 12 layers
    elif n_layer == 16:
        pos = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # 16 layers 
    elif n_layer == 20:
        pos = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] # 20 layers
    else:
        raise ValueError(f"n_layer {n_layer} not supported.")
    
    drop_path_rate =  0.1 if n_layer < 15 else 0.15# ori: 0.15 
    if n_layer == 20:
        drop_path_rate = 0.2
    
   
    # aux pace
    aux_pace = True
    aux_pace_learn_residual = True
    aux_pace_aug_input = True
    aux_pace_aug_feature = True
    aux_pace_aug_feature_enhance = True
    aux_n_layer = 8
    aux_pace_pos=[0, 1, 2, 3, 4, 5, 6, 7]
    aux_head_mode_list = [(40, 100)]*aux_n_layer
    aux_kernel_list=[kernel_list[0]]*aux_n_layer
    aux_padding_list=[1]*aux_n_layer
    aux_kernel_size_list=[kk]*aux_n_layer
    
    
    aux_output_weight = 1
    

    aux_loss_note = f"{aux_loss}_w-{aux_loss_w}" if aux_loss else ""
    skip_note = f"ls{int(layer_skip)}_bs{int(block_skip)}"
    model_note = f"{norm_fn}_pren{int(pre_norm_fno)}"
    resize_note = f"res_d{data_ratio}" if resize else f"raw_d{data_ratio}"
    dataset_note = f"{device}_slot_rHz{n_data}_grid_{grid_step}um_{resize_note}_bs{bs}"
    
    if len(ckpt) > 0:
        model_note = f"{model_note}_YP"
    else:
        model_note = f"{model_note}_NP"

    aux_head_note = f"AUX_len{int(aux_n_layer)}"
    
    if len(aux_head_ckpt) > 0:
        aux_head_note = f"{aux_head_note}_YP_feature_{int(aux_pace_aug_feature)}_enhance_{int(aux_pace_aug_feature_enhance)}"
    else:
        aux_head_note = f"{aux_head_note}_N_feature_{int(aux_pace_aug_feature)}_enhance_{int(aux_pace_aug_feature_enhance)}"
    
    note = f"{block_type}_{dataset_note}_{loss}_{aux_loss_note}_{optimizer_name}_enc-{enc}_nl-{n_layer}_{n_epochs}_id-{id}_{skip_note}_{model_note}_{aux_head_note}_mode-{mode_list[0][0]}x{mode_list[0][1]}"
    
    with open(os.path.join(root, f'{note}.log'), 'w') as wfid:
        exp = [
            f"--checkpoint.resume=True",
            f"--checkpoint.restore_checkpoint={ckpt}",
            f"--checkpoint.aux_head_restore_checkpoint={aux_head_ckpt}",
            f"--checkpoint.checkpoint_dir={dataset}/{model}/{exp_name}/{note}",
            f'--checkpoint.model_comment=mode-{mode_list[0][0]}x{mode_list[0][1]}_refine',
            # dataset args
            f"--dataset.pol_list={str(data_list)}",
            f"--dataset.device_list={str(device_list)}",
            f"--dataset.resize={resize}",
            f"--dataset.normalize={normalize}",
            f"--dataset.resize_mode={str(resize_mode)}",
            f"--dataset.resize_style={str(resize_style)}",
            f"--dataset.data_ratio={data_ratio}",
            f"--dataset.processed_dir={processed_dir}",
            f"--dataset.train_valid_split_ratio=[0.9, 0.1]",
            f"--dataset.test_ratio={test_ratio}",
            f"--dataset.augment.prob={mixup}",
            f"--plot.interval=2",
            f"--plot.dir_name={model}/{exp_name}/{note}/",
            f"--run.log_interval=50",
            f"--run.batch_size={bs}",
            f"--run.n_epochs={n_epochs}",
            f"--run.random_state={41+id}",
            f"--dataset.augment.prob={mixup}",
            # criterion args
            f"--criterion.name={loss}",
            f"--criterion.norm={loss_norm}",
            f"--criterion.aux_output_weight={aux_output_weight}",
            f"--aux_criterion.{aux_loss}.weight={aux_loss_w}",
            f"--aux_criterion.{aux_loss}.norm={aux_loss_norm}",
            # optimizer args
            f"--optimizer.name={optimizer_name}",
            f"--optimizer.weight_decay={weight_decay}",
            f"--optimizer.lr={lr}",
            f"--lp_optimizer.name={lp_optimizer_name}",
            f"--lp_optimizer.weight_decay={lp_weight_decay}",
            f"--lp_optimizer.lr={lp_lr}",
            # model args
            f"--model.pace_config.dim={dim}",
            f"--model.pace_config.kernel_list={kernel_list}",
            f"--model.pace_config.kernel_size_list={kernel_size_list}",
            f"--model.pace_config.padding_list={padding_list}",
            f"--model.pace_config.mode_list={mode_list}",
            f"--model.pace_config.layer_skip={layer_skip}",
            f"--model.pace_config.block_skip={block_skip}",
            f"--model.pace_config.norm_func={norm_fn}",
            f"--model.pace_config.drop_path_rate={drop_path_rate}",
            f"--model.pace_config.aug_path=False",
            f"--model.pace_config.pos={pos}",
            f"--model.pace_config.module_type={module_type}",
            f"--model.pace_config.pre_norm_fno={pre_norm_fno}",
            f"--model.pace_config.hidden_list={hidden_list}",
            f"--checkpoint.checkpoint_dir={dataset}/{model}/{exp_name}/{note}",
            f'--checkpoint.model_comment=mode-{mode_list[0]}x{mode_list[1]}',
            # aux model args
            f"--model.aux_pace={aux_pace}",
            f"--model.aux_pace_learn_residual={aux_pace_learn_residual}",
            f"--model.aux_pace_aug_input={aux_pace_aug_input}",
            f"--model.aux_pace_aug_feature={aux_pace_aug_feature}",
            f"--model.aux_pace_aug_feature_enhance={aux_pace_aug_feature_enhance}",
            f"--model.aux_pace_config.pos={aux_pace_pos}",
            f"--model.aux_pace_config.dim={dim}",
            f"--model.aux_pace_config.kernel_list={aux_kernel_list}",
            f"--model.aux_pace_config.kernel_size_list={aux_kernel_size_list}",
            f"--model.aux_pace_config.padding_list={aux_padding_list}",
            f"--model.aux_pace_config.mode_list={aux_head_mode_list}",
            f"--model.aux_pace_config.pre_norm_fno={pre_norm_fno}",
            
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    machine = os.uname()[1]
    
    # ckpt='./checkpoint/mmi/fftattffno/mmi3x3_etched_new/test_ffno_mmi3x3_slot_rHz5_grid_0.05um_res_d1_bs4_sd0_p0.0_cmse_tv_loss_w-0.005_caf_loss_w-0.1_adamw_enc-exp_nl-16_100_id-6_AT-fftatt15_cfno_4_ct0_m0_ls1_bs1_bn_inv0_s0_en0_pren1_mode-40x70_40x70/FftAttFFNO2d_mode-(40, 70)x(40, 70)_dynamic_resize_err-0.1210_epoch-98.pt'
    # # ckpt='./checkpoint/mmi/fftattffno/mmi3x3_etched_new/test_ffno_mmi3x3_slot_rHz5_grid_0.05um_res_d1_bs4_sd0_p0.0_cmse_tv_loss_w-0.005_adamw_enc-exp_nl-12_100_id-6_AT-fftatt15_cfno_3_ct0_m0_ls1_bs1_bn_inv0_s0_en0_pren1_mode-40x70_40x70/FftAttFFNO2d_mode-(40, 70)x(40, 70)_dynamic_resize_err-0.1425_epoch-99.pt'
    
    # # ckpt='./checkpoint/mmi/fftattffno/mmi3x3_etched_new/test_ffno_mmi3x3_slot_rHz5_grid_0.05um_res_d1_bs4_sd0_p0.0_cmse_tv_loss_w-0.005_adamw_enc-exp_nl-12_100_id-6_AT-fftatt15_cfno_3_ct0_m0_ls1_bs1_bn_inv0_s0_en0_pren1_mode-40x70_40x70/FftAttFFNO2d_mode-(40, 70)x(40, 70)_dynamic_resize_err-0.1425_epoch-99.pt' 
    
    # ckpt="./checkpoint/mmi/fftattffno/mmi3x3_etched_new/test_ffno_mmi3x3_slot_rHz5_grid_0.05um_res_d1_bs4_sd0_p0.0_cmse_tv_loss_w-0.005_adamw_enc-exp_nl-12_100_id-6_AT-fftatt15_cfno_2_ct0_m0_ls1_bs1_bn_inv0_s0_en0_pren1_mode-40x70_40x70/FftAttFFNO2d_mode-(40, 70)x(40, 70)_dynamic_resize_err-0.1340_epoch-100.pt"
    # # aux_head_ckpt="./checkpoint/mmi/RefineAttFFNO/mmi3x3_etched_pretrain_new/pretrain_drop_0.5_fill_nos1_0.1_bw0_s8_ffno_mmi3x3_slot_rHz5_grid_0.05um_res_d1_bs4_sd0_p0.0_cmse_tv_loss_w-0.005_adamw_enc-exp_nl-8_100_id-6_AT-fftatt15_cfno_2_ct0_m0_ls1_bs1_bn_inv0_s0_en0_pren1_mode-40x70_40x70/RefineAttFFNO_mode-(40, 70)x(40, 70).pt"
    # # aux_head_ckpt="./checkpoint/mmi/RefineAttFFNO/mmi3x3_etched_pretrain_new/pretrain_drop_0.5_fill_nos1_0.1_bw0_s8_ffno_mmi3x3_slot_rHz5_grid_0.05um_res_d1_bs4_sd0_p0.0_cmse_tv_loss_w-0.005_adamw_enc-exp_nl-8_100_id-6_AT-fftatt15_cfno_2_ct0_m0_ls1_bs1_bn_inv0_s0_en0_pren1_mode-40x70_40x70/RefineAttFFNO_mode-(40, 70)x(40, 70)_err-0.0010_epoch-87.pt"
    # aux_head_ckpt = "./checkpoint/mmi/RefineAttFFNO/mmi3x3_etched_pretrain_new/pretrain_drop_0_fill_nos1_0.8_bw0_s4_ffno_mmi3x3_slot_rHz5_grid_0.05um_res_d1_bs4_sd0_p0.0_cmse_tv_loss_w-0.005_adamw_enc-exp_nl-8_100_id-6_AT-fftatt15_cfno_2_ct0_m0_ls1_bs1_bn_inv0_s0_en0_pren1_mode-40x70_40x70/RefineAttFFNO_mode-(40, 70)x(40, 70)_err-0.0042_epoch-98.pt"
    # ckpt = "./checkpoint/mmi/fftattffno/mmi3x3_etched_new/new_testffno_mmi3x3_slot_rHz5_grid_0.05um_res_d1_bs4_sd0_p0.0_cmse_tv_loss_w-0.005_adamw_enc-exp_nl-12_100_id-6_AT-fftatt16_cfno_2_len9_ct0_m0_ls1_bs1_bn_inv0_s0_en0_pren1_mode-40x70_40x70/FftAttFFNO2d_mode-(40, 70)x(40, 70)_dynamic_resize_err-0.1059_epoch-100.pt"
    # aux_head_ckpt = "./checkpoint/mmi/RefineAttFFNO/mmi3x3_etched_pretrain_new/0508pretrain_drop_0_fill_nos1_0.5_bw0_s4_ffno_mmi3x3_slot_rHz5_grid_0.05um_res_d1_bs4_sd0_p0.0_cmse_tv_loss_w-0.005_adamw_enc-exp_nl-8_100_id-6_AT-fftatt16_cfno_0_ct0_m0_ls1_bs1_bn_inv0_s0_en0_pren1_mode-40x70_40x70/RefineAttFFNO_mode-[40, 70]x[40, 70]_err-0.0001_epoch-87.pt"
    # ckpt = "./checkpoint/mmi/fftattffno/mmi3x3_etched_new2/new_testffno_mmi3x3_slot_rHz5_grid_0.05um_res_d1_bs4_sd0_p0.0_masked0_cmse_tv_loss_w-0.005_adamw_enc-exp_nl-12_100_id-6_AT-fftatt19_cfno_2_len10_ct0_m0_ls1_bs1_bn_inv0_s0_en0_pren1_diff_mode-40x70_40x70/FftAttFFNO2d_mode-(40, 70)x(40, 70)_dynamic_resize.pt"
    
    # ckpt = "/home/local/eda03/hanqing/project/PACE/checkpoint/mmi/fftattffno/mmi3x3_etched_new2/new2_testffno_mmi3x3_slot_rHz5_grid_0.05um_res_d1_bs4_sd0_p0.0_masked0_cmse_tv_loss_w-0.005_adamw_enc-exp_nl-12_100_id-6_AT-fftatt18_cfno_2_len10_ct0_m0_ls1_bs1_bn_inv0_s0_en0_pren1_diff_mode-40x70_40x70/FftAttFFNO2d_mode-(40, 70)x(40, 70)_dynamic_resize_err-0.1059_epoch-100.pt"
    ckpt = ""
    aux_head_ckpt = ""
    tasks = [
             [1, "cmse", True, "tv_loss", 0.005, True, 0.002, "exp", 5, 12, 6, 4, 100, 1, ckpt, aux_head_ckpt], # baseline implementation on 10 data
             ] # 40, 70, dim=64, droppath=0.1, no augpath

    tasks = {
        "eda03": tasks[0:1],
        "eda-s01": tasks[0:1],
        # "eda14": tasks[3:4],
    }
    print(tasks)
    print(tasks[machine])

    with Pool(1) as p:
        p.map(task_launcher, tasks[machine])
    logger.info(f"Exp: {configs.run.experiment} Done.")
