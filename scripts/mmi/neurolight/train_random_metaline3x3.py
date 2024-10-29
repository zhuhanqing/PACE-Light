import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

test_flag = True

dataset = "mmi"
model = "neurolight"
device = "metaline3x3_small"
exp_name = "metaline3x3"
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)

exp_name = f"{exp_name}"
root = f"log/{dataset}/{model}/{exp_name}"
mlflow_name = f"{device}_{model}"


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    mixup, loss, loss_norm, aux_loss, aux_loss_w, aux_loss_norm, lr, enc, n_data, n_layer, id, bs, n_epochs, data_ratio = args

    device_list = ["metaline3x3_small"]
    data_list = [f"metaline3x3_small_rHz_{i}_fields_epsilon_normalize_0_grid_0.05um"  for i in range(n_data)]
    grid_step = 0.05
    processed_dir = f"random_size{n_data}_{device}_normalize0_data"
    img_height = 128
    img_width = 144
    

    ### dataset args
    resize = True
    resize_mode = 'bilinear'
    resize_style = 'trim'
    normalize = True
    test_ratio = 1 /  (n_data * data_ratio) #

    ### optimizr args
    optimizer_name = 'adamw'
    weight_decay = 1e-5

    ### normalization function
    norm_fn = 'bn' # bn, in not work for resize
    drop_path_rate =  0.1 if n_layer < 15 else 0.15# ori: 0.15 

    # neurolight does not have block_skip
    layer_skip = True
    block_skip = False

    ### ffno design
    kk =3
    block_type = 'ffno'
    dim=64 # make nerologht go to 3.23M
    mode_list = (36, 36)
    kernel_size_list = [kk]*n_layer # kernel size for depthwise conv
    mode_list = [mode_list]*n_layer
    padding_list = [1]*n_layer
    kernel_list = [dim]*n_layer
    
    # new_note = f"ls-{int(layer_skip)}_bs-{block_skip}_af-{int(att_act_fn_pos)}_gc-{global_conv_type}_{attention_act_func}_{drop_path_rate}"
    resize_note = f"res_d{data_ratio}" if resize else f"raw_d{data_ratio}"
        
    dataset_note = f"{device}_slot_rHz{n_data}_grid_{grid_step}um_{resize_note}_bs{bs}"
    
    note = f"test_{block_type}_{dataset_note}_{loss}_{optimizer_name}_enc-{enc}_nl-{n_layer}_{n_epochs}_id-{id}_x{mode_list[0][1]}"

    
   
    with open(os.path.join(root, f'{note}.log'), 'w') as wfid:
        exp = [
            # dataset args
            f"--dataset.pol_list={str(data_list)}",
            f"--dataset.device_list={str(device_list)}",
            f"--dataset.resize={resize}",
            f"--dataset.normalize={normalize}",
            f"--dataset.resize_mode={str(resize_mode)}",
            f"--dataset.resize_style={str(resize_style)}",
            f"--dataset.data_ratio={data_ratio}",
            f"--dataset.img_height={img_height}",
            f"--dataset.img_width={img_width}",
            f"--dataset.processed_dir={processed_dir}",
            f"--dataset.train_valid_split_ratio=[0.9, 0.1]",
            f"--dataset.test_ratio={test_ratio}",
            f"--dataset.augment.prob={mixup}",
            f"--plot.dir_name={model}/{exp_name}/{note}/",
            f"--plot.interval=2",
            # optimizer args
            f"--optimizer.lr={lr}",
            f"--optimizer.name={optimizer_name}",
            f"--optimizer.weight_decay={weight_decay}",
            # run args
            f"--run.log_interval=50",
            f"--run.batch_size={bs}",
            f"--run.n_epochs={n_epochs}",
            f"--run.random_state={41+id}",
            f"--run.experiment={mlflow_name}",
            # criterion args
            f"--criterion.name={loss}",
            f"--criterion.norm={loss_norm}",
            f"--criterion.apply_scaling=False",
            f"--criterion.apply_mask_scaler=False",
            f"--aux_criterion.{aux_loss}.weight={aux_loss_w}",
            f"--aux_criterion.{aux_loss}.norm={aux_loss_norm}",
            f"--model.pos_encoding={enc}",
            f"--dataset.train_valid_split_ratio=[0.9, 0.1]",
            f"--model.dim={dim}",
            f"--model.kernel_list={kernel_list}",
            f"--model.kernel_size_list={kernel_size_list}",
            f"--model.padding_list={padding_list}",
            f"--model.mode_list={mode_list}",
            f"--model.layer_skip={layer_skip}",
            f"--model.block_skip={block_skip}",
            f"--model.with_cp=True",
            f"--model.norm_func={norm_fn}",
            f"--model.drop_path_rate={drop_path_rate}",
            f"--checkpoint.checkpoint_dir={dataset}/{model}/{exp_name}/{note}",
            f'--checkpoint.model_comment=mode-{mode_list[0]}x{mode_list[1]}_dynamic_resize',
            ]
        
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    machine = os.uname()[1]

    tasks = [
             [1, "cmse", True, "tv_loss", 0.005, True, 0.002, "exp", 5, 16, 6, 4, 100, 1],
             ]

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
