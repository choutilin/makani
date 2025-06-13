# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import netCDF4
import numpy as np
import argparse
import torch
import logging
import warnings

# utilities
from makani.utils import logging_utils
from makani.utils.YParams import YParams

# distributed computing stuff
from makani.utils import comm

# import trainer
from makani.utils.parse_dataset_metada import parse_dataset_metadata
from makani import Inferencer

# lkkbox
from makani.utils.checktools import checkType


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning) # lkkbox ignore future warning
    parser = argparse.ArgumentParser()
    parser.add_argument("--fin_parallel_size", default=1, type=int, help="Input feature paralellization")
    parser.add_argument("--fout_parallel_size", default=1, type=int, help="Output feature paralellization")
    parser.add_argument("--h_parallel_size", default=1, type=int, help="Spatial parallelism dimension in h")
    parser.add_argument("--w_parallel_size", default=1, type=int, help="Spatial parallelism dimension in w")
    parser.add_argument("--run_num", default="00", type=str)
    parser.add_argument("--yaml_config", default="./config/sfnonet.yaml", type=str)
    parser.add_argument("--config", default="base_73chq", type=str)
    parser.add_argument("--batch_size", default=-1, type=int, help="Switch for overriding batch size in the configuration file.")
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--enable_synthetic_data", action="store_true")
    parser.add_argument("--amp_mode", default="none", type=str, choices=["none", "fp16", "bf16"], help="Specify the mixed precision mode which should be used.")
    parser.add_argument("--jit_mode", default="none", type=str, choices=["none", "script", "inductor"], help="Specify if and how to use torch jit.")
    parser.add_argument("--cuda_graph_mode", default="none", type=str, choices=["none", "fwdbwd", "step"], help="Specify which parts to capture under cuda graph")
    parser.add_argument("--enable_benchy", action="store_true")
    parser.add_argument("--disable_ddp", action="store_true")
    parser.add_argument("--enable_nhwc", action="store_true")
    parser.add_argument("--checkpointing_level", default=0, type=int, help="How aggressively checkpointing is used")
    parser.add_argument("--epsilon_factor", default=0, type=float)
    parser.add_argument("--split_data_channels", action="store_true")
    parser.add_argument("--mode", default="score", type=str, choices=["score", "ensemble"], help="Select inference mode")
    parser.add_argument("--enable_odirect", action="store_true")

    # checkpoint format
    parser.add_argument("--checkpoint_format", default="legacy", choices=["legacy", "flexible"], type=str, help="Format in which to load checkpoints.")

    # lkkbox 250509 for saving the output path
    parser.add_argument("--inference_output_path", default="./out.nc", type=str, help="path to save the output of inference")
    parser.add_argument("--overwrite_output_path", default=False, type=bool, help="overwrite the output path path")

    # parse
    args = parser.parse_args()

    # parse parameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params["epsilon_factor"] = args.epsilon_factor

    # distributed
    params["fin_parallel_size"] = args.fin_parallel_size
    params["fout_parallel_size"] = args.fout_parallel_size
    params["h_parallel_size"] = args.h_parallel_size
    params["w_parallel_size"] = args.w_parallel_size

    params["model_parallel_sizes"] = [args.h_parallel_size, args.w_parallel_size, args.fin_parallel_size, args.fout_parallel_size]
    params["model_parallel_names"] = ["h", "w", "fin", "fout"]

    # checkpoint format
    params["load_checkpoint"] = params["save_checkpoint"] = args.checkpoint_format

    # make sure to reconfigure logger after the pytorch distributed init
    comm.init(model_parallel_sizes=params["model_parallel_sizes"],
              model_parallel_names=params["model_parallel_names"],
              verbose=False)
    world_rank = comm.get_world_rank()

    # update parameters
    params["world_size"] = comm.get_world_size()
    if args.batch_size > 0:
        params.batch_size = args.batch_size
    params["global_batch_size"] = params.batch_size
    assert params["global_batch_size"] % comm.get_size("data") == 0, f"Error, cannot evenly distribute {params['global_batch_size']} across {comm.get_size('data')} GPU."
    params["batch_size"] = int(params["global_batch_size"] // comm.get_size("data"))

    # set device
    torch.cuda.set_device(comm.get_local_rank())
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set up directory
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        logging.info(f"writing output to {expDir}")
        if not os.path.isdir(expDir):
            os.makedirs(expDir, exist_ok=True)
            os.makedirs(os.path.join(expDir, "deterministic_scores"), exist_ok=True)
            os.makedirs(os.path.join(expDir, "deterministic_scores", "wandb"), exist_ok=True)

    params["experiment_dir"] = os.path.abspath(expDir)

    if args.checkpoint_path is None:
        params["checkpoint_path"] = os.path.join(expDir, "training_checkpoints/ckpt_mp{mp_rank}.tar")
        params["best_checkpoint_path"] = os.path.join(expDir, "training_checkpoints/best_ckpt_mp{mp_rank}.tar")
    else:
        params["checkpoint_path"] = os.path.join(args.checkpoint_path, "ckpt_mp{mp_rank}.tar")
        params["best_checkpoint_path"] = os.path.join(args.checkpoint_path, "best_ckpt_mp{mp_rank}.tar")

    # check if all files are there - do not comment out.
    for mp_rank in range(comm.get_size("model")):
        checkpoint_fname = params.checkpoint_path.format(mp_rank=mp_rank)
        if params["load_checkpoint"] == "legacy" or mp_rank < 1:
            assert os.path.isfile(checkpoint_fname)

    params["resuming"] = False
    params["amp_mode"] = args.amp_mode
    params["jit_mode"] = args.jit_mode
    params["cuda_graph_mode"] = args.cuda_graph_mode
    params["enable_odirect"] = args.enable_odirect
    params["enable_benchy"] = args.enable_benchy
    params["disable_ddp"] = args.disable_ddp
    params["enable_nhwc"] = args.enable_nhwc
    params["checkpointing"] = args.checkpointing_level
    params["enable_synthetic_data"] = args.enable_synthetic_data
    params["split_data_channels"] = args.split_data_channels
    params["n_future"] = 0

    # wandb configuration
    if params["wandb_name"] is None:
        params["wandb_name"] = args.config + "_inference_" + str(args.run_num)
    if params["wandb_group"] is None:
        params["wandb_group"] = "makani" + args.config
    if not hasattr(params, "wandb_dir") or params["wandb_dir"] is None:
        params["wandb_dir"] = os.path.join(expDir, "deterministic_scores")

    if world_rank == 0:
        logging_utils.config_logger()
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, "out.log"))
        logging_utils.log_versions()
        params.log(logging.getLogger())

    params["log_to_wandb"] = (world_rank == 0) and params["log_to_wandb"]
    params["log_to_screen"] = (world_rank == 0) and params["log_to_screen"]

    # parse dataset metadata
    if "metadata_json_path" in params:
        params, _ = parse_dataset_metadata(params["metadata_json_path"], params=params)
    else:
        raise RuntimeError(f"Error, please specify a dataset descriptor file in json format")

    # below is modeified from inference.py # lkkbox 250529
    # ---- validate prediction arguments
    predict_arguments = [
        'predict_with_best_ckpt',
        'predict_output_overwrite',
        'predict_output_dir',
        'predict_ic_mode',
        'predict_ic_start',
        'predict_ic_stop',
        'predict_ic_list',
        'predict_skipExists',
    ]
    for predict_argument in predict_arguments: # check existence
        if predict_argument not in params.params.keys():
            raise ValueError(f'key "{predict_argument}" not found in "{args.yaml_config}" - "{args.config}"')
    
    # spam the predict parameters to local
    predict_with_best_ckpt = params['predict_with_best_ckpt']
    predict_skipExists = params['predict_skipExists']
    predict_output_overwrite = params['predict_output_overwrite']
    predict_output_dir = params['predict_output_dir']
    predict_ic_mode = params['predict_ic_mode']
    predict_ic_start = params['predict_ic_start']
    predict_ic_stop = params['predict_ic_stop']
    predict_ic_step = params['predict_ic_step']
    predict_ic_list = params['predict_ic_list']

    # check types
    checkType(predict_output_overwrite, bool, "predict_output_overwrite")
    checkType(predict_skipExists, bool, "predict_skipExists")
    checkType(predict_output_dir, str, "predict_output_dir")
    checkType(predict_ic_mode, str, "predict_ic_mode")
    checkType(predict_ic_start, [None, int], "predict_ic_start")
    checkType(predict_ic_stop, [None, int], "predict_ic_stop")
    checkType(predict_ic_step, [None, int], "predict_ic_step")
    checkType(predict_ic_list, [None, list], "predict_ic_list")
    
    if predict_ic_list is not None:
        for predict_ic in predict_ic_list:
            checkType(predict_ic, int, 'element in predict_ic_list')

    MODE_CONTINUOUS = 'continuous'
    MODE_INCONTINUOUS = 'incontinuous'
    MODE_VALIDS = [MODE_CONTINUOUS, MODE_INCONTINUOUS]
    if predict_ic_mode not in MODE_VALIDS:
        raise ValueError(f'invalid {predict_ic_mode=} (valid={MODE_VALIDS})')

    # set which check point to be used
    if predict_with_best_ckpt: # update the checkpoint path
        params["checkpoint_path"] = params["best_checkpoint_path"] 

    # check inconsistent settings, and generate predict_ics
    if predict_ic_mode == MODE_CONTINUOUS:  # (O) start + count (X) list
        err_prefix = f'predict_ic_mode is {MODE_CONTINUOUS}'
        if predict_ic_start is None:
            raise ValueError(f'{err_prefix}, but predict_ic_start is null')
        if predict_ic_stop is None:
            raise ValueError(f'{err_prefix}, but predict_ic_stop is null')
        if predict_ic_step is None:
            raise ValueError(f'{err_prefix}, but predict_ic_step is null')
        if predict_ic_list is not None:
            raise ValueError(f'{err_prefix}, but predict_ic_list is not null')
        predict_ics = list(range(predict_ic_start, predict_ic_stop, predict_ic_step))

    elif predict_ic_mode == MODE_INCONTINUOUS:  # (X) start + count (O) list
        err_prefix = f'predict_ic_mode is {MODE_INCONTINUOUS}'
        if predict_ic_start is not None:
            raise ValueError(f'{err_prefix}, but predict_ic_start is not null')
        if predict_ic_stop is not None:
            raise ValueError(f'{err_prefix}, but predict_ic_stop is not null')
        if predict_ic_step is not None:
            raise ValueError(f'{err_prefix}, but predict_ic_step is not null')
        if predict_ic_list is None:
            raise ValueError(f'{err_prefix}, but predict_ic_list is null')
        predict_ics = predict_ic_list

    logging.info(f'{predict_ics = }')
    logging.info(f'{len(predict_ics) = }')

    # ---- check output path
    predict_output_paths = [
        f'{predict_output_dir}/ic_{ic:05d}.nc'
        for ic in predict_ics
    ]

    if not predict_output_overwrite and not predict_skipExists:
        for path in predict_output_paths:
            if os.path.exists(path):
                raise FileExistsError(f'output path already exists. mannualy delete it or set predict_output_overwrite to True. {path}')

    if not os.path.exists(predict_output_dir):
        logging.info(f'creating {predict_output_dir=}')
        os.system(f'mkdir -p {predict_output_dir}')

    if not os.access(predict_output_dir, os.W_OK):
        raise PermissionError(f'denied to write to {predict_output_dir=}')

    # ---- run inferencer and get predictions
    inferencer = Inferencer(params, world_rank)

    for ic, output_path in zip(predict_ics, predict_output_paths):
        if os.path.exists(output_path) and predict_skipExists:
            logging.info(f'skip existing {output_path}')
            continue
        inferencer.inference_single(ic=ic, output_data=True, output_channels=list(range(5)))
        predictions = np.array(inferencer.pred_outputs)
        predictions = predictions.squeeze(axis=1)

        # ---- save to file
        if predict_output_overwrite and os.path.exists(output_path):
            os.remove(output_path)

        numVars = predictions.shape[1]
        dimNames = [f'dim{iDim}' for iDim in range(predictions.ndim - 1)] # -1 for ivar
        dimVals = [list(range(predictions.shape[i])) for i in [0, 2, 3]]

        varNames = [f'var{iVar}' for iVar in range(predictions.shape[1])]

        with netCDF4.Dataset(output_path, 'w') as h:
            # dimensions
            for dimName, dimVal in zip(dimNames, dimVals):
                h.createDimension(dimName, len(dimVal))
                h.createVariable(dimName, np.int16, [dimName])
                h[dimName][:] = dimVal

            # variables
            for iVar, varName in enumerate(varNames):
                h.createVariable(varName, 'float', dimNames, compression='zlib', complevel=9, shuffle=True)
                h[varName][:] = predictions[:, iVar, :, :].squeeze()

        logging.info(f'predictions saved to {output_path}')
