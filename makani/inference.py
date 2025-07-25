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

# utilities
from makani.utils import logging_utils
from makani.utils.YParams import YParams

# distributed computing stuff
from makani.utils import comm

# import trainer
from makani.utils.parse_dataset_metada import parse_dataset_metadata
from makani import Inferencer


if __name__ == "__main__":
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
    parser.add_argument("--inference_target_path", default="./out_targ.nc", type=str, help="path to save the inference target (truth)")
    parser.add_argument("--overwrite_output_path", default=False, type=bool, help="overwrite the output path path")
    parser.add_argument("--inference_ic", default=False, type=bool, help="num of inits to infer")
    parser.add_argument("--inference_num_channels", default=False, type=bool, help="num of channels to infer")

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

    # instantiate trainer / inference / ensemble object
    if args.mode == "score":
        # lkkbox 250509 - modified
        # ---- check the stat paths
        for path in [
            params.global_means_path,
            params.global_stds_path,
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(path)

        # ---- read normalization stats
        global_means = np.load(params.global_means_path)
        global_stds = np.load(params.global_stds_path)

        # ---- check output path
        output_path = args.inference_output_path
        target_path = args.inference_target_path
        overwrite = args.overwrite_output_path

        if not overwrite and os.path.exists(output_path):
            raise FileExistsError(output_path)
        elif overwrite and os.path.exists(output_path):
            logging.info(f'overwriting {output_path = }')
        else:
            logging.info(f'{output_path = }')


        # ---- get the number of channels from inf data path
        pathdir = params['inf_data_path']
        files = os.listdir(pathdir)
        files = [f for f in files if f[-3:] in ['.h5', '.nc']]

        if not files:
            raise FileNotFoundError(pathdir)
        with netCDF4.Dataset(f'{pathdir}/{files[0]}', 'r') as h:
            variable_shape = h['fields'].shape
        
        output_channels = list(range(variable_shape[1]))


        # ---- run inferencer and get predictions
        inferencer = Inferencer(params, world_rank)
        inferencer.inference_single(
            compute_metrics = True,  #choutilin 250715
            ic=args.inference_ic, output_data=True, output_channels=output_channels
        )
        # choutilin1 250617:  I ran into this error. I managed to get it to work by changing the following:
        # FutureWarning: The input object of type 'Tensor' is an array-like implementing one of the corresponding protocols (`__array__`, `__array_interface__` or `__array_struct__`); but not a sequence (or 0-D). In the future, this object will be coerced as if it was first converted using `np.array(obj)`. To retain the old behaviour, you have to either modify the type 'Tensor', or assign to an empty array created with `np.empty(correct_shape, dtype=object)`.
        #predictions = np.array(inferencer.pred_outputs).squeeze(axis=1)
        predictions = torch.stack(inferencer.pred_outputs, dim=0).numpy().squeeze(axis=1)
        targets     = torch.stack(inferencer.targ_outputs, dim=0).numpy().squeeze(axis=1)

        # ---- de-normalization
        predictions = predictions * global_stds + global_means
        targets     = targets     * global_stds + global_means

        # ---- save to the file
        if overwrite and os.path.exists(output_path):
            os.remove(output_path)

        print(f'prediction data shape = {predictions.shape}')

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
                h.createVariable(varName, 'float', dimNames)
                h[varName][:] = predictions[:, iVar, :, :].squeeze()

        print(f'predictions saved to {output_path}')

        # choutilin 250630
        if overwrite and os.path.exists(target_path):
            os.remove(target_path)
        with netCDF4.Dataset(target_path, 'w') as h:
            # dimensions
            for dimName, dimVal in zip(dimNames, dimVals):
                h.createDimension(dimName, len(dimVal))
                h.createVariable(dimName, np.int16, [dimName])
                h[dimName][:] = dimVal
            # variables
            for iVar, varName in enumerate(varNames):
                h.createVariable(varName, 'float', dimNames)
                h[varName][:] = targets[:, iVar, :, :].squeeze()
        
        # # lkkbox 250509 - original
        # inferencer.score_model() 

    else:
        raise ValueError(f"Unknown training mode {args.mode}")
