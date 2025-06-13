# lkkbox 250602

import os
import netCDF4
import numpy as np
import argparse
import torch
import logging
import warnings
import pytz
import datetime

import torch.cuda.amp as amp

from makani.utils import logging_utils
from makani.utils.YParams import YParams
from makani.utils import comm
from makani.utils.parse_dataset_metada import parse_dataset_metadata
from makani.models import model_registry
from makani.third_party.checktools import checkType
from makani.third_party import nctools as nct
from makani.third_party import timetools as tt


class Predict():
    def __init__(self, args, params):
        self.params = params # spamming parameters
        self._validate_stat_paths()
        self._validate_predict_params()
        self._set_constants()
        self.args = args
        self.initialized = False


    def _validate_predict_params(self):
        predict_arguments = [
            'predict_with_best_ckpt',
            'predict_output_overwrite',
            'predict_output_dir',
            'predict_ic_path',
            'predict_ic_mode',
            'predict_ic_start',
            'predict_ic_stop',
            'predict_ic_list',
            'predict_output_skipExists',
        ]
        for predict_argument in predict_arguments: # check existence
            if predict_argument not in params.params.keys():
                raise ValueError(f'key "{predict_argument}" not found in "{self.args.yaml_config}" - "{self.args.config}"')
        
        # spam the predict parameters to local
        predict_with_best_ckpt = params['predict_with_best_ckpt']
        predict_output_skipExists = params['predict_output_skipExists']
        predict_output_overwrite = params['predict_output_overwrite']
        predict_output_dir = params['predict_output_dir']
        predict_ic_path = params['predict_ic_path']
        predict_ic_mode = params['predict_ic_mode']
        predict_ic_start = params['predict_ic_start']
        predict_ic_stop = params['predict_ic_stop']
        predict_ic_step = params['predict_ic_step']
        predict_ic_list = params['predict_ic_list']

        # check types
        checkType(predict_output_overwrite, bool, "predict_output_overwrite")
        checkType(predict_output_skipExists, bool, "predict_output_skipExists")
        checkType(predict_output_dir, str, "predict_output_dir")
        checkType(predict_ic_path, str, "predict_ic_path")
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

        with netCDF4.Dataset(predict_ic_path, 'r') as h:
            try:
                variable = h['fields']
                self.variable_shape = variable.shape
            except Exception as e:
                print(f'netCD4 error: {e}')
                print(f'Fatal: unable to read the file {predict_ic_path}')
                return

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

        if not predict_output_overwrite:
            for path in predict_output_paths:
                if os.path.exists(path):
                    raise FileExistsError(f'output path already exists. mannualy delete it or set predict_output_overwrite to True. {path}')

        if not os.path.exists(predict_output_dir):
            logging.info(f'creating {predict_output_dir=}')
            os.system(f'mkdir -p {predict_output_dir}')

        if not os.access(predict_output_dir, os.W_OK):
            raise PermissionError(f'denied to write to {predict_output_dir=}')

        # register in the required variables to self
        self.predict_with_best_ckpt = predict_with_best_ckpt
        self.predict_output_skipExists = predict_output_skipExists
        self.predict_output_overwrite = predict_output_overwrite
        self.predict_output_paths = predict_output_paths
        self.predict_ic_path = predict_ic_path
        self.predict_ics = predict_ics


    def _set_constants(self):
        self.dhours = 6
        self.device = "cpu"
        self.LON = np.r_[0:360:0.25]
        self.LAT = np.r_[90:-90.25:-0.25]
        self.dimNames = ["time", "lat", "lon"]
        params.img_crop_shape_x = self.variable_shape[-2] # x and y are reversed..
        params.img_crop_shape_y = self.variable_shape[-1]
        params.N_in_channels = self.variable_shape[-3]
        params.N_out_channels = self.variable_shape[-3]
        params["n_future"] = 0
        params["img_local_offset_x"] = 0
        params["img_local_offset_y"] = 0
        params["img_local_shape_x"] = self.variable_shape[-2]
        params["img_local_shape_y"] = self.variable_shape[-1] 
        if not hasattr(params, "history_normalization_mode"):
            params["history_normalization_mode"] = "none"

        # variable_names
        self.variable_names = [
            nct.ncreadatt(self.predict_ic_path, 'fields', f'f{ivar:02d}')
            for ivar in range(self.variable_shape[1])
        ]
        self.variable_names = [
            name if name else f'var{ivar:02d}' 
            for ivar, name in enumerate(self.variable_names)
        ]

        # derive N_in_channels
        params["N_in_predicted_channels"] = params.N_in_channels
        # sanitization:
        if not hasattr(params, "add_zenith"):
            params["add_zenith"] = False
        # input channels
        # zenith channel is appended to all the samples, so we need to do it here
        if params.add_zenith:
            params.N_in_channels += 1
        if params.n_history >= 1:
            params.N_in_channels = (params.n_history + 1) * params.N_in_channels
            params.N_in_predicted_channels *= params.n_history + 1
        # these are static and the same for all samples in the same time history
        if params.add_grid:
            n_grid_chan = 2
            if (params.gridtype == "sinusoidal") and hasattr(params, "grid_num_frequencies"):
                n_grid_chan *= params.grid_num_frequencies
            params.N_in_channels += n_grid_chan
        if params.add_orography:
            params.N_in_channels += 1
        if params.add_landmask:
            params.N_in_channels += 2

    def _validate_stat_paths(self):
        for path in [
            params.global_means_path,
            params.global_stds_path,
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(path)

        self.global_means_paths = params.global_means_path
        self.global_stds_paths = params.global_stds_path


    def _initialize_model(self):
        self.model = model_registry.get_model(self.params).to(self.device)
        self.preprocessor = self.model.preprocessor
        self.model.eval()
        self.preprocessor.eval()
        logging.info("model is loaded")


    def _initialize_statistics(self):
        self.global_means = np.load(self.global_means_paths)
        self.global_stds = np.load(self.global_stds_paths)
        logging.info("statistics is loaded")


    def _initialize_amp_mode(self):
        if hasattr(params, "amp_mode") and (params.amp_mode != "none"):
            self.amp_enabled = True
            if params.amp_mode == "fp16":
                self.amp_dtype = torch.float16
            elif params.amp_mode == "bf16":
                self.amp_dtype = torch.bfloat16
            else:
                raise ValueError(f"Unknown amp mode {params.amp_mode}")

            if params.log_to_screen:
                self.logger.info(f"Enabling automatic mixed precision in {params.amp_mode}.")
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32


    def _initialize_torch(self):
        torch.set_grad_enabled(False)


    def _initialize_ic_output(self, ic, output_path):
        if self.predict_output_overwrite and os.path.exists(output_path):
            os.remove(output_path)

        nt = self.params.valid_autoreg_steps+1

        # create variables
        output_shape = (nt, len(self.LAT), len(self.LON))
        for variable_name in self.variable_names:
            nct.create(output_path, variable_name, output_shape, self.dimNames)

        # write dimensions
        time = [self._ic_to_time(ic + i, format='float') for i in range(nt)]
        nct.write(output_path, self.dimNames[0], time)
        nct.write(output_path, self.dimNames[1], self.LAT)
        nct.write(output_path, self.dimNames[2], self.LON)


    def _initialize_zenith(self):
        if not self.params.add_zenith:
            return
        # import cupyx as cpx
        from makani.third_party.climt.zenith_angle import cos_zenith_angle
        self.cos_zenith_angle = cos_zenith_angle
        # self.zenith_input_buff = cpx.zeros_pinned(
        #     (self.params.n_history + 1, 1, self.variable_shape[-2], self.variable_shape[-1]),
        #     dtype=np.float32
        # )
        self.zenith_input_buff = np.zeros(
            (self.params.n_history + 1, 1, self.variable_shape[-2], self.variable_shape[-1]),
            dtype=np.float32
        )
        self.zenith_input_buff = torch.from_numpy(self.zenith_input_buff).to(self.device, dtype=torch.float32)

        # determine the year
        def panic():
            raise RuntimeError(f'unable to determine the year by {self.predict_ic_path=}')

        strings = os.path.basename(self.predict_ic_path).split('.')
        if len(strings) != 2:
            panic()
        try:
            self.predict_ic_year = int(strings[0])
        except Exception as e:
            print(e)
            panic()

        if not (1900 <= self.predict_ic_year <= 2100):
            panic()

        # handle the horizontal grids
        self.lon_grid, self.lat_grid = np.meshgrid(self.LON, self.LAT)

    def _ic_to_time(self, ic, format):
        if format not in ['datetime', 'float']:
            raise ValueError(f'unrecognized {format = }') 

        if format == 'datetime':
            time = datetime.datetime(
                    self.predict_ic_year, 1, 1, 0, 0, 0, tzinfo=pytz.utc
            ) + datetime.timedelta(hours=ic*self.dhours) 

        elif format == 'float':
            time = tt.ymd2float(self.predict_ic_year, 1, 1) + ic * self.dhours / 24
        return time


    def _compute_zenith_angle(self, zenith_input, ic):
        # torch.cuda.nvtx.range_push("Predict:_compute_zenith_angle") # nvtx range
        input_time = self._ic_to_time(ic, format='datetime')
        cos_zenith_inp = np.expand_dims(self.cos_zenith_angle([input_time], self.lon_grid, self.lat_grid).astype(np.float32), axis=1)
        cos_zenith_inp = torch.from_numpy(cos_zenith_inp).to(self.device, dtype=torch.float32)
        zenith_input[...] = cos_zenith_inp[...]

        # nvtx range
        # torch.cuda.nvtx.range_pop()


    def _preprocessor_append_zenith_angle(self, ic):
        if not self.params.add_zenith:
            return 
        zenith_input = self.zenith_input_buff
        self._compute_zenith_angle(zenith_input, ic)
        # zenith_input = torch.from_numpy(zenith_input).to(self.device, dtype=torch.float32) 
        self.preprocessor.unpredicted_inp_eval = zenith_input


    def _write_output(self, output_path, data, istep):
        data = np.squeeze(np.array(data))
        slices = [slice(istep, istep+1), *[slice(None)] * 2] 
        for ivar, varName in enumerate(self.variable_names):
            nct.write(output_path, varName, data[ivar, :], slices)


    def _autoregressive_inference(self, data, ic, output_path):
        data = self.preprocessor.flatten_history(data)
        num_steps = self.params.valid_autoreg_steps
        prediction = data
        with amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
            for istep in range(num_steps + 1):
                logging.info(f'    step = {istep}')
                prediction = self.model(prediction)
                if istep == 0:
                    self._initialize_ic_output(ic, output_path)
                self._write_output(output_path, prediction, istep)


    def _predict(self, ic, output_path):
        logging.info(f'    reading input data..')
        with netCDF4.Dataset(self.predict_ic_path, 'r') as h:
            input_data = np.float32(h['fields'][ic, :])
        input_data = (input_data - self.global_means) / self.global_stds # normalize
        input_data = torch.from_numpy(input_data).to(self.device, dtype=torch.float32)
        self._preprocessor_append_zenith_angle(ic)

        logging.info(f'    predicting..')
        torch.cuda.empty_cache()
        with torch.inference_mode():
            with torch.no_grad():
                predictions = self._autoregressive_inference(input_data, ic, output_path)
        

    def run(self):
        for ic, output_path in zip(self.predict_ics, self.predict_output_paths):
            if os.path.exists(output_path) and self.predict_output_skipExists:
                logging.info(f'{ic=}, skip existing {output_path}')
                continue

            if not self.initialized: 
                self._initialize_statistics()
                self._initialize_model()
                self._initialize_amp_mode()
                self._initialize_torch()
                self._initialize_zenith()
                self.initialized = True

            logging.info(f'{ic=}, path={output_path}')
            self._predict(ic, output_path)


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning) # lkkbox ignore future warning
    parser = argparse.ArgumentParser()
#     parser.add_argument("--fin_parallel_size", default=1, type=int, help="Input feature paralellization")
    # parser.add_argument("--fout_parallel_size", default=1, type=int, help="Output feature paralellization")
    # parser.add_argument("--h_parallel_size", default=1, type=int, help="Spatial parallelism dimension in h")
    # parser.add_argument("--w_parallel_size", default=1, type=int, help="Spatial parallelism dimension in w")
    parser.add_argument("--run_num", default="00", type=str)
    parser.add_argument("--yaml_config", default="./config/sfnonet.yaml", type=str)
    parser.add_argument("--config", default="base_73chq", type=str)
    # parser.add_argument("--batch_size", default=-1, type=int, help="Switch for overriding batch size in the configuration file.")
    parser.add_argument("--checkpoint_path", default=None, type=str)
    # parser.add_argument("--enable_synthetic_data", action="store_true")
    # parser.add_argument("--amp_mode", default="none", type=str, choices=["none", "fp16", "bf16"], help="Specify the mixed precision mode which should be used.")
    # parser.add_argument("--jit_mode", default="none", type=str, choices=["none", "script", "inductor"], help="Specify if and how to use torch jit.")
    # parser.add_argument("--cuda_graph_mode", default="none", type=str, choices=["none", "fwdbwd", "step"], help="Specify which parts to capture under cuda graph")
    # parser.add_argument("--enable_benchy", action="store_true")
    # parser.add_argument("--disable_ddp", action="store_true")
    # parser.add_argument("--enable_nhwc", action="store_true")
    # parser.add_argument("--checkpointing_level", default=0, type=int, help="How aggressively checkpointing is used")
    # parser.add_argument("--epsilon_factor", default=0, type=float)
    # parser.add_argument("--split_data_channels", action="store_true")
    # parser.add_argument("--mode", default="score", type=str, choices=["score", "ensemble"], help="Select inference mode")
    # parser.add_argument("--enable_odirect", action="store_true")

    # # checkpoint format
    parser.add_argument("--checkpoint_format", default="legacy", choices=["legacy", "flexible"], type=str, help="Format in which to load checkpoints.")

    # lkkbox 250509 for saving the output path
    parser.add_argument("--inference_output_path", default="./out.nc", type=str, help="path to save the output of inference")
    parser.add_argument("--overwrite_output_path", default=False, type=bool, help="overwrite the output path path")

    # parse
    args = parser.parse_args()

    # parse parameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    # params["epsilon_factor"] = args.epsilon_factor

    # distributed
    # params["fin_parallel_size"] = args.fin_parallel_size
    # params["fout_parallel_size"] = args.fout_parallel_size
    # params["h_parallel_size"] = args.h_parallel_size
    # params["w_parallel_size"] = args.w_parallel_size

    # params["model_parallel_sizes"] = [args.h_parallel_size, args.w_parallel_size, args.fin_parallel_size, args.fout_parallel_size]
    # params["model_parallel_names"] = ["h", "w", "fin", "fout"]

    # checkpoint format
    params["load_checkpoint"] = params["save_checkpoint"] = args.checkpoint_format

    # make sure to reconfigure logger after the pytorch distributed init
    # comm.init(model_parallel_sizes=params["model_parallel_sizes"],
    #           model_parallel_names=params["model_parallel_names"],
    #           verbose=False)
    # world_rank = comm.get_world_rank()

    # # update parameters
    # params["world_size"] = comm.get_world_size()
    # if args.batch_size > 0:
    #     params.batch_size = args.batch_size
    # params["global_batch_size"] = params.batch_size
    # assert params["global_batch_size"] % comm.get_size("data") == 0, f"Error, cannot evenly distribute {params['global_batch_size']} across {comm.get_size('data')} GPU."
    # params["batch_size"] = int(params["global_batch_size"] // comm.get_size("data"))

    # # set device
    # torch.cuda.set_device(comm.get_local_rank())
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    # # Set up directory
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    # if world_rank == 0:
    #     logging.info(f"writing output to {expDir}")
    #     if not os.path.isdir(expDir):
    #         os.makedirs(expDir, exist_ok=True)
    #         os.makedirs(os.path.join(expDir, "deterministic_scores"), exist_ok=True)
    #         os.makedirs(os.path.join(expDir, "deterministic_scores", "wandb"), exist_ok=True)

    # params["experiment_dir"] = os.path.abspath(expDir)

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

    # params["resuming"] = False
    # params["amp_mode"] = args.amp_mode
    # params["jit_mode"] = args.jit_mode
    # params["cuda_graph_mode"] = args.cuda_graph_mode
    # params["enable_odirect"] = args.enable_odirect
    # params["enable_benchy"] = args.enable_benchy
    # params["disable_ddp"] = args.disable_ddp
    # params["enable_nhwc"] = args.enable_nhwc
    # params["checkpointing"] = args.checkpointing_level
    # params["enable_synthetic_data"] = args.enable_synthetic_data
    # params["split_data_channels"] = args.split_data_channels

    # # wandb configuration
    # if params["wandb_name"] is None:
    #     params["wandb_name"] = args.config + "_inference_" + str(args.run_num)
    # if params["wandb_group"] is None:
    #     params["wandb_group"] = "makani" + args.config
    # if not hasattr(params, "wandb_dir") or params["wandb_dir"] is None:
    #     params["wandb_dir"] = os.path.join(expDir, "deterministic_scores")

    logging_utils.config_logger()
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, "out.log"))
    logging_utils.log_versions()
    params.log(logging.getLogger())

    # parse dataset metadata
    if "metadata_json_path" in params:
        params, _ = parse_dataset_metadata(params["metadata_json_path"], params=params)
    else:
        raise RuntimeError(f"Error, please specify a dataset descriptor file in json format")

    predict = Predict(args, params)
    predict.run()
