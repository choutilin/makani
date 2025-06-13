# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#%%


#%%

import logging
import os

import numpy as np
import torch
# from ai_models.model import Model

from afnonet import  unlog_tp_torch  # noqa
from inference_helper import nan_extend, normalise, load_model

LOG = logging.getLogger(__name__)

# model name
custom_model = "best_ckpt_2_in_2_out_20_epochs"
in_channels = 2
out_channels = 2
#backbone_channels = 26

backbone_channels = 2

# setting
#precip_flag = True
precip_flag = False
precip_channels = 20 #20

# weight_path = '/wk171/yungyun/FCN_test_from_ECMWF/ai-models/'
weight_path = 'model_weight/'
input_data_dir = 'input_data'# add inital time

if precip_flag:
  output_data_dir = 'output_data/20241224/with_precip/'+custom_model
else:
  output_data_dir = 'output_data/20241224/'+custom_model

if not os.path.isdir(output_data_dir):
    os.mkdir(output_data_dir)

# Input
area = [90, 0, -90, 360 - 0.25]
grid = [0.25, 0.25]

# setting
n_lat = 720
n_lon = 1440

device = "cpu"

# load global_means and global_stds to do data preprocessing
#def load_statistics(backbone_channels=26):
def load_statistics(backbone_channels=backbone_channels):
    #path = os.path.join(weight_path, "global_means.npy")
    path = os.path.join(weight_path, "new_global_means_2.npy")
    LOG.info("Loading %s", path)
    global_means = np.load(path)
    global_means = global_means[:, :backbone_channels, ...]
    global_means = global_means.astype(np.float32)

    #path = os.path.join(weight_path, "global_stds.npy")
    path = os.path.join(weight_path, "new_global_stds_2.npy")
    LOG.info("Loading %s", path)
    global_stds = np.load(path)
    global_stds = global_stds[:, :backbone_channels, ...]
    global_stds = global_stds.astype(np.float32)
    return global_means, global_stds

# load initial data and do data preprocessing
all_fields = np.load(os.path.join(input_data_dir, 'inital_condition_20241224.npy')).astype(np.float32)
global_means, global_stds = load_statistics()
# all_fields = np.float32(all_fields)

# all_fields_numpy = all_fields.to_numpy(dtype=np.float32)[np.newaxis, :, :-1, :]
#all_fields_numpy = all_fields[np.newaxis, :, :-1, :]
# get only 21
all_fields_numpy = all_fields[np.newaxis, :backbone_channels, :-1, :]

all_fields_numpy = normalise(all_fields_numpy,global_means,global_stds)

# load model wight (weather and precipitation)
#backbone_ckpt = os.path.join(weight_path, "backbone.ckpt")
backbone_ckpt = os.path.join(weight_path, custom_model+".ckpt")
backbone_model = load_model(backbone_ckpt, precip=False, backbone_channels=backbone_channels)
#backbone_model = load_model(backbone_ckpt, precip=False, backbone_channels=backbone_channels, in_channels=in_channels, out_channels=out_channels)

if precip_flag:
    precip_ckpt = os.path.join(weight_path, "precip.ckpt")
    precip_model = load_model(precip_ckpt, precip=True)

# Run the inference session
input_iter = torch.from_numpy(all_fields_numpy).to(device)

torch.set_grad_enabled(False)

# check inout data
print(f"Input shape before model: {input_iter.shape}")


# run model and save output
# 40*6h =240h
for i in range(40):
    
    output = backbone_model(input_iter)
    if precip_flag:
        precip_output = precip_model(output[:, : precip_channels, ...])
    print('finish '+str(i*6)+'h')
    input_iter = output

    output = nan_extend(normalise(output.cpu().numpy(),global_means, global_stds, reverse=True))
    # output = nan_extend(output.cpu().numpy())
    if precip_flag:
      precip_output = nan_extend(unlog_tp_torch(precip_output.cpu()).numpy())
    
    # Save the results
    np.save(os.path.join(output_data_dir, f'output_weather_{(i+1)*6}h'), output.squeeze())
    if precip_flag:
      np.save(os.path.join(output_data_dir, f'output_precipitation_{(i+1)*6}h'), precip_output.squeeze())
