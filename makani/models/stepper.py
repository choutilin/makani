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

import torch
import torch.nn as nn
#import numpy as np  # choutilin
from makani.models.preprocessor import Preprocessor2D

class SingleStepWrapper(nn.Module):
    def __init__(self, params, model_handle):
        super(SingleStepWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()
        self.channel_names = params.channel_names

    def forward(self, inp):
        # first append unpredicted features
        inpa = self.preprocessor.append_unpredicted_features(inp)

        # now normalize
        self.preprocessor.history_compute_stats(inpa)
        inpan = self.preprocessor.history_normalize(inpa, target=False)

        # now add static features if requested
        inpans = self.preprocessor.add_static_features(inpan)
"stepper.py" 150L, 5637C                                                                                                     1,1           Top
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
#import numpy as np  # choutilin
from makani.models.preprocessor import Preprocessor2D

        super(SingleStepWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()
    def forward(self, inp):
        # first append unpredicted features
        inpa = self.preprocessor.append_unpredicted_features(inp)

        # now normalize
        self.preprocessor.history_compute_stats(inpa)
        inpan = self.preprocessor.history_normalize(inpa, target=False)

        # now add static features if requested
-rw-r--r--  1 choutilin1 MST112107 9.4K Jun 11  2025 model_package.py
        # choutilin1 250613
        #np.save("/work/choutilin1/inpans.npy",inpans.detach().cpu().numpy())  # remember to import numpy first
        #raise Exception
        # currently the land-sea mask is inpans[0,-1,:,:]
        #       and the inverted mask is inpans[0,-2,:,:]

        for i,channel_name in enumerate(self.channel_names):  # solution by Sherman
            if channel_name in ["sst","ssh","ssu","ssv","D15","D20","MLD"]:
                inpans[0,i,:,:] *= inpans[0,-1,:,:]
            elif channel_name in ["sst-dt"]:
                inpans[0,i,:,:] *= 0  # DO NOT learn sst-dt based on the previous sst-dt

        # forward pass
        yn = self.model(inpans)

        # undo normalization
        y = self.preprocessor.history_denormalize(yn, target=True)
        # mask not just the input, but also the DENORMALIZED output of the model
        for i,channel_name in enumerate(self.channel_names):  # solution by Sherman
            if channel_name in ["ssh","ssu","ssv","D15","D20","MLD","sst-dt"]: # no need to handle the output sst because it's gonna be ignored later on anyway
                y[0,i,:,:] *= inpans[0,-1,:,:]

        # add residual (for residual learning, no-op for direct learning
        y = self.preprocessor.add_residual(inp, y)

        return y


class MultiStepWrapper(nn.Module):
    def __init__(self, params, model_handle):
        super(MultiStepWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()
        self.residual_mode = True if (params.target == "target") else False

        # collect parameters for history
"stepper.py" 150L, 5683C                                                                                                     57,83         33%

-rw-r--r--  1 choutilin1 MST112107 3.0K Jun 11  2025 Readme.md
        #raise Exception
        # currently the land-sea mask is inpans[0,-1,:,:]
        #       and the inverted mask is inpans[0,-2,:,:]

        for i,channel_name in enumerate(self.channel_names):  # solution by Sherman
            if channel_name in ["sst","ssh","ssu","ssv","D15","D20","MLD","ssha"]:
                inpans[0,i,:,:] *= inpans[0,-1,:,:]
            elif channel_name in ["sst-dt"]:
                inpans[0,i,:,:] *= 0  # DO NOT learn sst-dt based on the previous sst-dt

        # forward pass
        yn = self.model(inpans)

        # undo normalization
        y = self.preprocessor.history_denormalize(yn, target=True)
        # mask not just the input, but also the DENORMALIZED output of the model
        for i,channel_name in enumerate(self.channel_names):  # solution by Sherman
            if channel_name in ["ssh","ssu","ssv","D15","D20","MLD","sst-dt","ssha"]: # no need to handle the output sst because it's gonna be ignored later on anyway
                y[0,i,:,:] *= inpans[0,-1,:,:]

        # add residual (for residual learning, no-op for direct learning
        y = self.preprocessor.add_residual(inp, y)

        return y


class MultiStepWrapper(nn.Module):
    def __init__(self, params, model_handle):
        super(MultiStepWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()
        self.residual_mode = True if (params.target == "target") else False

        # collect parameters for history
        self.n_future = params.n_future

"stepper.py" 150L, 5697C                                                                                                     58,83         35%
        #raise Exception
(makani) choutilin1@cbi-lgn01:~/makani/makani/models$ diff stepper.py stepper_2025.py
26d25
<         self.channel_names = params.channel_names
44,50c43,49
<
<         for i,channel_name in enumerate(self.channel_names):
<             if channel_name in ["sst","ssh","ssu","ssv","D15","D20","MLD"]:
<                 inpans[0,i,:,:] *= inpans[0,-1,:,:]
<             elif channel_name in ["sst-dt"]:
<                 inpans[0,i,:,:] *= 0  # DO NOT learn sst-dt based on the previous sst-dt
<
---
>         inpans[0,77,:,:] *= 0
>         inpans[0,78,:,:] *= inpans[0,-1,:,:]
>         inpans[0,79,:,:] *= inpans[0,-1,:,:]
>         inpans[0,80,:,:] *= inpans[0,-1,:,:]
>         inpans[0,81,:,:] *= inpans[0,-1,:,:]
>         inpans[0,82,:,:] *= inpans[0,-1,:,:]
>         inpans[0,83,:,:] *= inpans[0,-1,:,:]
57,60c56,63
<         for i,channel_name in enumerate(self.channel_names):
<             if channel_name in ["ssh","ssu","ssv","D15","D20","MLD","sst-dt"]: # no need to handle the output sst because it's gonna be ignored later on anyway
<                 y[0,i,:,:] *= inpans[0,-1,:,:]
<
---
>         y[0,77,:,:] *= inpans[0,-1,:,:]
>         #y[0,78,:,:] *= inpans[0,-1,:,:]  # f78 is gonna be overwritten by f77 + the previous f78
>         y[0,79,:,:] *= inpans[0,-1,:,:]
>         y[0,80,:,:] *= inpans[0,-1,:,:]
>         y[0,81,:,:] *= inpans[0,-1,:,:]
>         y[0,82,:,:] *= inpans[0,-1,:,:]
>         y[0,83,:,:] *= inpans[0,-1,:,:]
>         #
(makani) choutilin1@cbi-lgn01:~/makani/makani/models$ vim stepper.py
(makani) choutilin1@cbi-lgn01:~/makani/makani/models$ vim stepper.py
(makani) choutilin1@cbi-lgn01:~/makani/makani/models$ vim stepper.py
(makani) choutilin1@cbi-lgn01:~/makani/makani/models$ lsl
total 68K
-rw-r--r--+ 1 choutilin1 MST112107 5.6K Mar  5 14:35 stepper.py
drwxr-xr-x  1 choutilin1 MST112107    0 Jan  5 21:37 __pycache__
-rw-r--r--+ 1 choutilin1 MST112107 5.6K Dec 16 21:42 stepper_2025.py
-rw-r--r--+ 1 choutilin1 MST112107  18K Jun 19  2025 preprocessor.py
-rw-r--r--  1 choutilin1 MST112107 5.5K Jun 11  2025 model_registry.py
drwxr-xr-x  1 choutilin1 MST112107    0 Jun 11  2025 common
-rw-r--r--  1 choutilin1 MST112107 1.6K Jun 11  2025 helpers.py
-rw-r--r--  1 choutilin1 MST112107  814 Jun 11  2025 __init__.py
drwxr-xr-x  1 choutilin1 MST112107    0 Jun 11  2025 networks
-rw-r--r--  1 choutilin1 MST112107 9.4K Jun 11  2025 model_package.py
-rw-r--r--  1 choutilin1 MST112107 3.0K Jun 11  2025 Readme.md
(makani) choutilin1@cbi-lgn01:~/makani/makani/models$
(makani) choutilin1@cbi-lgn01:~/makani/makani/models$ cat stepper.py
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

import torch
import torch.nn as nn
#import numpy as np  # choutilin
from makani.models.preprocessor import Preprocessor2D

class SingleStepWrapper(nn.Module):
    def __init__(self, params, model_handle):
        super(SingleStepWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()
        self.channel_names = params.channel_names

    def forward(self, inp):
        # first append unpredicted features
        inpa = self.preprocessor.append_unpredicted_features(inp)

        # now normalize
        self.preprocessor.history_compute_stats(inpa)
        inpan = self.preprocessor.history_normalize(inpa, target=False)

        # now add static features if requested
        inpans = self.preprocessor.add_static_features(inpan)
        #
        # choutilin1 250613
        #np.save("/work/choutilin1/inpans.npy",inpans.detach().cpu().numpy())  # remember to import numpy first
        #raise Exception
        # currently the land-sea mask is inpans[0,-1,:,:]
        #       and the inverted mask is inpans[0,-2,:,:]

        for i,channel_name in enumerate(self.channel_names):  # solution by Sherman
            if channel_name in ["sst","ssh","ssu","ssv","D15","D20","MLD","ssha"]:
                inpans[0,i,:,:] *= inpans[0,-1,:,:]
            elif channel_name in ["sst-dt"]:
                inpans[0,i,:,:] *= 0  # DO NOT learn sst-dt based on the previous sst-dt

        # forward pass
        yn = self.model(inpans)

        # undo normalization
        y = self.preprocessor.history_denormalize(yn, target=True)
        # mask not just the input, but also the DENORMALIZED output of the model
        for i,channel_name in enumerate(self.channel_names):  # solution by Sherman
            if channel_name in ["sst","ssh","ssu","ssv","D15","D20","MLD","sst-dt","ssha"]: # no need to handle the output sst because it's gonna be ignored later on anyway
                y[0,i,:,:] *= inpans[0,-1,:,:]

        # add residual (for residual learning, no-op for direct learning
        y = self.preprocessor.add_residual(inp, y)

        return y


class MultiStepWrapper(nn.Module):
    def __init__(self, params, model_handle):
        super(MultiStepWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()
        self.residual_mode = True if (params.target == "target") else False

        # collect parameters for history
        self.n_future = params.n_future

    def _forward_train(self, inp):
        result = []
        inpt = inp
        for step in range(self.n_future + 1):
            # add unpredicted features
            inpa = self.preprocessor.append_unpredicted_features(inpt)

            # do history normalization
            self.preprocessor.history_compute_stats(inpa)
            inpan = self.preprocessor.history_normalize(inpa, target=False)

            # add static features
            inpans = self.preprocessor.add_static_features(inpan)

            # prediction
            predn = self.model(inpans)

            # append the denormalized result to output list
            # important to do that here, otherwise normalization stats
            # will have been updated later:
            pred = self.preprocessor.history_denormalize(predn, target=True)

            # add residual (for residual learning, no-op for direct learning
            pred = self.preprocessor.add_residual(inpt, pred)

            # append output
            result.append(pred)

            if step == self.n_future:
                break

            # append history
            inpt = self.preprocessor.append_history(inpt, pred, step)

        # concat the tensors along channel dim to be compatible with flattened target
        result = torch.cat(result, dim=1)

        return result

    def _forward_eval(self, inp):
        # first append unpredicted features
        inpa = self.preprocessor.append_unpredicted_features(inp)

        # do history normalization
        self.preprocessor.history_compute_stats(inpa)
        inpan = self.preprocessor.history_normalize(inpa, target=False)

        # add static features
        inpans = self.preprocessor.add_static_features(inpan)

        # important, remove normalization here,
        # because otherwise normalization stats are already outdated
        yn = self.model(inpans)

        # important, remove normalization here,
        # because otherwise normalization stats are already outdated
        y = self.preprocessor.history_denormalize(yn, target=True)

        # add residual (for residual learning, no-op for direct learning
        y = self.preprocessor.add_residual(inp, y)

        return y

    def forward(self, inp):
        # choutilin 250618
        raise Exception("Let me make sure we're not using MultiStepWrapper")

        # decide which routine to call
        if self.training:
            y = self._forward_train(inp)
        else:
            y = self._forward_eval(inp)

        return y
