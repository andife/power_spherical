# SPDX-FileCopyrightText: 2024 Bernd Doser
# SPDX-FileCopyrightText: 2024 Andreas Fehlner
#
# SPDX-License-Identifier: MIT

import os
import random
import sys

import numpy as np
import pytest
import torch
from power_spherical import HypersphericalUniform, PowerSpherical


def test_dynamo_export_normal(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            self.normal = torch.distributions.normal.Normal(0, 1)
            super().__init__()

        def forward(self, x):
            return self.normal.sample(x.shape)

    x = torch.randn(2, 3)
    exported_program = torch.export.export(Model(), args=(x,))
    onnx_program = torch.onnx.dynamo_export(
        exported_program,
        x,
    )
    onnx_program.save(str(tmp_path) + os.sep + "normal.onnx")

def test_dynamo_export_power_spherical_githubexample(tmp_path):
    class PowerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            batch_size = 32
            cloc = torch.randn(batch_size, 3)
            cscale = torch.ones(batch_size)
            self.power_spherical = PowerSpherical(loc=cloc, scale=cscale)

        def forward(self, x):
            return self.power_spherical.rsample()


    exported_program = torch.export.export(PowerModel() , args=(torch.randn(1),))
    
def test_dynamo_export_power_githubexample_onnx_static(tmp_path):
    class PowerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            batch_size = 32
            cloc = torch.randn(batch_size, 3)
            cscale = torch.ones(batch_size)
            self.power_spherical = PowerSpherical(loc=cloc, scale=cscale)

        def forward(self, x):
            return self.power_spherical.rsample()


    exported_program = torch.export.export(PowerModel() , args=(torch.randn(1),))
    x = torch.randn(2, 3)

    #export_options = torch.onnx.ExportOptions(dynamic_shapes=True)

    onnx_program = torch.onnx.dynamo_export(
        exported_program,
        x,
        #export_options=export_options
    )
    onnx_program.save(str(tmp_path) + os.sep + "powerspherical_static.onnx")

def test_dynamo_export_power_githubexample_onnx_dynamic_shapes(tmp_path):
    class PowerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            batch_size = 32
            cloc = torch.randn(batch_size, 3)
            cscale = torch.ones(batch_size)
            self.power_spherical = PowerSpherical(loc=cloc, scale=cscale)

        def forward(self, x):
            return self.power_spherical.rsample()


    exported_program = torch.export.export(PowerModel() , args=(torch.randn(1),))
    x = torch.randn(2, 3)


    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)

    onnx_program = torch.onnx.dynamo_export(
        exported_program,
        x,
        export_options=export_options
    )
    onnx_program.save(str(tmp_path) + os.sep + "powerspherical_dyn.onnx")

#@pytest.mark.xfail(reason="not supported feature of ONNX")
def test_dynamo_export_power_spherical():
    class Model(torch.nn.Module):
        def __init__(self):
            self.spherical = PowerSpherical(
                torch.Tensor([0.0, 1.0]), torch.Tensor([1.0])
            )
            super().__init__()

        def forward(self, x):
            return self.spherical.sample(x.shape)

    x = torch.randn(2, 3)
    exported_program = torch.export.export(Model(), args=(x,))
    _ = torch.onnx.dynamo_export(
        exported_program,
        x,
    )

# # https://github.com/pytorch/pytorch/issues/116336
# #@pytest.mark.xfail(reason="not supported feature of ONNX")
# def test_dynamo_export_power_spherical():
#     class PowerModel(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             batch_size = 32
#             cloc = torch.randn(batch_size, 3)
#             cscale = torch.ones(batch_size)
#             self.power_spherical = PowerSpherical(loc=cloc, scale=cscale)

#         def forward(self, x):
#             return self.power_spherical.rsample()


#     x = torch.randn(1)
#     exported_program = torch.export.export(PowerModel() , args=(x,))

