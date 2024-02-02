# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Any, Optional, Union
import mmengine
import torch
from mmengine.config import Config, ConfigDict
from mmengine.registry import FUNCTIONS, MODELS, VISUALIZERS, DefaultScope
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from mmagic.utils import ConfigType, SampleList, register_all_modules, tensor2img

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['output'],
    input_shape=None,
    optimize=True)

register_all_modules()


def main():
    from mmdeploy.apis.core.pipeline_manager import no_mp
    from mmdeploy.utils import (Backend, get_backend, get_dynamic_axes,
                                get_input_shape, get_onnx_config, load_config)
    from mmdeploy.apis.onnx import export
    iters = 250000
    device = 'cuda'
    cfg_file = '/home/pop/mmlab/mmagic/mmagic/models/editors/av4/cfg_nv1_v22_60.py'
    model_file = f'/home/pop/mmlab/mmagic/work_dirs/nv1_v22_60/iter_{iters}.pth'
    cfg = Config.fromfile(cfg_file)
    model = MODELS.build(cfg.model)
    model.cfg = cfg
    checkpoint = _load_checkpoint(model_file, map_location='cpu')
    _load_checkpoint_to_model(model, checkpoint)
    torch_model = model.generator
    torch_model.to(device)
    torch_model.eval()

    input_names = ['imgs', 'fflo', 'bflo']
    output_names = ['output']
    opset_version = 18
    keep_initializers_as_inputs = False

    backend = 'tensorrt'
    optimize = onnx_config['optimize']
    verbose = False
    context_info = dict()
    context_info['deploy_cfg'] = onnx_config
    dynamic_axes = None
    input_metas = None
    # frames, h, w = 45, 1920, 1080
    # frames, h, w = 45, 1080, 1920
    # frames, h, w = 80, 720, 1280
    frames, h, w = 80, 1280, 720
    output_prefix = f'/home/pop/mmlab/mmdeploy/work_dir/nv1_v22_60_{iters}_{frames}x{h}x{w}'

    input_imgs = torch.zeros((1, frames, 3, h, w), dtype=torch.float).to(device)
    input_fflo = torch.zeros((1, frames - 1, 2, h // 4, w // 4), dtype=torch.float).to(device)
    input_bflo = torch.zeros((1, frames - 1, 2, h // 4, w // 4), dtype=torch.float).to(device)
    model_inputs = (input_imgs, input_fflo, input_bflo)

    with no_mp():
        export(
            torch_model,
            model_inputs,
            input_metas=input_metas,
            output_path_prefix=output_prefix,
            backend=backend,
            input_names=input_names,
            output_names=output_names,
            context_info=context_info,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            optimize=optimize)


if __name__ == '__main__':
    main()
