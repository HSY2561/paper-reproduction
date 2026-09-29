# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Analyzes the FLOPs and parameters of a model, and calculates FPS by default.

Usage:
  # Analyze the model and calculate FPS (default behavior)
  python analyze_model.py --config configs/your_config.yml

  # Analyze the model only, without calculating FPS
  python analyze_model.py --config configs/your_config.yml --no_benchmark

python analyze_model2.py --config configs/aa/unet.yml
"""



import argparse
import os
import sys
import time

import paddle
import numpy as np

# Note: Please ensure paddleseg is installed. If not, run: pip install paddleseg
from paddleseg.cvlibs import Config, manager
from paddleseg.utils import get_sys_env, logger, op_flops_funs
from paddle.hapi.dynamic_flops import (count_parameters, register_hooks,
                                       count_io_info)
from paddle.hapi.static_flops import Table


def parse_args():
    parser = argparse.ArgumentParser(description='Model Analysis Tool')
    parser.add_argument("--config", help="Path to the configuration file.", type=str, required=True)
    parser.add_argument(
        "--input_shape",
        nargs='+',
        type=int,
        help="Set the input shape, e.g., --input_shape 1 3 1024 1024",
        default=[1, 3, 512, 512])
    # --- Add --no_benchmark flag to skip FPS test ---
    parser.add_argument(
        '--no_benchmark',
        action='store_true',
        help='If set, FPS performance benchmark will be skipped.')
    parser.add_argument(
        '--warmup_iter',
        type=int,
        default=10,
        help='Warmup iterations for the performance benchmark.')
    parser.add_argument(
        '--repeats', type=int, default=100, help='Repeat iterations for the performance benchmark.')
    return parser.parse_args()


# This function is a modification of the dynamic_flops function in paddle/hapi/dynamic_flops.py
# to provide a clearer FLOPs output.
def _dynamic_flops(model, inputs, custom_ops=None, print_detail=False):
    handler_collection = []
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return
        m.register_buffer('total_ops', paddle.zeros([1], dtype='int64'))
        m.register_buffer('total_params', paddle.zeros([1], dtype='int64'))
        m_type = type(m)

        flops_fn = None
        if m_type in custom_ops:
            flops_fn = custom_ops[m_type]
            if m_type not in types_collection:
                logger.info("Custom FLOPs function has been applied to {}".format(m_type))
        elif m_type in register_hooks:
            flops_fn = register_hooks[m_type]
            if m_type not in types_collection:
                logger.info("FLOPs for {} have been counted".format(m_type))
        else:
            if m_type not in types_collection:
                logger.warning(
                    "Cannot find a suitable counting function for {}. Its FLOPs will be treated as 0.".format(m_type))

        if flops_fn is not None:
            flops_handler = m.register_forward_post_hook(flops_fn)
            handler_collection.append(flops_handler)
        params_handler = m.register_forward_post_hook(count_parameters)
        io_handler = m.register_forward_post_hook(count_io_info)
        handler_collection.append(params_handler)
        handler_collection.append(io_handler)
        types_collection.add(m_type)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with paddle.framework.no_grad():
        model(inputs)

    total_ops = 0
    total_params = 0
    for m in model.sublayers():
        if len(list(m.children())) > 0:
            continue
        if set(['total_ops', 'total_params', 'input_shape',
                'output_shape']).issubset(set(list(m._buffers.keys()))):
            total_ops += m.total_ops
            total_params += m.total_params

    if training:
        model.train()
    for handler in handler_collection:
        handler.remove()

    table = Table(
        ["Layer Name", "Input Shape", "Output Shape", "Params(M)", "FLOPs(G)"])

    for n, m in model.named_sublayers():
        if len(list(m.children())) > 0:
            continue
        if set(['total_ops', 'total_params', 'input_shape',
                'output_shape']).issubset(set(list(m._buffers.keys()))):
            table.add_row([
                m.full_name(), list(m.input_shape.numpy()),
                list(m.output_shape.numpy()),
                round(float(m.total_params / 1e6), 3),
                round(float(m.total_ops / 1e9), 3)
            ])
            # Clean up buffers
            m._buffers.pop("total_ops")
            m._buffers.pop("total_params")
            m._buffers.pop('input_shape')
            m._buffers.pop('output_shape')

    if print_detail:
        table.print_table()

    logger.info('Total FLOPs: {:.3f}G     Total Params: {:.3f}M'.format(
        round(float(total_ops / 1e9), 3), round(float(total_params / 1e6), 3)))
    return int(total_ops)


def analyze(args):
    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    # It is recommended to use GPU for performance testing to get more realistic data.
    if paddle.get_device().startswith('gpu'):
        paddle.set_device('gpu')
        logger.info("Device has been set to GPU.")
    else:
        paddle.set_device('cpu')
        logger.info("Device has been set to CPU. FPS benchmark results may be lower.")

    cfg = Config(args.config)

    model_cfg = cfg.dic['model'].copy()

    # --- FIX: Manually instantiate the backbone model from its config ---
    if 'backbone' in model_cfg and isinstance(model_cfg['backbone'], dict):
        backbone_cfg = model_cfg.pop('backbone')
        backbone_type = backbone_cfg.pop('type')
        backbone = manager.BACKBONES[backbone_type](**backbone_cfg)
        model_cfg['backbone'] = backbone

    model_type = model_cfg.pop('type')
    model = manager.MODELS[model_type](**model_cfg)

    custom_ops = {paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn}
    inputs = paddle.randn(args.input_shape)

    logger.info("Analyzing model structure, parameters, and FLOPs...")
    _dynamic_flops(model, inputs, custom_ops=custom_ops, print_detail=True)

    # --- Calculate FPS by default, unless --no_benchmark is specified ---
    if not args.no_benchmark:
        model.eval()
        with paddle.no_grad():
            logger.info('=' * 48)
            logger.info('Starting FPS performance benchmark...')

            # Warmup
            logger.info(f'Warming up for {args.warmup_iter} iterations...')
            for i in range(args.warmup_iter):
                model(inputs)

            # Synchronize CUDA computations
            if paddle.get_device().startswith('gpu'):
                paddle.device.cuda.synchronize()

            # Performance test
            logger.info(f'Benchmarking for {args.repeats} iterations...')
            start_time = time.time()
            for i in range(args.repeats):
                model(inputs)

            # Synchronize CUDA computations
            if paddle.get_device().startswith('gpu'):
                paddle.device.cuda.synchronize()

            end_time = time.time()

            total_time = end_time - start_time
            avg_time = total_time / args.repeats
            fps = args.repeats / total_time

            logger.info(f'Total time for {args.repeats} iterations: {total_time:.4f}s')
            logger.info(f'Average inference time: {avg_time * 1000:.4f}ms')
            logger.info(f'FPS: {fps:.2f}')
            logger.info('=' * 48)


if __name__ == '__main__':
    args = parse_args()
    if not args.config:
        # This is handled by argparse's required=True, but kept as a safeguard.
        raise RuntimeError('No configuration file specified. Please use the --config argument.')

    logger.info("Config file: " + args.config)
    logger.info("Input shape: " + str(args.input_shape))
    analyze(args)
