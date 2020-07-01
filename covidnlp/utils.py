# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Utils to train DistilBERT
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import json
import logging
import os
import socket

import numpy as np
import torch


logger = logging.getLogger(__name__)


def init_gpu_args(args):
    """
    Handle single and multi-GPU / multi-node.
    """
    if args.n_gpu <= 0:
        args.local_rank = 0
        args.master_port = -1
        args.is_master = True
        args.multi_gpu = False
        return

    assert torch.cuda.is_available()

    logger.info("Initializing GPUs")
    if args.n_gpu > 1:
        assert args.local_rank != -1

        args.world_size = int(os.environ["WORLD_SIZE"])
        args.n_gpu_per_node = int(os.environ["N_GPU_NODE"])
        args.global_rank = int(os.environ["RANK"])

        # number of nodes / node ID
        args.n_nodes = args.world_size // args.n_gpu_per_node
        args.node_id = args.global_rank // args.n_gpu_per_node
        args.multi_gpu = True

        assert args.n_nodes == int(os.environ["N_NODES"])
        assert args.node_id == int(os.environ["NODE_RANK"])

    # local job (single GPU)
    else:
        assert args.local_rank == -1

        args.n_nodes = 1
        args.node_id = 0
        args.local_rank = 0
        args.global_rank = 0
        args.world_size = 1
        args.n_gpu_per_node = 1
        args.multi_gpu = False

    # sanity checks
    assert args.n_nodes >= 1
    assert 0 <= args.node_id < args.n_nodes
    assert 0 <= args.local_rank <= args.global_rank < args.world_size
    assert args.world_size == args.n_nodes * args.n_gpu_per_node

    # define whether this is the master process / if we are in multi-node distributed mode
    args.is_master = args.node_id == 0 and args.local_rank == 0
    args.multi_node = args.n_nodes > 1

    # summary
    PREFIX = f"--- Global rank: {args.global_rank} - "
    logger.info(PREFIX + "Number of nodes: %i" % args.n_nodes)
    logger.info(PREFIX + "Node ID        : %i" % args.node_id)
    logger.info(PREFIX + "Local rank     : %i" % args.local_rank)
    logger.info(PREFIX + "World size     : %i" % args.world_size)
    logger.info(PREFIX + "GPUs per node  : %i" % args.n_gpu_per_node)
    logger.info(PREFIX + "Master         : %s" % str(args.is_master))
    logger.info(PREFIX + "Multi-node     : %s" % str(args.multi_node))
    logger.info(PREFIX + "Multi-GPU      : %s" % str(args.multi_gpu))
    logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(args.local_rank)

    # initialize multi-GPU
    if args.multi_gpu:
        logger.info("Initializing PyTorch distributed")
        torch.distributed.init_process_group(
            init_method="env://", backend="nccl",
        )


def set_seed(args):
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
