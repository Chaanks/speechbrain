"""
Script to quantize using K-means clustering over acoustic features.

Authors
 * Duret Jarod 2021
"""

# Adapted from https://github.com/facebookresearch/fairseq
# MIT License
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import json
import pathlib as pl

import joblib
import torch
import numpy as np

import sys
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb

from utils import quantize_features


def setup_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_device(use_cuda):
    use_cuda = use_cuda and torch.cuda.is_available()
    print('\n' + '=' * 30)
    print('USE_CUDA SET TO: {}'.format(use_cuda))
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()))
    print('=' * 30 + '\n')
    return torch.device("cuda" if use_cuda else "cpu")


def audio_files(hparams):
    files_name = []
    for split in hparams["splits"]:
        split = hparams[f"{split}_json"]
        ds = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=split,
            output_keys=["id", "wav"],
        )

        for item in ds:
            files_name.append(item['wav'])

    return files_name


if __name__ == "__main__":
    logger = setup_logger()

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Fetch device
    device = get_device(not hparams["no_cuda"])

    # Features loading/extraction for K-means
    logger.info(f"Extracting acoustic features from {hparams['feats_folder']} ...")

    # K-means model
    logger.info(f"Loading K-means model from {hparams['out_kmeans_model_path']} ...")
    kmeans_model = joblib.load(open(hparams["out_kmeans_model_path"], "rb"))
    kmeans_model.verbose = False

    for split in hparams["splits"]:
        split_path = hparams[f"{split}_json"]

        logger.info(f"Extracting {split} acoustic features ...")
        ds = quantize_features(
            kmeans_model,
            hparams["feats_folder"],
            split_path,
            hparams["embedding_type"],
        )

        with open(split_path, 'w') as f:
            json.dump(ds, f, indent=2)