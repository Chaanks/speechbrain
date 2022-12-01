import json
import itertools
import random
import pathlib as pl

import tqdm
import kaldiio
import numpy as np
import pandas as pd
import speechbrain as sb


def fmt(split, embedding):
    base = None
    if embedding == "XLSR":
        base = f"{split}/framewise_xlsr_1_0_1"
    elif embedding == "SAMU":
        base = f"{split}/framewise_samu_xlsr_1_0_1"
    elif embedding in ["Hubert", "mHubert", "speechT5"]:
        base = f"{split}/feats_{split}"
    return base

def fetch_data(splits, sample_pct, seed=1234):
    splits = [split for split in splits.split('\t')]
    ds_splits = {}
    for split in splits:
        split = pl.Path(split)
        ds_splits[split.stem] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=split,
            output_keys=["id", "wav"],
        )
        
    data = list(itertools.chain(*ds_splits.values()))
    random.seed(seed)
    if sample_pct < 1.0:
        data = random.sample(
            data, int(sample_pct * len(data))
        )
    
    return iter(data), len(data)


def extract_features(feats_folder, splits, sample_pct, flatten, embedding_type, device="cpu"):
    data, num_files = fetch_data(splits, sample_pct)
    features_list = []
    id_list = []

    feats_dict = {}
    for split in [pl.Path(split).stem for split in splits.split('\t')]:
        base = fmt(split, embedding_type)
        feats_scp =f"{feats_folder}/{base}.scp"
        feats_df = pd.read_csv(feats_scp, delimiter=' ', header=None)
        feats_0_fmt = feats_df[0]
        if embedding_type in ["XLSR", "SAMU"]:
            feats_0_fmt = [f.split('-')[1] for f in feats_df[0]]

        feats_dict.update({k:v for k, v in zip(feats_0_fmt, feats_df[1])})

    for item in tqdm.tqdm(data, total=num_files):
        if not item['id'] in feats_dict:
            print(f"CVSS filter: {item['id']}")
            continue
        feats = kaldiio.load_mat(feats_dict[item['id']])
        features_list.append(feats)
        id_list.append(item['id'])

    if flatten:
        return np.concatenate(features_list), id_list

    return features_list, id_list


def quantize_features(model, feats_folder,split_path, embedding_type, device="cpu"):
    split = pl.Path(split_path).stem
    base = base = fmt(split, embedding_type)
    feats_scp =f"{feats_folder}/{base}.scp"
    feats_df = pd.read_csv(feats_scp, delimiter=' ', header=None)
    
    feats_0_fmt = feats_df[0]
    if embedding_type in ["XLSR", "SAMU"]:
        feats_0_fmt = [f.split('-')[1] for f in feats_df[0]]

    feats_dict = {k:v for k, v in zip(feats_0_fmt, feats_df[1])}

    unique_count = 0
    ds = json.load(open(split_path, 'r'))
    for key, row in tqdm.tqdm(ds.items()):
        if not key in feats_dict:
            print("CVSS filter")
            continue
        feats = kaldiio.load_mat(feats_dict[key])
        pred = model.predict(feats)
        row['features'] = pred.tolist()
        unique_count += len(set(pred))

    print(
        f"Unit average per utterance = {unique_count / len(ds)}"
    )

    return ds
