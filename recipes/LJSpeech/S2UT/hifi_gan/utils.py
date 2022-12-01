import json

import pandas as pd
import kaldiio

class EmbeddingManager:
    def __init__(self, root_dir, lang, splits):
        self.embeddings = self._load_embedding(root_dir, lang, splits)
    
    def get_embedding_by_clip(self, clip_idx: str):
        feats_path = self.embeddings[clip_idx]
        return kaldiio.load_mat(feats_path)

    def _load_embedding(self, root_dir, lang, splits):
        utt2embs = {}
        for split in splits:
            scp_path = f"{root_dir}/{lang}_{split}.scp"
            df = pd.read_csv(scp_path, delimiter=" ", header=None)
            utt2embs.update({k:v for k, v in zip(df[0], df[1])})
        return utt2embs