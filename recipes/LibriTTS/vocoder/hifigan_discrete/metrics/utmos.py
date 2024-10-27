"""UTMOS.

Authors
 * Jarod Duret 2024
"""

import pathlib as pl
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2
from speechbrain.utils.fetching import fetch

__all__ = ["UTMOS"]


SAMPLE_RATE = 16000
ENCODER_HUB = "chaanks/wav2vec2-small"
MODEL_URL = "https://huggingface.co/chaanks/UTMOS/resolve/main"
MODEL_NAME = "utmos.ckpt"
SAVE_DIR = "pretrained_models"


class _UTMOS(nn.Module):
    def __init__(
        self,
        source,
        save_path,
        features_dim=768,
        num_domains=3,
        domain_dim=128,
        num_judges=3000,
        judge_dim=128,
        decoder_hidden_size=512,
    ):
        super(_UTMOS, self).__init__()

        self.ssl_encoder = Wav2Vec2(
            source,
            save_path,
            freeze=True,
            output_norm=False,
            freeze_feature_extractor=True,
            output_all_hiddens=False,
        )

        self.domain_embedding = nn.Embedding(num_domains, domain_dim)
        self.judge_embedding = nn.Embedding(num_judges, judge_dim)

        self.decoder = nn.LSTM(
            input_size=features_dim + domain_dim + judge_dim,
            hidden_size=decoder_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(decoder_hidden_size * 2, 2048),
            torch.nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1),
        )

    def forward(self, wav, domain_id, judge_id):
        ssl_features = self.ssl_encoder(wav)
        domain_emb = self.domain_embedding(domain_id)
        judge_emb = self.judge_embedding(judge_id)

        domain_emb = domain_emb.unsqueeze(1).expand(
            -1, ssl_features.size(1), -1
        )
        judge_emb = judge_emb.unsqueeze(1).expand(-1, ssl_features.size(1), -1)
        concatenated_feature = torch.cat(
            [ssl_features, domain_emb, judge_emb], dim=2
        )

        decoder_output, _ = self.decoder(concatenated_feature)
        pred = self.classifier(decoder_output)

        return pred.mean(dim=1).squeeze(1) * 2 + 3  # I have no idea why


class UTMOS(MetricStats):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.clear()

        encoder_path = pl.Path(SAVE_DIR) / "encoder"
        self.model = _UTMOS(
            source=ENCODER_HUB, save_path=encoder_path.as_posix()
        )

        # Download utmos model checkpoint
        fetch(MODEL_NAME, MODEL_URL, SAVE_DIR)
        model_path = pl.Path(SAVE_DIR) / MODEL_NAME
        assert model_path.exists()

        # Load weights
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def append(self, ids, hyp_audio, lens=None):
        assert hyp_audio.ndim == 2

        # Resample
        hyp_audio = torchaudio.functional.resample(
            hyp_audio, self.sample_rate, SAMPLE_RATE
        )

        self.model.device = hyp_audio.device
        self.model.to(hyp_audio.device)

        domain_id = torch.zeros(1, dtype=torch.int).to(hyp_audio.device)
        judge_id = torch.ones(1, dtype=torch.int).to(hyp_audio.device) * 288

        output = self.model(hyp_audio, domain_id, judge_id)
        self.scores += output.cpu().tolist()

        self.ids += ids
