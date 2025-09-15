#!/usr/bin/env python3
"""
Voxtral ASR training script for LibriSpeech.

Authors: Jarod Duret 2025
"""

import os
import sys
import torch
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class VoxtralASR(sb.Brain):
    
    def compute_forward(self, batch, stage):
        """Forward pass through Voxtral model."""
        batch = batch.to(self.device)

        raw_audio, _ = batch.sig
        prompt_tokens, _ = batch.prompt_tokens
        full_input_ids, _ = batch.full_input_ids
        full_attention_mask, _ = batch.full_attention_mask
        full_labels, _ = batch.full_labels
        
        # Extract mel features
        input_features = self.modules.voxtral.processor.extract_mel_features(raw_audio.data)
        input_features = input_features.to(dtype=torch.bfloat16, device=self.device)
        
        # Forward pass
        outputs = self.modules.voxtral(
            input_ids=full_input_ids,
            input_features=input_features,
            attention_mask=full_attention_mask,
            labels=full_labels,
            use_cache=False
        )
        
        logits = outputs.logits
        log_probs = self.hparams.log_softmax(logits)

        if stage != sb.Stage.TRAIN:
            config = self.hparams.generation_config
            hyps = self.modules.voxtral.generate(
                input_ids=prompt_tokens,
                input_features=input_features,
                attention_mask=batch.prompt_attention_mask[0],
                generation_config=config,
            )

        else:
            hyps = None
            
        return log_probs, hyps
    
    def compute_objectives(self, predictions, batch, stage):
        """Compute loss and metrics."""
        log_probs, hyps = predictions
        batch = batch.to(self.device)
        
        # Get targets
        full_labels, full_labels_lens = batch.full_labels
        
        # Compute loss
        loss = self.hparams.nll_loss(log_probs, full_labels, length=full_labels_lens)
        
        # Compute metrics for validation/test
        if stage != sb.Stage.TRAIN:
            ids = batch.id
            tokens, tokens_lens = batch.tokens
            
            # Decode predictions
            predicted_words = [
                self.tokenizer.decode(h, skip_special_tokens=True).strip()
                for h in hyps
            ]
            
            # Remove language prefix if present
            predicted_words = [
                pred_text[7:].strip() if pred_text.startswith("lang:") else pred_text
                for pred_text in predicted_words
            ]
            
            # Decode targets
            target_words = undo_padding(tokens, tokens_lens)
            target_words = [
                self.tokenizer.decode(t, skip_special_tokens=True).strip()
                for t in target_words
            ]
            
            # Apply normalization if specified in hparams
            if self.hparams.normalized_transcripts:
                predicted_words = [
                    normalize_text(text).split() for text in predicted_words
                ]

            else:
                predicted_words = [text.split() for text in predicted_words]

            target_words = [text.split() for text in target_words]


            # LOG: Print first few examples for debugging
            if stage == sb.Stage.VALID:
                logger.info("=== VALIDATION EXAMPLES ===")
                for i in range(min(3, len(ids))):  # Show first 3 examples
                    pred_text = " ".join(predicted_words[i])
                    target_text = " ".join(target_words[i])
                    logger.info(f"ID: {ids[i]}")
                    logger.info(f"TARGET:  {target_text}")
                    logger.info(f"PREDICT: {pred_text}")
                    logger.info("---")
            
            # Update metrics
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
        
        return loss
    
    def on_stage_start(self, stage, epoch):
        """Initialize metrics at start of each stage."""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Log results at end of each stage."""
        stage_stats = {"loss": stage_loss}
        
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
        
        if stage == sb.Stage.VALID:
            lr = self.hparams.lr_annealing.current_lr
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"]
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as f:
                    self.wer_metric.write_stats(f)


def normalize_text(text):
    """
    Normalize text for evaluation.
    - Convert to uppercase
    - Remove punctuation
    - Normalize whitespace
    """
    import string
    import re

    if not text or not text.strip():
        return ""

    text = text.upper()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text

def prepare_data(hparams, processor):
    """Prepare datasets with clean pipeline."""
    data_folder = hparams["data_folder"]
    
    # Load datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )
    
    if hparams["sorting"] == "ascending":
        train_data = train_data.filtered_sorted(sort_key="duration")
        hparams["train_loader_kwargs"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(sort_key="duration", reverse=True)
        hparams["train_loader_kwargs"]["shuffle"] = False
    
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    ).filtered_sorted(sort_key="duration")
    
    # Test datasets
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        ).filtered_sorted(sort_key="duration")
    
    datasets = [train_data, valid_data] + list(test_datasets.values())
    tokenizer = processor.tokenizer
    
    # Audio + prompt pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig", "prompt_tokens", "prompt_attention_mask")
    def audio_pipeline(wav):
        padded_audio, prompt_tokens = processor.process_audio(wav)
        prompt_attention_mask = (prompt_tokens != tokenizer.pad_token_id)
        
        yield padded_audio
        yield prompt_tokens
        yield prompt_attention_mask
    
    # Text pipeline
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides("wrd", "tokens", "tokens_eos")
    def text_pipeline(wrd):
        yield wrd
        
        tokens_list = tokenizer.encode(wrd, add_special_tokens=False)
        tokens = torch.LongTensor(tokens_list)
        tokens_eos = torch.LongTensor(tokens_list + [tokenizer.eos_token_id])
        
        yield tokens
        yield tokens_eos
    
    # Combined pipeline
    @sb.utils.data_pipeline.takes("prompt_tokens", "prompt_attention_mask", "tokens_eos")
    @sb.utils.data_pipeline.provides("full_input_ids", "full_attention_mask", "full_labels")
    def combine_pipeline(prompt_tokens, prompt_attention_mask, tokens_eos):
        # Combine prompt + transcription
        full_input_ids = torch.cat([prompt_tokens, tokens_eos])
        
        # Attention mask
        text_attention_mask = torch.ones_like(tokens_eos)
        full_attention_mask = torch.cat([prompt_attention_mask, text_attention_mask])
        
        # Mask prompt with -100, keep transcription tokens
        prompt_labels = torch.full_like(prompt_tokens, hparams["prompt_token"])
        full_labels = torch.cat([prompt_labels, tokens_eos])
        
        yield full_input_ids
        yield full_attention_mask
        yield full_labels
    
    # Add pipelines to datasets
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, combine_pipeline)
    
    # Set outputs
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id", "sig", "wrd", "tokens", "tokens_eos",
            "prompt_tokens", "prompt_attention_mask",
            "full_input_ids", "full_attention_mask", "full_labels"
        ],
    )
    
    return train_data, valid_data, test_datasets


def main():
    # Parse arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)
    
    # Load hyperparameters
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)
    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    # Prepare LibriSpeech data
    from librispeech_prepare import prepare_librispeech
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )
    
    # Prepare datasets
    processor = hparams["voxtral"].processor
    train_data, valid_data, test_datasets = prepare_data(hparams, processor)
    
    # Initialize model
    asr_brain = VoxtralASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["opt_class"],
    )
    
    # Load pretrained model if specified
    if "pretrainer" in hparams:
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected(asr_brain.device)
    
    # Add tokenizer to brain
    asr_brain.tokenizer = processor.tokenizer
    
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_loader_kwargs"],
        valid_loader_kwargs=hparams["valid_loader_kwargs"],
    )
    
    # Testing
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)
    for name, dataset in test_datasets.items():
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{name}.txt"
        )
        asr_brain.evaluate(
            dataset,
            test_loader_kwargs=hparams["test_loader_kwargs"],
            min_key="WER",
        )


if __name__ == "__main__":
    main()