#!/usr/bin/env/python3
"""
TODO

Authors
 * Jarod Duret 2025
"""

import os
import sys
from pathlib import Path

import torch
import transformers
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger
from utils import TokensLoader

logger = get_logger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        audio_tokens, _ = batch.audio_tokens

        tokens_prompt_transcription, tokens_prompt_transcription_len = (
            batch.tokens_prompt_transcription
        )  # Includes prompt and transcription
        prompt_len = batch.prompt_len

        down_feats_proj = self.modules.proj(audio_tokens)

        # Format input for LLM; [ audio emb ] + [ prompt emb ] + [ transcription emb ]
        # First get relevant lengths
        audio_len = 1
        text_len = tokens_prompt_transcription.shape[1]  # Max sequence length
        batch_size = tokens_prompt_transcription.shape[0]
        audio_prompt_len = (audio_len + prompt_len)[0]

        if hasattr(self.modules.llm, "module"):
            embeddings = (
                self.modules.llm.module.model.get_input_embeddings()
            )
        else:
            embeddings = self.modules.llm.model.get_input_embeddings()

        text_embeds = embeddings(tokens_prompt_transcription)

        if down_feats_proj.dtype != text_embeds.dtype:
            down_feats_proj = down_feats_proj.to(text_embeds.dtype)

        inputs_embeds = torch.cat(
            (down_feats_proj.unsqueeze(1), text_embeds), dim=1
        )

        # Prepare attn_mask for audio and text and combine them. This is not streaming compatible. For HF to work, masked frames should be 0.
        pad_token_id = self.hparams.pad_token
        text_mask = (tokens_prompt_transcription != pad_token_id)
        text_abs_len = text_mask.sum(dim=1)

        # Create attention masks
        audio_len_tensor = torch.ones(batch_size, dtype=torch.long, device=tokens_prompt_transcription.device)
        audio_attn_mask = length_to_mask(audio_len_tensor, max_len=1)
        text_attn_mask = length_to_mask(text_abs_len, max_len=text_len)
        attn_mask = torch.cat([audio_attn_mask, text_attn_mask], dim=-1)

        # LLM forward
        llm_logits = self.modules.llm(
            inputs_embeds=inputs_embeds, attention_mask=attn_mask
        ).logits

        p_seq = self.hparams.log_softmax(llm_logits)


        if hasattr(self.modules.llm, "module"):
            gen_func = self.modules.llm.module.model.generate
        else:
            gen_func = self.modules.llm.model.generate

        # Running decoding if not training
        if stage == sb.Stage.TRAIN:
            hyps = None

        elif stage == sb.Stage.VALID:
            # Define generation config depending on runtime values
            config = transformers.GenerationConfig(
                num_beams=self.hparams.valid_beam_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                max_new_tokens=200,
            )

            hyps = gen_func(
                inputs_embeds=inputs_embeds[
                    :, :audio_prompt_len
                ],  # give model audio features and prompt for inference
                attention_mask=attn_mask[:, :audio_prompt_len],
                generation_config=config,
            )
        elif stage == sb.Stage.TEST:
            # Define generation config depending on runtime values
            config = transformers.GenerationConfig(
                num_beams=self.hparams.test_beam_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                max_new_tokens=int(2 * audio_len),
            )

            hyps = gen_func(
                inputs_embeds=inputs_embeds[
                    :, :audio_prompt_len
                ],  # give model audio features and prompt for inference
                attention_mask=attn_mask[:, :audio_prompt_len],
                generation_config=config,
            )

        return p_seq, hyps, audio_prompt_len

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (
            p_seq,
            predicted_tokens,
            audio_prompt_len,
        ) = predictions

        ids = batch.id
        tokens_transcription, tokens_transcription_len = batch.tokens_transcription

        p_seq_transcription_only = p_seq[:, audio_prompt_len - 1 :]

        loss = self.hparams.nll_loss(
            p_seq_transcription_only,
            tokens_transcription,
            length=tokens_transcription_len,
        )

        if stage != sb.Stage.TRAIN:
            # Decode with automatic special token removal
            predictions_text = self.tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
            targets_text = self.tokenizer.batch_decode(tokens_transcription, skip_special_tokens=True)
            
            # Clean up text (remove extra whitespace, newlines, etc.)
            predictions_text = [pred.strip().replace('\n', ' ') for pred in predictions_text]
            targets_text = [tgt.strip().replace('\n', ' ') for tgt in targets_text]

            # Convert to words for WER computation
            predicted_words = [pred.split() for pred in predictions_text]
            target_words = [tgt.split() for tgt in targets_text]

            print(predicted_words)
            print(target_words)
            
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            # self.checkpointer.save_and_keep_only(
            #     meta={"WER": stage_stats["WER"]},
            #     min_keys=["WER"],
            # )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the model optimizer"
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )
        # save the optimizers in a dictionary
        # the key will be used in `freeze_optimizers()`
        self.optimizers_dict = {
            "model_optimizer": self.model_optimizer,
        }
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)


def remove_after_eos(list_of_str, eos_wrd="<|end_of_text|>"):
    """Remove all the text after EOS to obtain the clean translation. Receives a list of string e.g. ['the cat<|end_of_text|>[PAD]']"""
    cleaned = []
    for line in list_of_str:
        index = line.find(eos_wrd)
        if index != -1:
            cleaned.append(line[:index])
        else:
            cleaned.append(line)

    return cleaned


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    # for csv_file in hparams["test_csv"]:
    #     name = Path(csv_file).stem
    #     test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
    #         csv_path=csv_file, replacements={"data_root": data_folder}
    #     )
    #     test_datasets[name] = test_datasets[name].filtered_sorted(
    #         sort_key="duration"
    #     )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    tokens_loader = TokensLoader(data_path=hparams["tokens_path"], save_name=hparams["tokens_save_name"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("audio_tokens")
    def audio_pipeline(utt_id):
        tokens = tokens_loader.tokens_by_uttid(utt_id)
        return tokens

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)


    # Get the prompt from yaml and tokenize it
    prompt = hparams["llm_prompt"]
    logger.info(f"Using the following prompt: {repr(prompt)}")

    # Don't add EOS after prompt, only add EOS after transcripts
    # Always manually add eos and bos because HF is not consistent.
    eos_token_id = torch.LongTensor([tokenizer.eos_token_id])
    bos_token_id = torch.LongTensor([tokenizer.bos_token_id])

    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.squeeze()

    prompt_ids = torch.cat([prompt_ids, bos_token_id])

    # BOS + prompt + translation + EOS
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "transcription",
        "tokens_transcription",
        "tokens_prompt_transcription",
        "prompt_len",
    )
    def asr_text_pipeline(transcription):
        yield transcription
        tokens_transcription = tokenizer(
            transcription, return_tensors="pt", add_special_tokens=False
        ).input_ids.squeeze()
        no_eos_trans = tokens_transcription
        tokens_transcription = torch.cat([tokens_transcription, eos_token_id])
        yield tokens_transcription
        tokens_prompt_transcription = torch.cat((prompt_ids, no_eos_trans))
        yield tokens_prompt_transcription
        prompt_len = prompt_ids.size(0)
        yield prompt_len

    sb.dataio.dataset.add_dynamic_item(datasets, asr_text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "audio_tokens", "transcription", "tokens_transcription", "tokens_prompt_transcription", "prompt_len"],
    )

    return train_data, valid_data, test_datasets


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            # "dev_splits": hparams["dev_splits"],
            # "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Defining tokenizer and loading it
    tokenizer = hparams["modules"]["llm"].tokenizer
    vocab_size = len(tokenizer.get_vocab())


    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(
        hparams, tokenizer
    )

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )


    asr_brain.tokenizer = tokenizer

    # convert LoRA adapters to fp16
    if hparams["precision"] == "fp16":
        print("DEBUG: Converting LoRA adapters to fp16...")
        target_dtype = torch.float16
        for name, module in asr_brain.modules.llm.named_modules():
            if hasattr(module, "adapter_down_proj") and hasattr(module, "adapter_up_proj"):
                module.adapter_down_proj = module.adapter_down_proj.to(target_dtype)
                module.adapter_up_proj = module.adapter_up_proj.to(target_dtype)

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    # if not os.path.exists(hparams["output_wer_folder"]):
    #     os.makedirs(hparams["output_wer_folder"])

    # for k in test_datasets.keys():  # keys are test_clean, test_other etc
    #     asr_brain.hparams.test_wer_file = os.path.join(
    #         hparams["output_wer_folder"], f"wer_{k}.txt"
    #     )
    #     asr_brain.evaluate(
    #         test_datasets[k],
    #         test_loader_kwargs=hparams["test_dataloader_opts"],
    #         min_key="WER",
    #     )