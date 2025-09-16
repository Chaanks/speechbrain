"""This lobe enables the integration of huggingface pretrained Voxtral model.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Jarod Duret 2025
"""

import io
import torch
import torch.nn as nn
import torchaudio
import speechbrain as sb
import soundfile as sf
from functools import cached_property
from typing import Optional, List, Dict, Any

from speechbrain.integrations.huggingface.huggingface import (
    HFTransformersInterface,
)
from speechbrain.utils.logger import get_logger
from transformers import VoxtralForConditionalGeneration, WhisperFeatureExtractor, MistralCommonTokenizer
from transformers import AutoProcessor
from mistral_common.protocol.transcription.request import TranscriptionRequest

logger = get_logger(__name__)

# Voxtral constants
SAMPLE_RATE = 16000
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
MAX_SOURCE_POSITIONS = 3000


class VoxtralProcessor(nn.Module):
    """Processor for Voxtral that handles audio processing and prompt creation.

    Arguments
    ---------
    source : str
        HuggingFace Voxtral model source (e.g., "mistralai/Voxtral-Mini-3B-2507").
    save_path : str, optional
        Path to cache the feature extractor.
    sampling_rate : int
        Expected sampling rate (default: 16000).
    language : str
        Language for transcription (default: "en").
    max_source_positions : int
        Maximum sequence length for features (default: 3000).
    model_id : str, optional
        Model ID for transcription requests (defaults to source).
    enable_chunking : bool
        If False, audio limited to 30s for training mode.
        If True, full audio length with chunking for inference mode (default: False).
    """
    
    def __init__(
        self, 
        source,
        save_path=None,
        sampling_rate=16000,
        language="en",
        max_source_positions=3000,
        model_id=None,
        enable_chunking=False,
    ):
        super().__init__()
        self.source = source
        self.sampling_rate = sampling_rate
        self.language = language
        self.max_source_positions = max_source_positions
        self.model_id = model_id or source
        self.enable_chunking = enable_chunking

        logger.info(f"Loading MistralCommonTokenizer for {source}")
        self.tokenizer = MistralCommonTokenizer.from_pretrained(source)

        self.audio_token_id = 24
        self.audio_token = self.tokenizer.convert_ids_to_tokens(self.audio_token_id)
        
        # Load WhisperFeatureExtractor and extract required components
        logger.info(f"Loading feature extractor components from {source}")
        self._load_feature_extractor_components(source, save_path, sampling_rate)
        
        mode_str = "full length with chunking" if enable_chunking else "truncated (30s max)"
        logger.info(f"VoxtralProcessor initialized with {mode_str}")
        
    def _load_feature_extractor_components(self, source, save_path, sampling_rate):
        """Load and extract required components from WhisperFeatureExtractor."""
        from transformers import WhisperFeatureExtractor
        
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            source, cache_dir=save_path
        )
        
        # Extract essential parameters
        self._n_fft = feature_extractor.n_fft
        self._hop_length = feature_extractor.hop_length
        self._n_samples = feature_extractor.n_samples
        self.feature_size = feature_extractor.feature_size
        
        # Handle mel filters with backward compatibility
        mel_filters = feature_extractor.mel_filters
        if mel_filters.shape[0] != feature_extractor.feature_size:
            mel_filters = mel_filters.T
        assert mel_filters.shape[0] == feature_extractor.feature_size
        self.register_buffer(
            "_mel_filters", torch.as_tensor(mel_filters, dtype=torch.float32)
        )
        
        # Validate sampling rate
        if hasattr(feature_extractor, 'sampling_rate'):
            if feature_extractor.sampling_rate != sampling_rate:
                logger.warning(
                    f"Feature extractor sampling rate ({feature_extractor.sampling_rate}) "
                    f"differs from expected ({sampling_rate})"
                )
        
        del feature_extractor
        
    @property
    def n_samples(self):
        """Number of samples for 30-second chunks (480000)."""
        return self._n_samples
    
    def pad_or_trim_audio(self, audio_batch, target_length=None):
        """Pad or trim batch of audio to target length.
        
        Arguments
        ---------
        audio_batch : torch.Tensor
            Audio batch of shape (batch_size, n_samples) or (batch_size, n_samples, channels).
        target_length : int, optional
            Target length in samples. If None, uses n_samples when enable_chunking=False,
            or keeps original length when enable_chunking=True.

        Returns
        -------
        torch.Tensor
            Audio batch of shape (batch_size, target_length).
        """
        assert audio_batch.dim() >= 2, f"Expected batched audio (batch_size, n_samples), got {audio_batch.shape}"
        
        if target_length is None:
            # For training: fixed 30s length. For inference: keep original length
            target_length = self.n_samples if not self.enable_chunking else audio_batch.shape[1]
            
        # Convert to mono if multi-channel (select first channel)
        if audio_batch.dim() == 3:
            audio_batch = audio_batch[:, :, 0]  # Take first channel
        
        batch_size, current_length = audio_batch.shape
        
        if current_length > target_length and not self.enable_chunking:
            # Trim to target length (only in training mode)
            audio_batch = audio_batch[:, :target_length]
        elif current_length < target_length:
            # Zero-pad to target length
            padding_needed = target_length - current_length
            audio_batch = torch.nn.functional.pad(audio_batch, (0, padding_needed))
        
        return audio_batch

    def _audio_to_buffer(self, audio_tensor):
        """Convert single audio tensor to BytesIO buffer for mistral_common.
        
        Arguments
        ---------
        audio_tensor : torch.Tensor
            Audio tensor of shape (n_samples,).

        Returns
        -------
        io.BytesIO
            Audio data as WAV buffer.
        """
        buffer = io.BytesIO()
        
        # Convert to numpy for soundfile
        audio_array = audio_tensor.cpu().numpy()
        
        # Write as WAV format to buffer
        sf.write(buffer, audio_array, samplerate=self.sampling_rate, format='wav')
        buffer.seek(0)
        
        return buffer

    def extract_mel_features(self, audio_batch):
        """Extract mel features from batched audio.
        
        Arguments
        ---------
        audio_batch : torch.Tensor
            Batched audio of shape (batch_size, n_samples)
            
        Returns
        -------
        torch.Tensor
            Training mode (enable_chunking=False): 
                (batch_size, feature_size, max_source_positions)
            Inference mode (enable_chunking=True):
                (batch_size, n_chunks, feature_size, max_source_positions)
        """
        assert audio_batch.dim() == 2, f"Expected batched audio (batch_size, n_samples), got {audio_batch.shape}"
            
        window = torch.hann_window(self._n_fft, device=audio_batch.device)
        stft = torch.stft(audio_batch, self._n_fft, self._hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = self._mel_filters.to(audio_batch.device, dtype=magnitudes.dtype)
        mel_spec = mel_filters @ magnitudes
        
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        
        batch_size, feature_size, time_steps = log_spec.shape
        
        if not self.enable_chunking:
            # Training mode: no chunk dimension, fixed max_source_positions
            if time_steps != self.max_source_positions:
                if time_steps > self.max_source_positions:
                    log_spec = log_spec[:, :, :self.max_source_positions]
                else:
                    padding = self.max_source_positions - time_steps
                    log_spec = torch.nn.functional.pad(log_spec, (0, padding))
            
            # Return without chunk dimension: (batch_size, feature_size, max_source_positions)
            return log_spec
            
        else:
            # Inference mode: preserve chunk structure for context handling
            if time_steps <= self.max_source_positions:
                # Single chunk - pad if needed
                padding = max(0, self.max_source_positions - time_steps)
                if padding > 0:
                    log_spec = torch.nn.functional.pad(log_spec, (0, padding))
                # Add chunk dimension: (batch_size, 1, feature_size, max_source_positions)
                return log_spec.unsqueeze(1)
            else:
                # Multiple chunks
                n_chunks = (time_steps + self.max_source_positions - 1) // self.max_source_positions
                padded_time = n_chunks * self.max_source_positions
                padding = padded_time - time_steps
                if padding > 0:
                    log_spec = torch.nn.functional.pad(log_spec, (0, padding))
                
                # Reshape to chunks: (batch_size, n_chunks, feature_size, max_source_positions)
                chunked = log_spec.reshape(batch_size, feature_size, n_chunks, self.max_source_positions)
                return chunked.transpose(1, 2)

    def create_transcription_prompt(self, audio_tensor, language=None):
        """Create transcription prompt for single audio using mistral_common.
        
        Arguments
        ---------
        audio_tensor : torch.Tensor
            Single audio tensor of shape (n_samples,).
        language : str, optional
            Language for transcription (defaults to self.language).

        Returns
        -------
        torch.Tensor
            Prompt token IDs of shape (n_tokens,).
        """
        assert audio_tensor.dim() == 1, f"Expected single audio (n_samples,), got {audio_tensor.shape}"
        
        if language is None:
            language = self.language
        
        # Convert audio tensor to buffer
        audio_buffer = self._audio_to_buffer(audio_tensor)
        
        # Create OpenAI-style transcription request
        openai_transcription_request = {
            "model": self.model_id,
            "file": audio_buffer,
            "language": language,
        }
        
        # Use mistral_common to create transcription request
        transcription_request = TranscriptionRequest.from_openai(openai_transcription_request)
        tokenized_request = self.tokenizer.tokenizer.encode_transcription(transcription_request)
        
        return torch.tensor(tokenized_request.tokens, dtype=torch.long)
    
    def create_prompt_batch(self, audio_batch, language=None):
        """Create transcription prompts for batch of audio.
        
        Arguments
        ---------
        audio_batch : torch.Tensor
            Batched audio of shape (batch_size, n_samples)
        language : str, optional
            Language for transcription (defaults to self.language).
            
        Returns
        -------
        dict
            Dictionary containing:
            - input_ids: (batch_size, max_prompt_len) - padded prompt tokens
            - attention_mask: (batch_size, max_prompt_len) - attention masks
        """
        assert audio_batch.dim() == 2, f"Expected batched audio (batch_size, n_samples), got {audio_batch.shape}"
        
        batch_size = audio_batch.shape[0]
        
        # Process individual audios for prompts (mistral_common requires individual processing)
        prompt_tokens_list = []
        for i in range(batch_size):
            audio_sample = audio_batch[i]
            prompt_tokens = self.create_transcription_prompt(audio_sample, language)
            prompt_tokens_list.append(prompt_tokens)
        
        # Pad prompt tokens to same length
        max_prompt_len = max(tokens.shape[0] for tokens in prompt_tokens_list)
        pad_token_id = self.tokenizer.pad_token_id
        
        padded_prompts = []
        attention_masks = []
        
        for tokens in prompt_tokens_list:
            if tokens.shape[0] < max_prompt_len:
                padding = max_prompt_len - tokens.shape[0]
                tokens_padded = torch.nn.functional.pad(tokens, (0, padding), value=pad_token_id)
            else:
                tokens_padded = tokens
            
            padded_prompts.append(tokens_padded)
            attention_masks.append((tokens_padded != pad_token_id).long())
        
        return {
            'input_ids': torch.stack(padded_prompts),
            'attention_mask': torch.stack(attention_masks)
        }
    
    def process_audio(self, audio_path):
        """Audio processing for single file in data pipeline.
        
        Arguments
        ---------
        audio_path : str
            Path to audio file.

        Returns
        -------
        tuple
            (processed_audio, prompt_tokens) - both without batch dimension
        """
        # Load and process audio
        info = torchaudio.info(audio_path)
        audio = sb.dataio.dataio.read_audio(audio_path)
        if info.sample_rate != SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(
                info.sample_rate, SAMPLE_RATE
            )(audio)
        
        # Process as single sample then remove batch dimension
        audio_batch = audio.unsqueeze(0)  # Add batch dim
        processed_batch = self.pad_or_trim_audio(audio_batch)
        processed_audio = processed_batch.squeeze(0).cpu()  # Remove batch dim
        
        # Create prompt tokens
        prompt_tokens = self.create_transcription_prompt(processed_audio)
        
        return processed_audio, prompt_tokens
    
    def process_batch(self, audio_batch, text_batch=None):
        """Process batch of audio tensors for training/inference.
        
        Arguments
        ---------
        audio_batch : torch.Tensor
            Batch of audio tensors (batch_size, n_samples)
        text_batch : list[str], optional
            Batch of transcription texts for training
            
        Returns
        -------
        dict
            Dictionary containing:
            - input_features: mel features with appropriate shape for mode
            - input_ids: (batch_size, max_len) - padded tokens (prompt only or prompt+text)
            - attention_mask: (batch_size, max_len) - attention masks
            - labels: (batch_size, max_len) - if text_batch provided for training
        """
        assert audio_batch.dim() == 2, f"Expected batched audio (batch_size, n_samples), got {audio_batch.shape}"
        
        # Process audio features
        audio_batch = self.pad_or_trim_audio(audio_batch)
        input_features = self.extract_mel_features(audio_batch)
        
        # Create prompt tokens
        prompt_result = self.create_prompt_batch(audio_batch)
        
        result = {
            'input_features': input_features,
            'input_ids': prompt_result['input_ids'],
            'attention_mask': prompt_result['attention_mask']
        }
        
        # Add training labels if text provided
        if text_batch is not None:
            batch_size = len(text_batch)
            labels_list = []
            max_total_len = 0
            
            # Create labels for each sample
            for i, text in enumerate(text_batch):
                prompt_len = (prompt_result['attention_mask'][i] == 1).sum().item()
                
                # Tokenize text
                text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
                text_tokens = torch.LongTensor(text_tokens)
                
                # Create labels: mask prompt (-100), learn on text
                labels = torch.cat([
                    torch.full((prompt_len,), -100, dtype=torch.long),
                    text_tokens
                ])
                labels_list.append(labels)
                max_total_len = max(max_total_len, labels.shape[0])
            
            # Pad labels and extend input sequences
            padded_labels = []
            extended_input_ids = []
            extended_attention = []
            pad_token_id = self.tokenizer.pad_token_id
            
            for i, (text, labels) in enumerate(zip(text_batch, labels_list)):
                # Tokenize text again for extending input_ids
                text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
                text_tokens = torch.LongTensor(text_tokens)
                
                # Extend input_ids and attention_mask
                full_ids = torch.cat([prompt_result['input_ids'][i], text_tokens])
                full_attention = torch.cat([
                    prompt_result['attention_mask'][i],
                    torch.ones(text_tokens.shape[0], dtype=torch.long)
                ])
                
                # Pad to max_total_len
                if full_ids.shape[0] < max_total_len:
                    padding = max_total_len - full_ids.shape[0]
                    full_ids = torch.nn.functional.pad(full_ids, (0, padding), value=pad_token_id)
                    full_attention = torch.nn.functional.pad(full_attention, (0, padding), value=0)
                    labels = torch.nn.functional.pad(labels, (0, padding), value=-100)
                
                extended_input_ids.append(full_ids)
                extended_attention.append(full_attention)
                padded_labels.append(labels)
            
            result.update({
                'input_ids': torch.stack(extended_input_ids),
                'attention_mask': torch.stack(extended_attention),
                'labels': torch.stack(padded_labels)
            })
        
        return result
    
    def forward(self, audio_paths):
        """Process batch of audio paths - primarily for compatibility.
        
        Arguments
        ---------
        audio_paths : list[str]
            List of audio file paths.

        Returns
        -------
        dict
            Processed batch data with consistent shapes.
        """
        # Load all audio files and create batch
        audio_tensors = []
        for audio_path in audio_paths:
            audio = sb.dataio.dataio.read_audio(audio_path)
            audio_tensors.append(audio)
        
        # Create batched tensor (pad to same length if needed)
        max_len = max(audio.shape[0] for audio in audio_tensors)
        padded_audios = []
        for audio in audio_tensors:
            if audio.shape[0] < max_len:
                padding = max_len - audio.shape[0]
                audio = torch.nn.functional.pad(audio, (0, padding))
            padded_audios.append(audio)
        
        audio_batch = torch.stack(padded_audios)
        
        # Process batch
        return self.process_batch(audio_batch)


class Voxtral(HFTransformersInterface):
    """SpeechBrain integration for HuggingFace pretrained Voxtral model.
    
    Provides clean interface to Voxtral with consistent batch processing
    and mode-appropriate feature extraction for training/inference scenarios.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "mistralai/Voxtral-Mini-3B-2507"
    save_path : str
        Path (dir) of the downloaded model.
    freeze : bool (default: False)
        If True, the model is frozen.
    freeze_encoder : bool (default: False)
        If True, only the audio encoder is frozen.
    language : str (default: "en")
        Language for transcription requests.
    device : any, optional
        Device to migrate the model to.
    enable_chunking : bool (default: False)
        If False, training mode with 30s audio limit.
        If True, inference mode with full chunking support.

    Example
    -------
    >>> model_hub = "mistralai/Voxtral-Mini-3B-2507"
    >>> save_path = "savedir"
    >>> model = Voxtral(model_hub, save_path, language="en")
    >>> # Inputs prepared by SpeechBrain pipeline
    >>> outputs = model(**voxtral_inputs)
    """

    def __init__(
        self,
        source: str,
        save_path: str,
        freeze: bool = False,
        freeze_encoder: bool = False,
        language: str = "en",
        device: str = None,
        enable_chunking: bool = False,
        **kwargs,
    ):
        self.freeze_encoder = freeze_encoder
        self.source = source
        self.language = language
        
        # Initialize the base class
        super().__init__(
            source=source,
            save_path=save_path,
            freeze=freeze,
            device=device,
            **kwargs,
        )        

        self.processor = VoxtralProcessor(
            source=source,
            save_path=save_path,
            language=language,
            enable_chunking=enable_chunking,
        )
        
        # Freeze encoder if requested
        if not self.freeze and self.freeze_encoder:
            logger.warning(
                "speechbrain.integrations.huggingface.voxtral - Voxtral encoder is frozen."
            )
            for param in self.model.audio_tower.parameters():
                param.requires_grad = False
            for param in self.model.multi_modal_projector.parameters():
                param.requires_grad = False

    def _load_model(self, source, save_path, **kwargs):
        """Override to load VoxtralForConditionalGeneration specifically"""
        return VoxtralForConditionalGeneration.from_pretrained(
            source, 
            cache_dir=save_path,
            **kwargs
        )

    def forward(self, **kwargs):
        """
        Forward pass through Voxtral model.
        
        Arguments
        ---------
        **kwargs : dict
            Should contain 'input_ids', 'input_features', 'attention_mask', 
            and optionally 'labels' - all prepared by the data pipeline.
        
        Returns
        -------
        output : CausalLMOutputWithPast
            Model output from VoxtralForConditionalGeneration.forward()
        """
        if self.freeze:
            with torch.no_grad():
                return self.model(**kwargs)
        else:
            return self.model(**kwargs)

    def generate(self, **kwargs) -> torch.Tensor:
        """
        Generate transcription from prepared inputs.
        
        Arguments
        ---------
        **kwargs : dict
            Should contain 'input_ids', 'input_features', 'attention_mask'
            prepared by the data pipeline for generation (no labels).
        
        Returns
        -------
        generated_ids : torch.Tensor
            Generated token IDs.
        """
        with torch.no_grad():
            return self.model.generate(**kwargs)

    def batch_decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """
        Decode token IDs to text.
        
        Arguments
        ---------
        token_ids : torch.Tensor
            Token IDs to decode.
        skip_special_tokens : bool
            Whether to skip special tokens in decoding.
            
        Returns
        -------
        decoded_texts : List[str]
            Decoded text strings.
        """
        return self.processor.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    @cached_property
    def pad_token_id(self) -> int:
        """Returns the padding token ID."""
        return self.processor.tokenizer.pad_token_id

    @cached_property
    def eos_token_id(self) -> int:
        """Returns the EOS token ID."""
        return self.processor.tokenizer.eos_token_id

    @cached_property
    def bos_token_id(self) -> int:
        """Returns the BOS token ID.""" 
        return self.processor.tokenizer.bos_token_id

    @cached_property
    def audio_token_id(self) -> int:
        """Returns the audio token ID used by Voxtral."""
        return self.processor.audio_token_id