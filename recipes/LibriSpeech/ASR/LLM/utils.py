import math
import logging
import pathlib as pl
import kaldiio
import torch
import torchaudio
import numpy as np
from tqdm.auto import tqdm
import speechbrain as sb
from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataio import load_pkl, save_pkl

class TokensLoader:
    """
    A loader class for retrieving tokens corresponding to utterance IDs.

    Arguments
    ---------
    data_path: str
        The path to the data directory containing the token files.
    save_name: str, optional
        The base name of the tokens files (default: "tokens").
    """

    def __init__(
        self,
        data_path,
        save_name="tokens",
    ):
        self.data_path = pl.Path(data_path)
        if not self.data_path.exists():
            raise ValueError(f"Data folder not found: {self.data_path.as_posix()}")
        self.tokens = self._load(data_path, save_name)
        # Debug the .scp file parsing
        # print(f"Loaded {len(self.tokens)} tokens")
        # print("First 5 keys:")
        # for i, key in enumerate(list(self.tokens.keys())[:5]):
        #     print(f"  {i}: '{key}'")
        # print("First 5 values:")
        # for i, (key, value) in enumerate(list(self.tokens.items())[:5]):
        #     print(f"  {i}: '{key}' -> '{value}'")

    def tokens_by_uttid(self, utt_id, num_codebooks=None):
        """
        Retrieves the tokens corresponding to a given utterance ID.

        Arguments
        ---------
        utt_id : str
            The utterance ID to retrieve tokens for.
        num_codebooks : int or list, optional
            The number of codebooks to retrieve from the tokens. If specified as an int, the tokens
            will be truncated to include only the first `num_codebooks` codebooks. If specified as a list,
            the tokens will include only the codebooks at the specified indices. If not specified, all codebooks are returned.

        Returns
        -------
        result : torch.LongTensor [T, N_Q]
            The tokens associated with the utterance ID, possibly truncated to `num_codebooks` codebooks.

        Raises
        ------
        KeyError
            If the utterance ID is not found in the tokens.
        ValueError
            If `num_codebooks` is invalid or exceeds the number of available codebooks.
        """
        if utt_id not in self.tokens:
            raise KeyError(f"Utterance ID '{utt_id}' not found in tokens.")
        tokens_path = self.tokens[utt_id]
        tokens = kaldiio.load_mat(tokens_path)
        tokens = torch.tensor(tokens)
        


        if num_codebooks is not None:
            if isinstance(num_codebooks, int):
                if num_codebooks <= 0:
                    raise ValueError(
                        f"Invalid num_codebooks value: {num_codebooks}. It must be a positive integer."
                    )
                if num_codebooks > tokens.size(-1):
                    raise ValueError(
                        f"Invalid number of codebooks: {num_codebooks}. "
                        f"Available codebooks: {tokens.size(-1)}."
                    )
                tokens = tokens[:, :num_codebooks]
            elif isinstance(num_codebooks, list):
                if not all(
                    isinstance(idx, int) and 0 <= idx < tokens.size(-1)
                    for idx in num_codebooks
                ):
                    raise ValueError(
                        f"Invalid indices in num_codebooks list: {num_codebooks}. "
                        f"All indices must be integers within the range [0, {tokens.size(-1) - 1}]."
                    )
                tokens = tokens[:, num_codebooks]
            else:
                raise ValueError("num_codebooks must be an int or a list.")

        return tokens

    # def _load(self, data_path, save_name):
    #     """
    #     Loads the mapping from utterance IDs to token file paths.

    #     Arguments
    #     ---------
    #     data_path: str
    #         The path to the data directory containing the token files.
    #     save_name: str
    #         The base name of the tokens files.

    #     Returns
    #     -------
    #     utt2toks: dict
    #         A dictionary mapping utterance IDs to their corresponding token file paths.
    #     """
    #     scp_path = f"{data_path}/{save_name}.scp"
    #     with open(scp_path, "r") as f:
    #         utt2toks = {
    #             line.strip().split(None, 1)[0]: line.strip().split(None, 1)[1]
    #             for line in f
    #             if line.strip()
    #         }
    #     return utt2toks

    def _load(self, data_path, save_name):
        """
        Loads the mapping from utterance IDs to token file paths.
        """
        scp_path = f"{data_path}/{save_name}.scp"
        with open(scp_path, "r") as f:
            return {
                line.split(' ', 1)[0].split('_')[0]: line.split(' ', 1)[1]
                for line in f
                if line.strip() and ' ' in line.strip()
            }

    def load_pretrained_embeddings(self, data_path, save_name="embeddings"):
        """
        Loads pretrained embeddings from a specified path.

        Arguments
        ---------
        data_path : str
            The directory where the embeddings are saved.
        save_name : str, optional
            The name of the embeddings file (default: "embeddings").

        Returns
        -------
        embeddings : torch.Tensor
            The loaded embeddings as a PyTorch tensor.

        Raises
        ------
        FileNotFoundError
            If the embeddings file does not exist at the specified path.
        """
        data_path = pl.Path(data_path).absolute()
        if not self.data_path.exists():
            raise ValueError(f"Data folder not found: {data_path.as_posix()}")
        embeddings = np.load(data_path / f"{save_name}.npy")
        embeddings = torch.tensor(embeddings)
        
        return embeddings