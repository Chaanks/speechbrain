from utils import TokensLoader

tokens_path = "/local_disk/brontes/smdhaffar/librispeech_SAMU_XLSR/"
tokens_save_name = "merged"

tokens_loader = TokensLoader(data_path=tokens_path, save_name=tokens_save_name)
