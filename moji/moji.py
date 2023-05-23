# imports
import json
import torch

from moji.sentence_tokenizer import SentenceTokenizer
from moji.model_def import torchmoji_feature_encoding
from moji.global_variables import PRETRAINED_PATH, VOCAB_PATH

class TorchMoji():
    def __init__(self, verbose=False):
        if verbose:
            print(f'Tokenizing using dictionary from {VOCAB_PATH}')
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)
        
        self.torchmoji_tokenizer = SentenceTokenizer(vocabulary, fixed_length=120)
        
        if verbose:
            print(f'Loading model from {PRETRAINED_PATH}.')
        self.torchmoji_model = torchmoji_feature_encoding(PRETRAINED_PATH)
    
    def __call__(self, text):# str[txt_T]
        with torch.no_grad():
            tokenized, _, _ = self.torchmoji_tokenizer.tokenize_sentences([text,])
            embed = self.torchmoji_model(tokenized)
        return torch.from_numpy(embed).unsqueeze(0).float()# [1, 1, embed]