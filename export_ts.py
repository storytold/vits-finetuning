import os
import json
import argparse

import torch
import numpy as np
import os, shutil

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
from bertfe import BERTFrontEnd
from moji.moji import TorchMoji

import sys

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

from text.cleaners import arpa_cleaners
from text import cleaned_text_to_sequence
def get_arpa_text(text, hps):
  text_pre = arpa_cleaners(text)
  print(text_pre)
  text_norm = cleaned_text_to_sequence(text_pre)
  if hps.data.add_blank:
    text_norm = commons.intersperse(text_norm, 0)
  text_norm = torch.LongTensor(text_norm)
  return text_norm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="Output path of traced model file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to load model into and export. cpu or cuda",
    )
    parser.add_argument(
        "--test-string",
        type=str,
        required=True,
        help="String to test with",
    )
    parser.add_argument("--bert", default="huawei-noah/TinyBERT_General_4L_312D")

    args = parser.parse_args()
    
    device = torch.device(args.device)

    if not os.path.isfile(args.checkpoint):
      print(f"Model path {args.checkpoint} incorrect, file not found.")
    
    bert_model_name = args.bert
 
    print(f"Loading model {args.checkpoint} with config {args.config}...")
    moji = TorchMoji(verbose=True)
    bert_f = BERTFrontEnd(model_name=bert_model_name)
    
    hps = utils.get_hparams_from_file(args.config)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.checkpoint, net_g, None)
    
    test_string = args.test_string
    
    if "arpa_cleaners" in hps.data.text_cleaners:
        TT_stn_tst = get_arpa_text(test_string, hps)
    else:
        TT_stn_tst = get_text(test_string, hps)
    

    TT_x_tst = TT_stn_tst.unsqueeze(0).to(device)
    TT_x_tst_lengths = torch.LongTensor([TT_stn_tst.size(0)]).to(device)
    TT_len_scale = torch.FloatTensor([1.]).to(device)
    TT_em = moji(test_string).squeeze().unsqueeze(0).to(device) # [1, 1, 2304] -> [2304] -> [1, 2304]
        
    TT_bert_emb, _ = bert_f.infer(test_string) # [1, tokens, channels]
    TT_bert_emb = TT_bert_emb.to(device)
    TT_bert_lens = torch.LongTensor([TT_bert_emb.size(1)]).to(device)
    print("Doing build model run..")
    net_g.infer_ts(TT_x_tst, TT_x_tst_lengths, TT_em, TT_bert_emb, TT_bert_lens,TT_len_scale)
    
    
    print("Tracing and exporting...")
    traced_vits_noi = torch.jit.trace_module(net_g, 
                                         {"infer_ts_noistft": (TT_x_tst, TT_x_tst_lengths, TT_em, TT_bert_emb, TT_bert_lens,TT_len_scale)}
                                         ,check_trace=False,strict=True)

    torch.jit.save(traced_vits_noi, args.out_path)

    
 