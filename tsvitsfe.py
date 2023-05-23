import torch
import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import utils
from text import text_to_sequence
from bertfe import BERTFrontEnd
from moji.moji import TorchMoji
from stft import TorchSTFT
from text.cleaners import arpa_cleaners
from text import cleaned_text_to_sequence


class TSVITSFE:
  # device can be cpu or cuda
  def load(self, model_path, config_path, device, bert_model_name = "huawei-noah/TinyBERT_General_4L_312D"):
    self.model = torch.jit.load(model_path).to(device)
    self.moji = TorchMoji(verbose=True)
    self.bert_f = BERTFrontEnd("cuda" in device, model_name=bert_model_name)
    self.hps = utils.get_hparams_from_file(config_path)
    self.stft = TorchSTFT(self.hps.model.gen_istft_n_fft, hop_length= self.hps.model.gen_istft_hop_size,
                        win_length=self.hps.model.gen_istft_n_fft).to(device)
    self.device = device

  
 

  def get_arpa_text(self, text):
    text_pre = arpa_cleaners(text)
    text_norm = cleaned_text_to_sequence(text_pre)
    if self.hps.data.add_blank:
      text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm
 
  def get_text(self, text):
    text_norm = text_to_sequence(text, self.hps.data.text_cleaners)
    if self.hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm
  
  # Inference
  # in_em = Input to the emotion predictor. Default empty, where it will use same as text
  def infer(self,in_text, in_em = "", in_speed = 1.):
  
    if "arpa_cleaners" in self.hps.data.text_cleaners:
      stn_tst = self.get_arpa_text(in_text)
    else:
      stn_tst = self.get_text(in_text)
    
    if len(in_em) == 0:
      in_em = in_text

    with torch.no_grad():
      x_tst = stn_tst.unsqueeze(0).to(self.device)
      x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
      len_scale = torch.FloatTensor([in_speed]).to(self.device)
      em = self.moji(in_em).squeeze().unsqueeze(0).to(self.device) # [1, 1, 2304] -> [2304] -> [1, 2304]
        
      bert_emb, _ = self.bert_f.infer(in_text) # [1, tokens, channels]
      bert_emb = bert_emb.to(self.device)
      bert_lens = torch.LongTensor([bert_emb.size(1)]).to(self.device)

      spec, phase, x1, attn = self.model.infer_ts_noistft(x_tst, x_tst_lengths, em, bert_emb, bert_lens, len_scale)
      audio = self.stft.inverse(spec, phase)
      audio = audio.squeeze().cpu().data.float().numpy()
    
    return audio
