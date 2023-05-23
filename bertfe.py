import torch

import io
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

#TODO: Add batch inference fun
class BERTFrontEnd():
  def __init__(self,is_cuda = False,model_name = "huawei-noah/TinyBERT_General_4L_312D"): 
    self.model = AutoModel.from_pretrained(model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.is_cuda = is_cuda
    
    if is_cuda:
      self.model = self.model.cuda()

    print("Loaded BERT")
    
  #Perform inference with single text
  # in_txt: String to infer from
  # returns: hidden states, [1, n_tokens, bert_size] ; pooled, [1, bert_size]
  def infer(self,in_txt):
    inputs = self.tokenizer(in_txt, return_tensors="pt")
    

    if self.is_cuda:
      inputs["input_ids"] = inputs["input_ids"].cuda()
      inputs["token_type_ids"] = inputs["token_type_ids"].cuda()
      inputs["attention_mask"] = inputs["attention_mask"].cuda()

    with torch.no_grad():
      encoded_layers, pooled = self.model(**inputs,return_dict=False)
    
    return encoded_layers, pooled