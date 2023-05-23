import argparse
import text
from utils import load_filepaths_and_text
from tqdm import tqdm
import os
from moji.moji import TorchMoji
import torch
from bertfe import BERTFrontEnd

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=1, type=int)
  parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
  parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])
  parser.add_argument("--bert", default="huawei-noah/TinyBERT_General_4L_312D")
  

  args = parser.parse_args()
  moji = TorchMoji(verbose=True)
  bert_f = BERTFrontEnd(model_name=args.bert)
  
  
  if "arpa_cleaners" in args.text_cleaners:
    print("ARPA cleaners detected, will run in dual mode")
    
  for filelist in args.filelists:
    print("START:", filelist)
    filepaths_and_text = load_filepaths_and_text(filelist)
    new_fp_text = []
    
    for i in tqdm(range(len(filepaths_and_text))):
      original_text = filepaths_and_text[i][args.text_index]
      original_fname = filepaths_and_text[i][0].split(".")[0] # wavs/cat.wav -> wavs/cat
      
      moji_filename = original_fname + ".torchmoji"
      if not os.path.isfile(moji_filename):
        moji_result = moji(original_text)
        torch.save(moji_result, moji_filename)
        
        
      bert_filename = original_fname + ".bert"
      if not os.path.isfile(bert_filename):
        bert_res, _ = bert_f.infer(original_text)
        torch.save(bert_res,bert_filename)
        
      
      cleaned_text = text._clean_text(original_text, args.text_cleaners)
      text_arrs = filepaths_and_text[i]
      text_arrs[args.text_index] = cleaned_text
      new_fp_text.append(text_arrs)
      if "arpa_cleaners" in args.text_cleaners:
        # ZDisket: Python's default behavior when equalling a list is a reference, so unless we use .copy() this ends up
        # overriding the previous entry and leading into 2 text entries (wtf, why like this?)
        text_arrs2 = filepaths_and_text[i].copy()
        text_arrs2[args.text_index] = text._clean_text(original_text, ["arpa_precleaners"])
        new_fp_text.append(text_arrs2)
        

    new_filelist = filelist + "." + args.out_extension
    with open(new_filelist, "w", encoding="utf-8") as f:
      for x in new_fp_text:
        f.write("|".join(x) + "\n")
      
