import argparse
import text
from utils import load_filepaths_and_text
from tqdm import tqdm
import os
from moji.moji import TorchMoji
import torch
from bertfe import BERTFrontEnd
import commons

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=1, type=int)
  parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
  parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])
  parser.add_argument("--bert", default="huawei-noah/TinyBERT_General_4L_312D")
  parser.add_argument("--test", default="")
  

  args = parser.parse_args()
  moji = TorchMoji(verbose=True)
  bert_f = BERTFrontEnd(model_name=args.bert)
  test_sentences = ["The quick brown fox jumps over the lazy dog",
                    "In a galaxy far, far away, a young hero embarks on an epic adventure",
                    "Ladies and gentlemen, welcome to the annual science fair!",
                    "Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
                    "The crowd erupted in cheers as the team scored the winning goal in the final seconds of the match"]
  
  if len(args.test) > 1:
    with open(args.test) as f:
        test_sentences = f.readlines()
  
  
  if "arpa_cleaners" in args.text_cleaners:
    print("ARPA cleaners detected, will run in dual mode")
    
  for filelist in args.filelists:
    print("START:", filelist)
    filepaths_and_text = load_filepaths_and_text(filelist)
    new_fp_text = []
    
    for i in tqdm(range(len(filepaths_and_text))):
      original_text = filepaths_and_text[i][args.text_index]
      original_fname = filepaths_and_text[i][0]
      
      moji_filename = original_fname.replace(".wav",".torchmoji")
      if not os.path.isfile(moji_filename):
        moji_result = moji(original_text)
        torch.save(moji_result, moji_filename)
        
        
      bert_filename = original_fname.replace(".wav",".bert")
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
  
  print("Writing test data...")
  test_path = "test_preproc"
  for i, test_sent in tqdm(enumerate(test_sentences)):
    moji_t = moji(test_sent)
    bert_t, _ = bert_f.infer(test_sent)
    
    test_cleaned_text = text._clean_text(test_sent, args.text_cleaners)
    cleaned_text_indices = text.cleaned_text_to_sequence(test_cleaned_text)
    text_norm = commons.intersperse(cleaned_text_indices, 0)
    text_norm = torch.LongTensor(text_norm)
    
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    test_rawfn = f"test{i}.pt"
    test_fullfn = os.path.join(test_path, test_rawfn)
    
    torch.save([text_norm, moji_t, bert_t], test_fullfn)
    
    
        
   
    
      
