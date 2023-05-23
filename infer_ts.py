import os
import json
import argparse

import torch
import numpy as np
import soundfile as sf
from tsvitsfe import TSVITSFE

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to TorchScript-exported model file.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="Path to store files",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="device. cuda or cpu",
    )
    args = parser.parse_args()

    os.makedirs(
        args.out_path, exist_ok=True)
    
    vits_fe = TSVITSFE()
    print("Loading model...")
    vits_fe.load(args.checkpoint,args.config,args.device)
    print("Doing inference...")
    out_aud = vits_fe.infer(args.text)
    
    out_fn = os.path.join(args.out_path,"audio1.wav")
    sf.write(out_fn, out_aud, vits_fe.hps.data.sampling_rate)
    print(f"Wrote audio file {out_fn}")
    
    
    
    
    
    
    
    
    
    
    


