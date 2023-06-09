﻿# VITS-Finetuning

This repo presents an efficiently finetuneable (on Colab) version of a heavily modified VITS model.

The model has had several enhancements:

1. [iSTFTNet generator + Avocodo discriminators](https://github.com/rishikksh20/iSTFT-Avocodo-pytorch/tree/faster).  Provides a 2.3x inference speed increase, and somewhat faster training, at the cost of a bit of audio quality
2. [Mixer-TTS Convolutional Attention module](https://nv-adlr.github.io/one-tts-alignment). More up to date technique for learning attention and removes the Cython compile step for the default MAS. 
3. [Mixer-TTS Self Attention module](https://github.com/NVIDIA/NeMo/blob/8685468036bebcc2eaea580d3ab5e140b7751b04/nemo/collections/tts/modules/mixer_tts.py#L185). Allows for injecting BERT context for more rich and expressive speech. We use TinyBERT.
4. TorchMoji embedding. Using hidden states from TorchMoji allows for some more explicit emotion control.
5. ARPA phonemizer with custom dictionary. 

But most importantly for enabling training on Google Colab, we implement single-GPU [gradient accumulation](https://kozodoi.me/blog/20210219/gradient-accumulation). Most common GPUs on Colab have just 16GB of VRAM, which is insufficient to fit a batch size of 32 natively (unless max sample length is very harshly limited). However, the model results in very hard to converge and choppy speech (think like they were words pieced together in a YTP) if the batch size is reduced.
Gradient accumulation allows us to "fake" higher batch size training, so we can use `batch_size=16` with 2x gradient accumulation and the model will turn out as if we used `batch_size=32`. 
Lastly, for some more VRAM savings, we use [8bit optimizer from bitsandbytes](https://github.com/TimDettmers/bitsandbytes). [(nivibilla et al)](https://github.com/nivibilla/efficient-vits-finetuning).

With all these modifications (and fp16 enabled), the training consumes roughly 11 - 14GB of VRAM.

## Scripts

### Training
The training script can take a `-p` argument, which will warm start from a pretrained model.
```sh
python train.py -c ./configs/ljs_li44_tmbert_ft_small_s1_arpa.json -m "train/MyVoice" -p pt_hanashi
```
Additionally, for reducing disk space consumption, you can add `--use-latest` which will save each time into a single `G_latest`, `D0_latest`, `D1_latest` instead of separate files every checkpoint interval. 


### Exporting to TorchScript
Device can be either cpu or cuda. They must match at inference time, cpu-exported model on CPU and CUDA-exported model on GPU.
```sh
python export_ts.py --checkpoint G_latest.pth --device cpu --config ./configs/ljs_li44_tmbert_ft_small_s1_arpa.json --out-path my_tsvits.pt --test-string "I like pizza"
```

### Inference with TorchScript
Device can be either cpu or cuda. Again, this must match the device you exported to TS with, cpu-exported model on CPU and CUDA-exported model on GPU.
```sh
python infer_ts.py --out-path res --device cpu --config ./configs/ljs_li44_tmbert_nmp_s1_arpa.json --checkpoint my_tsvits.pt --text "I like hamburgers"
```
Check `tsvitsfe.py` for the simple inference frontend class. The script above just calls it and saves to a wave file


# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

### Jaehyeon Kim, Jungil Kong, and Juhee Son

In our recent [paper](https://arxiv.org/abs/2106.06103), we propose VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.

Visit our [demo](https://jaywalnut310.github.io/vits-demo/index.html) for audio samples.

We also provide the [pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing).

** Update note: Thanks to [Rishikesh (ऋषिकेश)](https://github.com/jaywalnut310/vits/issues/1), our interactive TTS demo is now available on [Colab Notebook](https://colab.research.google.com/drive/1CO61pZizDj7en71NQG_aqqKdGaA_SaBf?usp=sharing).

<table style="width:100%">
  <tr>
    <th>VITS at training</th>
    <th>VITS at inference</th>
  </tr>
  <tr>
    <td><img src="resources/fig_1a.png" alt="VITS at training" height="400"></td>
    <td><img src="resources/fig_1b.png" alt="VITS at inference" height="400"></td>
  </tr>
</table>


## Pre-requisites
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```


## Training Exmaple
```sh
# LJ Speech
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK
python train_ms.py -c configs/vctk_base.json -m vctk_base
```
## Exporting to TorchScript
Device can be either cpu or cuda. They must match at inference time, cpu-exported model on CPU and CUDA-exported model on GPU.
```sh
python export_ts.py --checkpoint vits.pth --device cpu --config ./configs/ljs_li44_tmbert_nmp_s1.json --out-path my_tsvits.pt --test-string "I like pizza"
```

## Inference with TorchScript
Device can be either cpu or cuda. Again, this must match the device you exported to TS with, cpu-exported model on CPU and CUDA-exported model on GPU.
```sh
python infer_ts.py --out-path res --device cpu --config ./configs/ljs_li44_tmbert_nmp_s1.json --checkpoint my_tsvits.pt --text "I like hamburgers"
```
Check `tsvitsfe.py` for the simple inference frontend class. The script above just calls it and saves to a wave file

## Inference Example
See [inference.ipynb](inference.ipynb)
