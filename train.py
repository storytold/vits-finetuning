import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import commons
import utils
from data_utils import (
  TextAudioLoader,
  TextAudioCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
  MultiCoMBDiscriminator,
  MultiSubBandDiscriminator
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss,
  ForwardSumLoss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

import logging

torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True

global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '35000'

  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step

  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioCollate()
  train_loader = DataLoader(train_dataset, num_workers=hps.n_workers, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=hps.n_workers, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)

  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda(rank)
  
  mcmbd = MultiCoMBDiscriminator(hps.disc.combd_kernels, hps.disc.combd_channels, hps.disc.combd_groups, hps.disc.combd_strides).cuda(rank)
  msbd = MultiSubBandDiscriminator(hps.disc.tkernels, hps.disc.fkernel, hps.disc.tchannels, hps.disc.fchannels, hps.disc.tstrides, hps.disc.fstride,
                                     hps.disc.tdilations, hps.disc.fdilations, hps.disc.tsubband, hps.disc.pqmf_n, hps.disc.pqmf_m, hps.disc.freq_init_ch).cuda(rank)
  if hps.train.use_8bit:
    import bitsandbytes as bnb
    optim_g = bnb.optim.AdamW(
          net_g.parameters(), 
          hps.train.learning_rate, 
          betas=hps.train.betas, 
          eps=hps.train.eps)
    optim_d = bnb.optim.AdamW(
          itertools.chain(msbd.parameters(), mcmbd.parameters()),
          hps.train.learning_rate, 
          betas=hps.train.betas, 
          eps=hps.train.eps)
  else:
    optim_g = torch.optim.AdamW(
          net_g.parameters(), 
          hps.train.learning_rate, 
          betas=hps.train.betas, 
          eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
          itertools.chain(msbd.parameters(), mcmbd.parameters()),
          hps.train.learning_rate, 
          betas=hps.train.betas, 
          eps=hps.train.eps)
    
  
  net_g = DDP(net_g, device_ids=[rank],find_unused_parameters=True)
  mcmbd = DDP(mcmbd, device_ids=[rank],find_unused_parameters=True)
  msbd = DDP(msbd, device_ids=[rank])

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D0_*.pth"), mcmbd, optim_d)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D1_*.pth"), msbd, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0
  
  if len(hps.pt_path) > 1 and global_step == 0:
    if rank == 0:
        logger.info(f"Trying to load pretrained weights from {hps.pt_path}")
    _, _, _, s_ = utils.load_checkpoint(utils.latest_checkpoint_path(hps.pt_path, "G_*.pth"), net_g, None)
    _, _, _, s_ = utils.load_checkpoint(utils.latest_checkpoint_path(hps.pt_path, "D0_*.pth"), mcmbd, None)
    _, _, _, s_ = utils.load_checkpoint(utils.latest_checkpoint_path(hps.pt_path, "D1_*.pth"), msbd, None)
    

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  
  if n_gpus > 1 and hps.train.grad_acc_steps > 1:
    hps.train.grad_acc_steps = 1
    if rank == 0:
        logger.info("Gradient accumulation not available for Multi-GPU")
    
  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g,  mcmbd, msbd], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g,  mcmbd, msbd], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, mcmbd, msbd = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  forward_sum = ForwardSumLoss()
  global global_step


  net_g.train()
  mcmbd.train()
  msbd.train()
  n_acc_steps = hps.train.grad_acc_steps
  if rank == 0 and n_acc_steps > 1:
    logger.info(f"Running with {n_acc_steps}x gradient accumulation")
    
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, tm_hidden, bert, bert_lens) in tqdm(enumerate(train_loader)):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    tm_hidden = tm_hidden.cuda(rank, non_blocking=True)
    bert = bert.cuda(rank, non_blocking=True)
    bert_lens = bert_lens.cuda(rank, non_blocking=True)

    with autocast(enabled=hps.train.fp16_run):
      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
    
      
      y_hat, x1, l_length, attn, attn_logprob, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, mel, tm_hidden, bert, bert_lens)
      
      


      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )

      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

      # MSBD first
      y_d_hat_r, y_d_hat_g, _, _ =  msbd(y, y_hat.detach())
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        
      #MCMBD second
      
      y_d1_hat_r, y_d1_hat_g, _, _ = mcmbd(y, y_hat.detach(), x1.detach())
      with autocast(enabled=False):
        loss_disc_d1, losses_disc_d1_r, losses_disc_d1_g = discriminator_loss(y_d1_hat_r, y_d1_hat_g)
        loss_disc_all = loss_disc + loss_disc_d1
        loss_disc_all /= n_acc_steps
      
      
    if ((batch_idx + 1) % n_acc_steps == 0) or (batch_idx + 1 == len(train_loader)):
        optim_d.zero_grad()
        
    
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(msbd.parameters(), None) + commons.clip_grad_value_(mcmbd.parameters(), None)
    
    if ((batch_idx + 1) % n_acc_steps == 0) or (batch_idx + 1 == len(train_loader)):
        scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mcmbd(y, y_hat, x1)
      y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msbd(y, y_hat)
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float())
        ctc_loss = forward_sum(attn_logprob=attn_logprob, in_lens=x_lengths, out_lens=spec_lengths)
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl


        loss_fm_f = 2 * feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = 2 * feature_loss(fmap_s_r, fmap_s_g)
        loss_fm = loss_fm_f + loss_fm_s
        
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        
        loss_gen = loss_gen_f + loss_gen_s
        losses_gen = losses_gen_f + losses_gen_s
        
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + ctc_loss
        loss_gen_all /= n_acc_steps
        
    if ((batch_idx + 1) % n_acc_steps == 0) or (batch_idx + 1 == len(train_loader)):
        optim_g.zero_grad()
    
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    if ((batch_idx + 1) % n_acc_steps == 0) or (batch_idx + 1 == len(train_loader)):
        scaler.step(optim_g)
        
    scaler.update()


    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl, ctc_loss]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl, "loss/g/ctc" : ctc_loss})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)
        
      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        
        
      if global_step % hps.train.save_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        g_name, d0_name, d1_name = "G_{}.pth".format(global_step), "D0_{}.pth".format(global_step), "D1_{}.pth".format(global_step)
        if hps.use_latest:
          g_name, d0_name, d1_name = "G_latest.pth", "D0_latest.pth", "D1_latest.pth"

        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, g_name))
        utils.save_checkpoint(mcmbd, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, d0_name))
        utils.save_checkpoint(msbd, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, d1_name))
        
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, tm_hidden, bert, bert_lens) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)
        tm_hidden = tm_hidden.cuda(0)
        bert = bert.cuda(0)
        bert_lens = bert_lens.cuda(0)

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        tm_hidden = tm_hidden[:1]
        bert = bert[:1]
        bert_lens = bert_lens[:1]
        break
        
      image_dict = {}
      audio_dict = {}
      y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, tm_hidden, bert, bert_lens, max_len=1000)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
      image_dict["gen/mel"] = utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
      audio_dict["gen/audio"] = y_hat[0,:,:y_hat_lengths[0]]
      print("Inferring test sentences...")
      for t_idx, test_file in enumerate(sorted(os.listdir(hps.test_dp))):
        if not ".pt" in test_file:
            continue
        
        t_text_norm, t_moji, t_bert = torch.load(os.path.join(hps.test_dp, test_file))
        
        t_text_norm = t_text_norm.unsqueeze(0).cuda(0)
        t_text_lengths = torch.LongTensor([t_text_norm.size(1)]).cuda(0)
        
        t_moji = t_moji.squeeze().unsqueeze(0).cuda(0)
        t_bert_lens = torch.LongTensor([t_bert.size(1)]).cuda(0)
        t_bert = t_bert.cuda(0)
        
        with torch.no_grad():
            audio, attn, t_mask, _ = generator.module.infer(t_text_norm, t_text_lengths, t_moji, t_bert, t_bert_lens, noise_scale=.667, noise_scale_w=0.8, length_scale=1.0)
        
        test_audio_lengths = t_mask.sum([1,2]).long() * hps.data.hop_length
        y_test_mel = mel_spectrogram_torch(
            audio.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
        image_dict[f"gen/mel_test{t_idx}"] = utils.plot_spectrogram_to_numpy(y_test_mel[0].cpu().numpy())
        audio_dict[f"gen/audio_test{t_idx}"] = audio[0,:,:test_audio_lengths[0]]
        
        
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  main()
