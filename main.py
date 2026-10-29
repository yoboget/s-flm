import gc
import glob
import json
import os
import time

import fsspec
import hydra
import lightning as L
import matplotlib.pyplot as plt
from lightning.fabric import Fabric
import numpy as np
import omegaconf
from pathlib import Path
import rich.syntax
import rich.tree
import torch
from tqdm import tqdm, trange

import algo
from callbacks.throughput import ThroughputCallback
import dataloader
import samplers
import sandbox_gsm8k
import utils


omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  # if 'hf' in config.algo.backbone:
  #   return diffusion_model(
  #     config, tokenizer=tokenizer).to('cuda')
  
  return diffusion_model.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    strict=config.eval.strict_loading)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    try:
      print(f'First {k} tokens:', tokenizer.decode(first))
      print('ids:', first)
      print(f'Last {k} tokens:', tokenizer.decode(last))
      print('ids:', last)
    except:
      print('First tokens:', first)
      print('Last tokens:', last)


def _token_entropy(tokens, ignore_index=None):
  """Entropy of the token histogram of each sequence (as in
  metrics.record_entropy), averaged over the batch. Tokens equal to
  ignore_index (e.g. the mask) are left out."""
  entropies = []
  for sample in tokens:
    if ignore_index is not None:
      sample = sample[sample != ignore_index]
    if sample.numel() == 0:
      entropies.append(0.0)
      continue
    _, counts = torch.unique(sample, return_counts=True)
    entropies.append(torch.special.entr(
      counts.float() / counts.sum()).sum().item())
  return sum(entropies) / len(entropies)


class _EntropyTrajectory:
  """eval.entropy_every: sampler callback that records, every
  `every` fraction of the sampling steps (and at the prior), the
  entropy of the current state xt and of x0 ~ p(x0 | xt), the
  independent draw from the denoiser at the step that produced xt.
  For MDLM, xt_entropy is over the unmasked tokens only; for UNSID,
  xt is the argmax of the simplex point."""

  def __init__(self, model, config):
    if config.sampler.predictor not in samplers.ANCESTRAL_PREDICTORS:
      raise ValueError('eval.entropy_every needs sampler in '
                       f'{samplers.ANCESTRAL_PREDICTORS}, '
                       f'got {config.sampler.predictor}')
    self.every = config.eval.entropy_every
    self.mask_index = (model.mask_index
                       if model.diffusion_type == 'absorbing'
                       else None)
    self.records = {}  # step -> list of per-batch records

  def __call__(self, state):
    num_steps = len(state.timesteps) - 1
    interval = max(1, round(self.every * num_steps))
    step = state.step_idx
    if step % interval != 0 and step != num_steps:
      return
    xt = state.xt
    if xt.ndim == 3:
      xt = xt.argmax(-1)
    record = dict(
      xt_entropy=_token_entropy(xt, self.mask_index),
      x0_entropy=(None if step == 0
                  else _token_entropy(state.x0_pred)))
    if self.mask_index is not None:
      record['mask_frac'] = (xt == self.mask_index).float().mean().item()
    self.records.setdefault(step, []).append(record)
    self._num_steps = num_steps
    self._timesteps = state.timesteps

  def summary(self):
    """One entry per recorded step, averaged over the batches. t is
    the time of the state xt after the step."""
    out = []
    for step in sorted(self.records):
      recs = self.records[step]
      entry = dict(step=step, frac=step / self._num_steps,
                   t=(0.0 if step == self._num_steps
                      else self._timesteps[step].item()))
      for key in recs[0]:
        values = [r[key] for r in recs if r[key] is not None]
        entry[key] = sum(values) / len(values) if values else None
      out.append(entry)
    return out


@torch.no_grad
def _generate_samples(diffusion_model, config, logger,
                      tokenizer):
  logger.info('Starting Sample Eval.')
  fabric = Fabric(accelerator=config.trainer.accelerator, 
                  devices=config.trainer.devices, 
                  num_nodes=config.trainer.num_nodes)
  fabric.launch()
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  model.metrics.gen_ppl.reset()
  model.metrics.sample_entropy.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  num_devices = config.trainer.num_nodes * config.trainer.devices
  if config.sampler.num_sample_batches % num_devices != 0:
    raise ValueError(
      f'Num. batches ({config.sampler.num_sample_batches}) '
      f'not divided by num. devices ({num_devices})')

  num_batch_per_device = (config.sampler.num_sample_batches
                          // num_devices)
  total_num_samples = (config.sampler.num_sample_batches
                      * config.loader.eval_batch_size)
  samples_fname = utils.vars_to_fname(
    predictor=config.sampler.predictor,
    noise_removal=config.sampler.noise_removal,
    use_float64=config.sampler.use_float64,
    p_nucleus=config.sampler.p_nucleus,
    num_samples=total_num_samples,
    num_sampling_steps=config.sampler.steps,
    ckpt_hash=utils.short_hash(config.eval.checkpoint_path)
  ) + ".json"

  entropy_trajectory = None
  if config.eval.get('entropy_every') is not None:
    entropy_trajectory = _EntropyTrajectory(model, config)

  model.to(fabric.device)
  if fabric.global_rank == 0:
    logger.info(f'Sampling {total_num_samples} samples.')
  # eval.also_greedy: the last step is also decoded with argmax, from
  # the same state, scored and saved apart (<results>_greedy.json)
  also_greedy = config.eval.get('also_greedy', False)
  all_text_samples = []
  all_np_samples = []
  all_greedy_samples = []
  total_nfe = 0
  for _ in trange(num_batch_per_device, desc=f'Sampling '
                  f'({config.sampler.steps} steps)',
                  disable=fabric.global_rank != 0):
    samples, metadata = model.restore_model_and_sample(
      num_steps=config.sampler.steps, on_step=entropy_trajectory,
      also_greedy=also_greedy)
    total_nfe += metadata['nfe']
    if also_greedy:
      greedy_samples = fabric.all_gather(metadata['xt_greedy'])
      if fabric.world_size > 1:
        greedy_samples = greedy_samples.flatten(0, 1)
      all_greedy_samples.append(greedy_samples.cpu())
    samples = fabric.all_gather(samples)
    if fabric.world_size > 1:
      samples = samples.flatten(0, 1)

    np_samples = samples.cpu().numpy()
    text_samples = model.tokenizer.batch_decode(samples)
    model.metrics.record_entropy(samples)
    all_text_samples.extend(text_samples)
    all_np_samples.append(np_samples)

  fabric.barrier()
  if fabric.global_rank == 0:
    logger.info("Evaluating generative perplexity...")
    all_np_samples = np.concatenate(all_np_samples)
    # Evaluate with retokenize and first chunk only (orig)
    model.metrics.record_generative_perplexity(
      all_text_samples, 
      config.model.length,
      retokenize=True,
      first_chunk_only=True,
      device=model.device)
    gen_ppl_first_chunk_retok = model.metrics.gen_ppl.compute().item()
    model.metrics.gen_ppl.reset()
    gen_ppl_all_no_retok = None
    entropy = model.metrics.sample_entropy.compute().item()

    avg_nfe = total_nfe / max(num_batch_per_device, 1)
    logger.info('Generative perplexity (retokenize, first '
               f'chunk only): {gen_ppl_first_chunk_retok:.5f}')
    logger.info(f'Sample entropy: {entropy:.5f}')
    logger.info(f'Average NFE per batch: {avg_nfe:.1f}')
    if config.eval.results_json_path is not None:
      samples_path = config.eval.results_json_path
    else:
      samples_path = os.path.join(os.getcwd(), 'samples',
                                  samples_fname)
    os.makedirs(os.path.dirname(samples_path), exist_ok=True)

    save_dict = dict(
      text=all_text_samples,
      np_tokens_b64=utils.np_to_base64(all_np_samples),
      gen_ppl_first_chunk_retok=gen_ppl_first_chunk_retok,
      gen_ppl_all_no_retok=gen_ppl_all_no_retok,
      entropy=entropy,
      ckpt_name=config.eval.checkpoint_path,
      config=omegaconf.OmegaConf.to_container(config,
                                              resolve=True),
      avg_nfe=avg_nfe)
    if entropy_trajectory is not None:
      # Rank 0 batches only (when sampling on several devices)
      save_dict['entropy_trajectory'] = entropy_trajectory.summary()
      for e in save_dict['entropy_trajectory']:
        logger.info(
          f"Entropy at step {e['step']} (t={e['t']:.3f}): "
          f"xt {e['xt_entropy']:.3f}, x0 "
          + ('-' if e['x0_entropy'] is None
             else f"{e['x0_entropy']:.3f}"))
    with fsspec.open(samples_path, 'w') as f:
      json.dump(save_dict, f, indent=4)
    print('Samples saved at:', samples_path)

    if also_greedy:
      greedy_tokens = torch.cat(all_greedy_samples)
      greedy_text = model.tokenizer.batch_decode(greedy_tokens)
      model.metrics.sample_entropy.reset()
      model.metrics.record_entropy(greedy_tokens)
      model.metrics.record_generative_perplexity(
        greedy_text,
        config.model.length,
        retokenize=True,
        first_chunk_only=True,
        device=model.device)
      greedy_gen_ppl = model.metrics.gen_ppl.compute().item()
      model.metrics.gen_ppl.reset()
      greedy_entropy = model.metrics.sample_entropy.compute().item()
      logger.info('Greedy last step: generative perplexity (retokenize, '
                  f'first chunk only): {greedy_gen_ppl:.5f}')
      logger.info(f'Greedy last step: sample entropy: {greedy_entropy:.5f}')
      greedy_dict = dict(save_dict)
      greedy_dict.update(
        text=greedy_text,
        np_tokens_b64=utils.np_to_base64(greedy_tokens.numpy()),
        gen_ppl_first_chunk_retok=greedy_gen_ppl,
        entropy=greedy_entropy,
        noise_removal='greedy',
        # Fraction of the tokens that differ from the sampled last step
        frac_tokens_changed=(greedy_tokens.numpy()
                             != all_np_samples).mean().item())
      greedy_path = os.path.splitext(samples_path)[0] + '_greedy.json'
      with fsspec.open(greedy_path, 'w') as f:
        json.dump(greedy_dict, f, indent=4)
      print('Greedy samples saved at:', greedy_path)
  fabric.barrier()


def _eval_ppl(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Perplexity Eval.')

  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  results = trainer.validate(model, valid_ds)
  if config.eval.results_json_path is not None:
    save_path = Path(config.eval.results_json_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
      json.dump(results[0], f)
      print(f'Saved results to `{save_path}`')


def _train(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      **config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None):
    ckpt_path = config.checkpointing.resume_ckpt_path
    # Optionally load checkpoint last-v1 etc. Pick the one with highest global step
    if ckpt_path.endswith('last.ckpt'):
      ckpt_path = ckpt_path.replace('last.ckpt', 'last*')
      matching_ckpts = glob.glob(ckpt_path)
      if len(matching_ckpts) == 0:
        ckpt_path = None
      elif len(matching_ckpts) == 1:
        ckpt_path = matching_ckpts[0]
      else:
        # If there are multiple last.ckpt checkpoints, pick 
        #  the one with latest global_step.
        latest_path = None
        latest_step = 0
        for path in matching_ckpts:
          ckpt = torch.load(path, map_location='cpu', weights_only=False)
          if ckpt['global_step'] > latest_step:
            latest_step = ckpt['global_step']
            latest_path = path
        ckpt_path = latest_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  callbacks.append(ThroughputCallback(
    log_every_n_batches=config.trainer.log_every_n_steps))

  # Ensure dataset processing happens on rank 0 first
  fabric = L.Fabric(num_nodes=config.trainer.num_nodes, 
                  devices=config.trainer.devices, 
                  accelerator='cuda')
  fabric.launch()
  with fabric.rank_zero_first():
    train_ds, valid_ds = dataloader.get_dataloaders(
      config, tokenizer)

  _print_batch(train_ds, valid_ds, tokenizer)
  fabric.barrier()
  del fabric

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    model = diffusion_model.load_from_checkpoint(
      config.training.finetune_path,
      tokenizer=tokenizer,
      config=config)
    # Start the data from the beginning, not from the position of the
    # pretraining run (on_load_checkpoint)
    model.fast_forward_epochs = None
    model.fast_forward_batches = None
  else:
    model = diffusion_model(config, 
                            tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


def _load_generated_gsm8k(config, logger):
  logger.info(f'Loading GSM8K predictions from {config.gsm8k.input_file}')
  with open(config.gsm8k.input_file) as f:
    loaded = json.load(f)
  # Accept either a bare list of records or a previously-saved results.json
  # (which wraps the list under 'records').
  if isinstance(loaded, dict) and 'records' in loaded:
    return loaded['records']
  return loaded


def _pad_prefix_batch(id_lists, device):
  """Pad a list of 1D tensors into a batch.

  Returns:
    padded: (B, max_len) int64 tensor, zero-padded.
    lengths: (B,) int64 tensor of actual lengths.
  """
  lengths = torch.tensor(
    [len(ids) for ids in id_lists], device=device)
  max_len = int(lengths.max())
  padded = torch.zeros(
    len(id_lists), max_len, dtype=torch.long, device=device)
  for i, ids in enumerate(id_lists):
    padded[i, :len(ids)] = ids.to(device)
  return padded, lengths


def _sample_gsm8k(diffusion_model, config, logger, tokenizer, fabric):
  model = _load_from_checkpoint(diffusion_model, config, tokenizer)
  model.to(fabric.device)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  model._eval_mode()

  # Load dataset on rank 0 first, then broadcast to others
  with fabric.rank_zero_first():
    dataset = dataloader.get_dataset(config, tokenizer, mode='valid')

  # Interleaved split: rank r gets indices r, r+W, r+2W, ...
  rank = fabric.global_rank
  world_size = fabric.world_size
  local_indices = list(range(rank, len(dataset), world_size))
  batch_size = config.loader.eval_batch_size

  os.makedirs(config.gsm8k.output_dir, exist_ok=True)
  rank_json_path = os.path.join(config.gsm8k.output_dir,
                                f'rank_{rank:04d}.json')
  records = []
  total_nfe = 0
  num_batches = 0
  for batch_start in trange(
      0, len(local_indices), batch_size,
      desc=f'Rank {rank} sampling',
      disable=rank != 0):
    batch_indices = local_indices[
      batch_start:batch_start + batch_size]
    B = len(batch_indices)

    all_ids = [torch.tensor(dataset[idx]['input_ids'])
               for idx in batch_indices]
    padded, lengths = _pad_prefix_batch(
      all_ids, fabric.device)

    samples, metadata = model.generate_samples(
      num_samples=B,
      num_steps=config.sampler.steps,
      prefix_tokens=padded,
      prefix_lengths=lengths)
    total_nfe += metadata['nfe']
    num_batches += 1

    for i, idx in enumerate(batch_indices):
      prompt_len = int(lengths[i])
      generated_ids = samples[i, prompt_len:].cpu().tolist()

      response = tokenizer.decode(
        generated_ids, skip_special_tokens=True)
      records.append({
        'prompt': dataset[idx]['prompt'],
        'response_ground_truth':
          dataset[idx]['response_ground_truth'],
        'response': response,
      })
    # Cleanup memory for the next sampling batch
    del samples, metadata
    gc.collect()

  avg_nfe = total_nfe / max(num_batches, 1)
  logger.info(f'Rank {rank}: average NFE per batch: {avg_nfe:.1f}')

  with open(rank_json_path, 'w') as f:
    json.dump(records, f, indent=2)
  logger.info(f'Rank {rank}: saved {len(records)} records to {rank_json_path}')

  fabric.barrier()

  if fabric.global_rank != 0:
    return None

  all_records = []
  for r in range(world_size):
    path = os.path.join(config.gsm8k.output_dir, f'rank_{r:04d}.json')
    with open(path) as f:
      all_records.extend(json.load(f))
  return all_records


@torch.no_grad()
def _gsm8k_eval(diffusion_model, config, logger, tokenizer):
  logger.info('Starting GSM8K eval.')
  fabric = Fabric(
    accelerator=config.trainer.accelerator,
    devices=config.trainer.devices,
    num_nodes=config.trainer.num_nodes)
  fabric.launch()

  seed = config.seed + fabric.global_rank
  L.seed_everything(seed)

  if config.gsm8k.input_file is not None:
    if fabric.global_rank == 0:
      samples = _load_generated_gsm8k(config, logger)
    else:
      samples = None
  else:
    samples = _sample_gsm8k(
      diffusion_model, config, logger, tokenizer, fabric)
  if fabric.global_rank == 0:
    logger.info(f'Total records evaluated: {len(samples)}')
    per_sample_correct = np.array([
      int(sandbox_gsm8k.evaluate_samples(
        rec['response'],
        rec['response_ground_truth'],
        timeout_s=config.gsm8k.timeout))
      for rec in samples], dtype=np.int32)
    correct = int(per_sample_correct.sum())
    # Optionally: bootstrapped 95% CI (test-set bootstrap, percentile method).
    # When bootstrap_size > 1 the reported `accuracy` is the bootstrap mean
    # (per Raschka's recommendation) with 95% CI from the 2.5/97.5 percentiles
    # of the bootstrap distribution. `accuracy_se` is the std of the bootstrap
    # means (= bootstrap-estimated standard error of the accuracy).
    if config.gsm8k.bootstrap_size > 1:
      rng = np.random.default_rng(config.seed)
      idx = rng.integers(0, len(samples),
                         size=(config.gsm8k.bootstrap_size, len(samples)))
      boot_acc = per_sample_correct[idx].mean(axis=1)
      acc = float(boot_acc.mean())
      acc_se = float(boot_acc.std(ddof=1))
      acc_ci_lo = float(np.percentile(boot_acc, 2.5))
      acc_ci_hi = float(np.percentile(boot_acc, 97.5))
      logger.info(
        f'GSM8K accuracy: {acc * 100:.2f}% '
        f'[{acc_ci_lo * 100:.2f}%, {acc_ci_hi * 100:.2f}%] (95% CI) '
        f'({correct}/{len(samples)}, bootstrap N={config.gsm8k.bootstrap_size})')
    else:
      acc = correct / len(samples)
      acc_se = 0.0
      acc_ci_lo = acc
      acc_ci_hi = acc
      logger.info(f'GSM8K accuracy: {acc * 100:.2f}% ({correct}/{len(samples)})')

    os.makedirs(config.gsm8k.output_dir, exist_ok=True)
    merged_path = os.path.join(config.gsm8k.output_dir, 'results.json')
    results = {'accuracy': acc,
               'accuracy_se': acc_se,
               'accuracy_ci_lo': acc_ci_lo,
               'accuracy_ci_hi': acc_ci_hi,
               'num_correct': correct,
               'num_total': len(samples),
               'records': samples}
    if config.gsm8k.input_file is not None:
      results['input_file'] = config.gsm8k.input_file
    with open(merged_path, 'w') as f:
      json.dump(results, f, indent=2)
    logger.info(f'Results saved to {merged_path}')

  fabric.barrier()


@torch.no_grad()
def _sudoku_eval(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Sudoku eval.')
  fabric = Fabric(
    accelerator=config.trainer.accelerator,
    devices=config.trainer.devices,
    num_nodes=config.trainer.num_nodes)
  fabric.launch()

  seed = config.seed + fabric.global_rank
  L.seed_everything(seed)

  model = _load_from_checkpoint(diffusion_model, config, tokenizer)
  model.to(fabric.device)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  model._eval_mode()

  unconditional = config.data.get('unconditional', False)
  if unconditional and model.num_tokens != tokenizer.uncond_seq_len:
    raise ValueError(
      'data.unconditional=True needs model.length='
      f'{tokenizer.uncond_seq_len}, got {model.num_tokens}')

  with fabric.rank_zero_first():
    # Unconditional: the training grids, to measure memorization
    dataset = dataloader.get_dataset(
      config, tokenizer,
      mode='train' if unconditional else 'valid')
  if unconditional:
    train_grids = {tuple(ids.tolist()) for ids in dataset['input_ids']}

  # Evaluate every number of sampling steps in sudoku.steps_list
  # (default: sampler.steps), each in its own subdirectory.
  steps_list = config.sudoku.get('steps_list', None)
  if steps_list is None:
    steps_list = [config.sampler.steps]
  summary = {}
  for num_steps in steps_list:
    # Same seed for every number of steps, so that they are comparable
    L.seed_everything(seed)
    output_dir = os.path.join(
      config.sudoku.output_dir, f'steps_{num_steps}')
    if unconditional:
      results = _sudoku_uncond_eval_steps(
        model, train_grids, tokenizer, fabric, config, logger,
        num_steps, output_dir)
    else:
      results = _sudoku_eval_steps(
        model, dataset, tokenizer, fabric, config, logger,
        num_steps, output_dir)
    if results is not None:
      summary[num_steps] = _without_records(results['sample'])
      if 'greedy' in results:
        # eval.also_greedy: same runs, last step decoded with argmax
        summary[num_steps]['greedy'] = _without_records(
          results['greedy'])

  if fabric.global_rank == 0:
    summary_path = os.path.join(
      config.sudoku.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
      json.dump(summary, f, indent=2)
    if unconditional:
      logger.info('Unconditional sudoku summary (steps: valid grids '
                  '/ valid units / entropy):')
      for num_steps, r in summary.items():
        for name, rr in [('', r)] + (
            [(' greedy', r['greedy'])] if 'greedy' in r else []):
          logger.info(
            f'  {num_steps:>5}{name}: {rr["valid_rate"] * 100:6.2f}% / '
            f'{rr["valid_unit_rate"] * 100:6.2f}% / '
            f'{rr["entropy"]:.3f}')
      logger.info(f'Summary saved to {summary_path}')
      return
    logger.info('Sudoku summary (steps: accuracy / cell / blank cell):')
    for num_steps, r in summary.items():
      for name, rr in [('', r)] + (
          [(' greedy', r['greedy'])] if 'greedy' in r else []):
        logger.info(
          f'  {num_steps:>5}{name}: {rr["accuracy"] * 100:6.2f}% / '
          f'{rr["cell_accuracy"] * 100:6.2f}% / '
          f'{rr["blank_cell_accuracy"] * 100:6.2f}%')
    logger.info(f'Summary saved to {summary_path}')


def _without_records(results):
  return {k: v for k, v in results.items() if k != 'records'}


def _greedy_suffix(variant):
  """File name suffix of the results of a last step variant."""
  return '' if variant == 'sample' else f'_{variant}'


def _merge_rank_files(output_dir, world_size, variant):
  """Loads the per-rank results of a variant, in rank order."""
  merged = []
  for r in range(world_size):
    path = os.path.join(
      output_dir, f'rank_{r:04d}{_greedy_suffix(variant)}.json')
    with open(path) as f:
      merged.append(json.load(f))
  return merged


def _frac_tokens_changed(variant_samples):
  """eval.also_greedy: fraction of the generated tokens that the
  greedy last step changes, compared to the sampled one."""
  if 'greedy' not in variant_samples:
    return None
  sample = torch.cat(variant_samples['sample'])
  greedy = torch.cat(variant_samples['greedy'])
  return (sample != greedy).float().mean().item()


def _sudoku_eval_steps(model, dataset, tokenizer, fabric, config,
                       logger, num_steps, output_dir):
  """Solves the validation puzzles with `num_steps` sampling steps.

  Reports the accuracy (fully solved grids), the cell accuracy over
  the 81 cells and the blank cell accuracy over the cells left empty
  in the puzzle (the ones the model has to find).
  Returns the merged results on rank 0, None on the other ranks.
  """
  prompt_len = tokenizer.prompt_len
  rank = fabric.global_rank
  world_size = fabric.world_size
  local_indices = list(range(rank, len(dataset), world_size))
  batch_size = config.sudoku.batch_size
  # eval.also_greedy: the last step is also decoded with argmax, from
  # the same state, and scored apart (results_greedy.json)
  variants = (['sample', 'greedy'] if config.eval.get('also_greedy')
              else ['sample'])
  records = {v: [] for v in variants}
  generated_tokens = {v: [] for v in variants}

  for batch_start in trange(
      0, len(local_indices), batch_size,
      desc=f'Sudoku eval ({num_steps} steps)', disable=rank != 0):
    batch_indices = local_indices[
      batch_start:batch_start + batch_size]
    B = len(batch_indices)

    # All sudoku examples have the same length, no padding needed
    batch = torch.stack(
      [dataset[idx]['input_ids'] for idx in batch_indices])
    prefix = batch[:, :prompt_len].to(fabric.device)
    prefix_lengths = torch.full(
      (B,), prompt_len, device=fabric.device)

    samples, metadata = model.generate_samples(
      num_samples=B,
      num_steps=num_steps,
      prefix_tokens=prefix,
      prefix_lengths=prefix_lengths,
      also_greedy='greedy' in variants)
    variant_samples = {'sample': samples}
    if 'greedy' in variants:
      variant_samples['greedy'] = metadata['xt_greedy']

    gt = batch[:, prompt_len:]
    # The puzzle (after the first BOS) is aligned with the solution
    puzzle = batch[:, 1:1 + tokenizer.seq_len]
    is_cell = gt != tokenizer.row_separator_id
    is_blank = is_cell & (puzzle == tokenizer.empty_token_id)
    for variant, variant_tokens in variant_samples.items():
      generated = variant_tokens[:, prompt_len:].cpu()
      generated_tokens[variant].append(generated)
      correct = (generated == gt).all(dim=1)
      cell_correct = (generated == gt) & is_cell
      blank_correct = (generated == gt) & is_blank

      for i, idx in enumerate(batch_indices):
        records[variant].append({
          'generated': tokenizer.decode(generated[i]),
          'ground_truth': tokenizer.decode(gt[i]),
          'correct': correct[i].item(),
          'num_cells_correct': cell_correct[i].sum().item(),
          'num_blank_cells': is_blank[i].sum().item(),
          'num_blank_cells_correct': blank_correct[i].sum().item(),
        })

    del samples, metadata, variant_samples
    gc.collect()

  # Save per-rank results and aggregate on rank 0
  os.makedirs(output_dir, exist_ok=True)
  frac_changed = _frac_tokens_changed(generated_tokens)
  for variant in variants:
    rank_path = os.path.join(
      output_dir, f'rank_{rank:04d}{_greedy_suffix(variant)}.json')
    with open(rank_path, 'w') as f:
      json.dump({'records': records[variant],
                 'frac_tokens_changed': frac_changed}, f, indent=2)

  fabric.barrier()

  results = None
  if fabric.global_rank == 0:
    results = {}
    for variant in variants:
      rank_results = _merge_rank_files(output_dir, world_size, variant)
      all_records = [rec for r in rank_results for rec in r['records']]
      num_correct = sum(r['correct'] for r in all_records)
      total = len(all_records)
      cell_accuracy = (sum(r['num_cells_correct'] for r in all_records)
                       / (total * tokenizer.num_cells))
      blank_cell_accuracy = (
        sum(r['num_blank_cells_correct'] for r in all_records)
        / max(sum(r['num_blank_cells'] for r in all_records), 1))
      name = '' if variant == 'sample' else ', greedy last step'
      logger.info(
        f'Sudoku accuracy ({num_steps} steps{name}): '
        f'{num_correct}/{total} '
        f'({num_correct / total * 100:.2f}%), cell accuracy '
        f'{cell_accuracy * 100:.2f}%, blank cell accuracy '
        f'{blank_cell_accuracy * 100:.2f}%')
      merged_path = os.path.join(
        output_dir, f'results{_greedy_suffix(variant)}.json')
      results[variant] = {
        'num_steps': num_steps,
        'last_step': variant,
        'accuracy': num_correct / total,
        'cell_accuracy': cell_accuracy,
        'blank_cell_accuracy': blank_cell_accuracy,
        'num_correct': num_correct,
        'num_total': total,
      }
      if variant == 'greedy':
        # Mean over the ranks (same number of samples each, up to one)
        results[variant]['frac_tokens_changed'] = sum(
          r['frac_tokens_changed'] for r in rank_results) / world_size
      results[variant]['records'] = all_records
      with open(merged_path, 'w') as f:
        json.dump(results[variant], f, indent=2)
      logger.info(f'Results saved to {merged_path}')

  fabric.barrier()
  return results


def _check_sudoku_grid(ids, tokenizer):
  """Checks a generated [BOS] grid(89) sequence. Returns well_formed
  (BOS and row separators in place, a digit in every cell), valid
  (well formed and every row, column and box holds the 9 digits),
  num_valid_units (out of the 27 rows, columns and boxes) and
  num_conflicts (sum over the units of 9 - number of distinct
  digits, 0 for a valid grid)."""
  grid = ids[1:]
  rows = [grid[10 * i:10 * i + 9] for i in range(9)]
  separators = [grid[10 * i + 9] for i in range(8)]
  layout_ok = (ids[0] == tokenizer.bos_token_id
               and all(s == tokenizer.row_separator_id
                       for s in separators))
  digits_ok = all(1 <= c <= 9 for row in rows for c in row)
  columns = [[row[c] for row in rows] for c in range(9)]
  boxes = [[rows[r + i][c + j] for i in range(3) for j in range(3)]
           for r in (0, 3, 6) for c in (0, 3, 6)]
  distinct = [len({c for c in unit if 1 <= c <= 9})
              for unit in rows + columns + boxes]
  num_valid_units = sum(d == 9 for d in distinct)
  well_formed = layout_ok and digits_ok
  return {
    'well_formed': well_formed,
    'valid': well_formed and num_valid_units == 27,
    'num_valid_units': num_valid_units,
    'num_conflicts': sum(9 - d for d in distinct),
  }


def _sudoku_uncond_eval_steps(model, train_grids, tokenizer, fabric,
                              config, logger, num_steps, output_dir):
  """data.unconditional=True: generates sudoku.num_samples complete
  grids from scratch with `num_steps` sampling steps.

  Reports the rate of valid grids, of valid units (rows, columns and
  boxes holding the 9 digits), the mean number of conflicts, the
  number of distinct valid grids, the rate of valid grids copied from
  the training set, and the token entropy of the samples, to compare
  with reference_entropy (the same for every valid grid). With
  eval.entropy_every, also the entropy trajectory during sampling.
  Returns the merged results on rank 0, None on the other ranks.
  """
  rank = fabric.global_rank
  world_size = fabric.world_size
  local_num = len(range(rank, config.sudoku.num_samples, world_size))
  batch_size = config.sudoku.batch_size
  trajectory = None
  if config.eval.get('entropy_every') is not None:
    trajectory = _EntropyTrajectory(model, config)
  # eval.also_greedy: the last step is also decoded with argmax, from
  # the same state, and scored apart (results_greedy.json)
  variants = (['sample', 'greedy'] if config.eval.get('also_greedy')
              else ['sample'])
  records = {v: [] for v in variants}
  generated_tokens = {v: [] for v in variants}

  for batch_start in trange(
      0, local_num, batch_size,
      desc=f'Unconditional sudoku ({num_steps} steps)',
      disable=rank != 0):
    B = min(batch_size, local_num - batch_start)
    samples, metadata = model.generate_samples(
      num_samples=B, num_steps=num_steps, on_step=trajectory,
      also_greedy='greedy' in variants)
    variant_samples = {'sample': samples}
    if 'greedy' in variants:
      variant_samples['greedy'] = metadata['xt_greedy']
    for variant, variant_tokens in variant_samples.items():
      variant_tokens = variant_tokens.cpu()
      generated_tokens[variant].append(variant_tokens)
      for i in range(B):
        ids = variant_tokens[i].tolist()
        records[variant].append({
          'generated': tokenizer.decode(ids[1:]),
          'entropy': _token_entropy(variant_tokens[i:i + 1]),
          'in_train': tuple(ids) in train_grids,
          **_check_sudoku_grid(ids, tokenizer),
        })
    del samples, metadata, variant_samples
    gc.collect()

  # Save per-rank results and aggregate on rank 0
  os.makedirs(output_dir, exist_ok=True)
  frac_changed = _frac_tokens_changed(generated_tokens)
  for variant in variants:
    rank_path = os.path.join(
      output_dir, f'rank_{rank:04d}{_greedy_suffix(variant)}.json')
    with open(rank_path, 'w') as f:
      # The entropy trajectory is the one of the shared steps
      json.dump({'records': records[variant],
                 'frac_tokens_changed': frac_changed,
                 'entropy_trajectory': (None if trajectory is None
                                        else trajectory.summary())},
                f, indent=2)

  fabric.barrier()

  results = None
  if rank == 0:
    results = {}
    for variant in variants:
      results[variant] = _sudoku_uncond_merge(
        _merge_rank_files(output_dir, world_size, variant), variant,
        train_grids, num_steps, output_dir, logger)

  fabric.barrier()
  return results


def _sudoku_uncond_merge(rank_results, variant, train_grids,
                         num_steps, output_dir, logger):
  """Aggregates the per-rank results of _sudoku_uncond_eval_steps for
  one last step variant (sample / greedy) and saves them."""
  all_records = [rec for r in rank_results for rec in r['records']]
  trajectories = [r['entropy_trajectory'] for r in rank_results
                  if r['entropy_trajectory'] is not None]
  total = len(all_records)
  valid = [r for r in all_records if r['valid']]
  results = {
    'num_steps': num_steps,
    'last_step': variant,
    'num_samples': total,
    'valid_rate': len(valid) / total,
    'well_formed_rate': sum(
      r['well_formed'] for r in all_records) / total,
    'valid_unit_rate': sum(
      r['num_valid_units'] for r in all_records) / (27 * total),
    'mean_conflicts': sum(
      r['num_conflicts'] for r in all_records) / total,
    'num_distinct_valid': len({r['generated'] for r in valid}),
    'memorized_rate': (sum(r['in_train'] for r in valid)
                       / max(len(valid), 1)),
    'entropy': sum(r['entropy'] for r in all_records) / total,
    'reference_entropy': _token_entropy(
      torch.tensor([next(iter(train_grids))])),
  }
  if variant == 'greedy':
    results['frac_tokens_changed'] = sum(
      r['frac_tokens_changed'] for r in rank_results) / len(rank_results)
  if trajectories:
    # Average over the ranks, entry by entry (same steps)
    results['entropy_trajectory'] = [
      {k: (v if k in ('step', 'frac', 't') or v is None
           else sum(t[j][k] for t in trajectories) / len(trajectories))
       for k, v in entry.items()}
      for j, entry in enumerate(trajectories[0])]
  name = '' if variant == 'sample' else ', greedy last step'
  logger.info(
    f'Unconditional sudoku ({num_steps} steps{name}): '
    f'{len(valid)}/{total} valid grids '
    f'({results["valid_rate"] * 100:.2f}%), valid units '
    f'{results["valid_unit_rate"] * 100:.2f}%, mean conflicts '
    f'{results["mean_conflicts"]:.2f}, distinct valid '
    f'{results["num_distinct_valid"]}, memorized '
    f'{results["memorized_rate"] * 100:.2f}%, entropy '
    f'{results["entropy"]:.3f} (valid grid: '
    f'{results["reference_entropy"]:.3f})')
  for e in results.get('entropy_trajectory', []):
    logger.info(
      f"  entropy at step {e['step']} (t={e['t']:.3f}): "
      f"xt {e['xt_entropy']:.3f}, x0 "
      + ('-' if e['x0_entropy'] is None
         else f"{e['x0_entropy']:.3f}"))
  results['records'] = all_records
  merged_path = os.path.join(
    output_dir, f'results{_greedy_suffix(variant)}.json')
  with open(merged_path, 'w') as f:
    json.dump(results, f, indent=2)
  logger.info(f'Results saved to {merged_path}')
  return results


class FakeTokenizer:
  def __init__(self, vocab_length):
    self.vocab_size = vocab_length
    self.bos_token_id = 0
    self.mask_id = vocab_length - 1
  
  def __len__(self):
    return self.vocab_size


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  if config.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
  if config.cudnn_benchmark:
    torch.backends.cudnn.benchmark = True
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name == 'ar':
    diffusion_model = algo.AR
  elif config.algo.name == 'mdlm':
    diffusion_model = algo.MDLM
  elif config.algo.name == 'duo_base':
    diffusion_model = algo.DUO_BASE
  elif config.algo.name == 'unsid':
    diffusion_model = algo.UNSID
  elif config.algo.name == 'sfm':
    diffusion_model = algo.SFM
  elif config.algo.name == 'flm':
    diffusion_model = algo.FLM
  elif config.algo.name == 'candi':
    diffusion_model = algo.CANDI
  else:
    raise ValueError(
      f'Invalid algorithm name: {config.algo.name}')
  kwargs = {'diffusion_model': diffusion_model,
            'config': config,
            'tokenizer': tokenizer,
            'logger': logger}
  
  if hasattr(config.model, 'vocab_size'):
    kwargs['tokenizer'] = FakeTokenizer(config.model.vocab_size)
  
  if config.mode == 'sample_eval':
    _generate_samples(**kwargs)
  elif config.mode == 'ppl_eval':
    _eval_ppl(**kwargs)
  elif config.mode == 'gsm8k_eval':
    _gsm8k_eval(**kwargs)
  elif config.mode == 'sudoku_eval':
    _sudoku_eval(**kwargs)
  elif config.mode == 'train':
    _train(**kwargs)
  else:
    raise ValueError(config.mode)


if __name__ == '__main__':
  main()
