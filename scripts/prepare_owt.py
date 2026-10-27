"""Downloads OpenWebText (jdeschena/openwebtext) and writes the tokenized, wrapped train and
valid sets that training reads from data.cache_dir
(openwebtext-{train,valid}_{train,valid}_bs<length>_wrapped.dat), on CPU only.

Usage (from the repo root, see prepare_owt.sh):
  python scripts/prepare_owt.py data.cache_dir=... loader.num_workers=32 [hydra overrides]
The overrides must match the training run for the files to be found (data=openwebtext-split
and model.length=1024 by default). loader.num_workers sets the number of processes.
"""

import os
import sys

from hydra import compose, initialize_config_dir

sys.path.insert(0, os.getcwd())
import dataloader
import main  # noqa: F401 (registers the omegaconf resolvers)


def run():
  overrides = ['data=openwebtext-split', 'model=small'] + sys.argv[1:]
  with initialize_config_dir(
      config_dir=os.path.abspath('configs'), version_base=None):
    config = compose(config_name='config', overrides=overrides)
  print(f'cache_dir: {config.data.cache_dir}, '
        f'length: {config.model.length}, '
        f'processes: {config.loader.num_workers}', flush=True)
  tokenizer = dataloader.get_tokenizer(config)
  for mode in ('valid', 'train'):
    dataset = dataloader.get_dataset(config, tokenizer, mode=mode)
    print(f'{mode}: {len(dataset)} sequences of '
          f'{config.model.length} tokens', flush=True)


if __name__ == '__main__':
  run()
