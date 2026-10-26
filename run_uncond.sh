#!/bin/sh
#SBATCH --job-name sudoku-uncond
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --cpus-per-task 4
#SBATCH --mem=32000
#SBATCH --partition private-kalousis-gpu,private-cui-gpu,public-gpu,shared-gpu

#SBATCH --time 12:00:00

# Baobab: train a denoiser to generate complete sudoku grids from scratch (unconditional,
# data.unconditional=True: [BOS] solution, 90 tokens, loss on every token), inside the
# apptainer image. Evaluate it with run_sample_uncond.sh.
# Submit from the repo root: `mkdir -p sbatch && sbatch run_uncond.sh`
# Override e.g. with: `ALGO=mdlm NUM_MODES=1 sbatch run_uncond.sh`
# The grids are the solutions of the conditional datasets (same seed), generated once
# in data_cache/sudoku_uncond_train48000_valid2000_seed42.

ALGO="${ALGO:-duo-base}"              # duo-base / mdlm / unsid
NUM_MODES="${NUM_MODES:-32}"           # 1: standard single-mode denoiser
HYP_LOSS="${HYP_LOSS:-mixture}"       # mixture / wta / tempered (NUM_MODES > 1)
LOSS_WEIGHTING="${LOSS_WEIGHTING:-none}"  # none (cross-entropy of x0) / elbo (ELBO, duo-base / mdlm; noise-level weight, unsid) / log (unsid)
NOISE="${NOISE:-log-linear}"          # log-linear / cosine-squared
RUN_NAME="${RUN_NAME:-sudoku_uncond_${ALGO}_M${NUM_MODES}_${HYP_LOSS}_${LOSS_WEIGHTING}}"

srun apptainer exec --nv --env WANDB_MODE=online sfm.sif \
    python -u -m main \
    data=sudoku \
    data.unconditional=True \
    data.cache_dir="${PWD}/data_cache" \
    data.sudoku_num_workers=4 \
    model=tiny \
    model.length=90 \
    algo="${ALGO}" \
    algo.num_modes="${NUM_MODES}" \
    algo.hyp_loss="${HYP_LOSS}" \
    algo.loss_weighting="${LOSS_WEIGHTING}" \
    sampler=sid \
    noise="${NOISE}" \
    loader.global_batch_size=256 \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    loader.num_workers=4 \
    eval.generate_samples=True \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.val_check_interval=1_000 \
    trainer.limit_val_batches=1.0 \
    trainer.max_steps=40_000 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=1_000 \
    wandb.project=s-flm \
    +wandb.name="${RUN_NAME}" \
    hydra.run.dir="${PWD}/outputs/sudoku/${RUN_NAME}"
