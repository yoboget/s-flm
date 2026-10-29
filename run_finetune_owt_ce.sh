#!/bin/sh
#SBATCH --job-name owt-duo-ce
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
# 2 A5000 on one node, one task per GPU (Lightning DDP under srun: ntasks-per-node = DEVICES),
# 8 CPUs per task for its 8 data workers
#SBATCH --gres=gpu:nvidia_rtx_a5000:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task 8
#SBATCH --mem=64000
#SBATCH --partition private-kalousis-gpu,private-cui-gpu,public-gpu,shared-gpu

#SBATCH --time 12:00:00

# Baobab: fine-tune the pretrained OWT DUO (trained with the DUO ELBO) with the plain
# cross-entropy of x0 (algo.loss_weighting=none), so that the denoiser output is the
# posterior p(x0 | xt), as sid and the other x0-sampling samplers assume. Inside the
# apptainer image. Evaluate it with run_sample_owt.sh (CKPT_PATH=.../checkpoints/best.ckpt).
# Submit from the repo root: `mkdir -p sbatch && sbatch run_finetune_owt_ce.sh`
# Resubmitting with the same RUN_NAME resumes from the last checkpoint.
# Download the pretrained checkpoint first:
#   huggingface-cli download jdeschena/s-flm owt/duo.ckpt --local-dir ./checkpoints
# The first launch downloads OpenWebText (jdeschena/openwebtext) and tokenizes it into
# CACHE_DIR (tens of GB, a few hours); later launches reuse it.

PRETRAINED="${PRETRAINED:-${PWD}/checkpoints/owt/duo.ckpt}"
CACHE_DIR="${CACHE_DIR:-${HOME}/scratch/data_cache}"  # tokenized OWT, written by prepare_owt.sh
HF_HOME="${HF_HOME:-${CACHE_DIR}/hf}"         # raw download of jdeschena/openwebtext (not the default ~/.cache)
DEVICES="${DEVICES:-2}"                       # GPUs (match --gres)
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-120}" # sequences of 1024 tokens per optimizer step, a multiple of BATCH_SIZE * DEVICES
BATCH_SIZE="${BATCH_SIZE:-12}"                 # per GPU and forward pass (gradient accumulation for the rest); 16 runs out of memory on a 24 GB A5000
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-1000}"
EMA="${EMA:-0.999}"                           # the EMA starts from the pretrained EMA; 0.9999 would keep it close to the ELBO model for ~10k steps
MAX_STEPS="${MAX_STEPS:-50000}"             # optimizer steps
VAL_EVERY="${VAL_EVERY:-2000}"              # optimizer steps between validations
RUN_NAME="${RUN_NAME:-owt_duo_ce_finetune}"
mkdir -p "${CACHE_DIR}" "${HF_HOME}"
# ~/scratch is a link to /srv/beegfs/scratch/...: use the real paths and bind them (apptainer
# only mounts $HOME, the link target is not visible inside the container).
CACHE_DIR="$(readlink -f "${CACHE_DIR}")"
HF_HOME="$(readlink -f "${HF_HOME}")"
# Lightning counts val_check_interval in forward passes, i.e. with gradient accumulation
ACCUMULATE=$((GLOBAL_BATCH_SIZE / (BATCH_SIZE * DEVICES)))
if [ $((GLOBAL_BATCH_SIZE % (BATCH_SIZE * DEVICES))) -ne 0 ]; then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} is not a multiple of BATCH_SIZE * DEVICES = $((BATCH_SIZE * DEVICES))" >&2
  exit 1
fi

srun apptainer exec --nv --bind "${CACHE_DIR}" --env WANDB_MODE=online --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 --env HF_HOME="${HF_HOME}" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True sfm.sif \
    python -u -m main \
    training.finetune_path="${PRETRAINED}" \
    training.ema="${EMA}" \
    data=openwebtext-split \
    data.cache_dir="${CACHE_DIR}" \
    model=small \
    algo=duo-base \
    algo.loss_weighting=none \
    noise=log-linear \
    optim.lr="${LR}" \
    lr_scheduler.num_warmup_steps="${WARMUP}" \
    loader.global_batch_size="${GLOBAL_BATCH_SIZE}" \
    loader.batch_size="${BATCH_SIZE}" \
    loader.eval_batch_size="${BATCH_SIZE}" \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes=1 \
    trainer.devices="${DEVICES}" \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.val_check_interval=$((VAL_EVERY * ACCUMULATE)) \
    trainer.limit_val_batches=50 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=1_000 \
    callbacks.checkpoint_every_n_steps.save_top_k=0 \
    wandb.project=s-flm \
    +wandb.name="${RUN_NAME}" \
    hydra.run.dir="${PWD}/outputs/owt/${RUN_NAME}"
