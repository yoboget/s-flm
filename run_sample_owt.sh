#!/bin/sh
#SBATCH --job-name owt-duo-sid
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
# One GPU of whichever of these 24-32 GB models is free first (a gres request takes a single
# type). No RTX 5090: Blackwell (sm_120) needs CUDA 12.8 wheels, the image has torch 2.7 cu126.
#SBATCH --gres=gpu:1
#SBATCH --constraint="COMPUTE_MODEL_NVIDIA_GEFORCE_RTX_3090|COMPUTE_MODEL_NVIDIA_RTX_A5000|COMPUTE_MODEL_NVIDIA_RTX_A5500|COMPUTE_MODEL_NVIDIA_GEFORCE_RTX_4090|COMPUTE_MODEL_NVIDIA_RTX_5000"
#SBATCH --cpus-per-task 4
#SBATCH --mem=32000
#SBATCH --partition private-kalousis-gpu,private-cui-gpu,public-gpu,shared-gpu

#SBATCH --time 04:00:00

# Unconditional samples from the pretrained OWT DUO (single mode) with the sid sampler, scored
# by generative perplexity (gpt2-large) and sample entropy, inside the apptainer image.
# Same setup as scripts/sample/owt/duo.sh, with sampler=sid.
# Run from the repo root:
#   Baobab: `mkdir -p sbatch && sbatch run_sample_owt.sh`
#   Local:  `sh run_sample_owt.sh` (lower BATCH_SIZE if the GPU runs out of memory)
# Override e.g. with `STEPS=128 SAMPLER=ancestral sh run_sample_owt.sh`
# or `STEPS=1024 SAMPLER=sid_partial RHO=0.3 sh run_sample_owt.sh`.
# Download the checkpoint first:
#   huggingface-cli download jdeschena/s-flm owt/duo.ckpt --local-dir ./checkpoints
# Results: ${OUTPUT_DIR}/samples.json (texts, gen. perplexity, entropy, and entropy_trajectory: the
# entropy of xt and of x0 ~ p(x0 | xt) every ENTROPY_EVERY of the steps); the metrics are also in the log.

#CKPT_PATH="${CKPT_PATH:-${PWD}/checkpoints/owt/duo.ckpt}"
CKPT_PATH='outputs/owt/owt_duo_ce_finetune/checkpoints/best.ckpt'  # finetuned with run_finetune_owt_ce.sh
SAMPLER="${SAMPLER:-sid}"       # sid / sid_partial / ancestral_sid / ancestral_restart / dfm / ancestral
RHO="${RHO:-0.1}"                     # sid_partial: proportion of the positions resampled at each step
LAM="${LAM:-0.8}"                    # ancestral_restart: proportion of the steps of the ancestral pass (restart at t = 1 - LAM)
STEPS="${STEPS:-128}"                  # number of sampling steps
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_BATCHES="${NUM_BATCHES:-4}"       # total samples = NUM_BATCHES * BATCH_SIZE
NOISE="${NOISE:-log-linear}"          # log-linear / cosine-squared
ENTROPY_EVERY="${ENTROPY_EVERY:-0.1}"  # entropy of xt and x0 every 10% of the steps (null to disable)
SAMPLER_NAME="${SAMPLER}"
[ "${SAMPLER}" = "sid_partial" ] && SAMPLER_NAME="${SAMPLER}_rho${RHO}"
[ "${SAMPLER}" = "ancestral_restart" ] && SAMPLER_NAME="${SAMPLER}_lam${LAM}"
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/eval_runs/owt/duo_${SAMPLER_NAME}_${NOISE}_steps${STEPS}}"
[ "${SAMPLER}" = "sid_partial" ] && SAMPLER_ARGS="sampler.rho=${RHO}"
[ "${SAMPLER}" = "ancestral_restart" ] && SAMPLER_ARGS="sampler.lam=${LAM}"

apptainer exec --nv --env WANDB_MODE=offline --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 sfm.sif \
    python -u -m main \
    mode=sample_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.compute_generative_perplexity=True \
    eval.results_json_path="${OUTPUT_DIR}/samples.json" \
    eval.entropy_every="${ENTROPY_EVERY}" \
    data=openwebtext-split \
    data.cache_dir="${PWD}/data_cache" \
    model=small \
    algo=duo-base \
    sampler="${SAMPLER}" ${SAMPLER_ARGS} \
    sampler.steps="${STEPS}" \
    sampler.num_sample_batches="${NUM_BATCHES}" \
    sampler.temperature=1 \
    noise="${NOISE}" \
    loader.eval_batch_size="${BATCH_SIZE}" \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"
