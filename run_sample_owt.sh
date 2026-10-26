#!/bin/sh
#SBATCH --job-name owt-duo-sid
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
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
# Override e.g. with `STEPS=128 SAMPLER=ancestral sh run_sample_owt.sh`.
# Download the checkpoint first:
#   huggingface-cli download jdeschena/s-flm owt/duo.ckpt --local-dir ./checkpoints
# Results: ${OUTPUT_DIR}/samples.json (texts, gen. perplexity, entropy); the metrics are also in the log.

CKPT_PATH="${CKPT_PATH:-${PWD}/checkpoints/owt/duo.ckpt}"
SAMPLER="${SAMPLER:-sid}"             # sid / dfm / ancestral
STEPS="${STEPS:-128}"                  # number of sampling steps
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_BATCHES="${NUM_BATCHES:-4}"       # total samples = NUM_BATCHES * BATCH_SIZE
NOISE="${NOISE:-log-linear}"          # log-linear / cosine-squared
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/eval_runs/owt/duo_${SAMPLER}_${NOISE}_steps${STEPS}}"

apptainer exec --nv --env WANDB_MODE=offline --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 sfm.sif \
    python -u -m main \
    mode=sample_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.compute_generative_perplexity=True \
    eval.results_json_path="${OUTPUT_DIR}/samples.json" \
    data=openwebtext-split \
    data.cache_dir="${PWD}/data_cache" \
    model=small \
    algo=mdlm \
    sampler="${SAMPLER}" \
    sampler.steps="${STEPS}" \
    sampler.num_sample_batches="${NUM_BATCHES}" \
    sampler.temperature=0.7 \
    noise="${NOISE}" \
    loader.eval_batch_size="${BATCH_SIZE}" \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"
