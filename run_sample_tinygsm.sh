#!/bin/sh
#SBATCH --job-name tinygsm-duo-sid
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --cpus-per-task 4
#SBATCH --mem=32000
#SBATCH --partition private-kalousis-gpu,private-cui-gpu,public-gpu,shared-gpu

#SBATCH --time 04:00:00

# Evaluate the pretrained TinyGSM DUO (single mode) on the GSM8K test set with the sid sampler,
# inside the apptainer image. Same setup as scripts/sample/tinygsm/duo.sh, with sampler=sid.
# Run from the repo root:
#   Baobab: `mkdir -p sbatch && sbatch run_sample_tinygsm.sh`
#   Local:  `sh run_sample_tinygsm.sh` (lower BATCH_SIZE if the GPU runs out of memory)
# Override e.g. with `STEPS=64 SAMPLER=ancestral sh run_sample_tinygsm.sh`.
# Download the checkpoint first (see README):
#   huggingface-cli download jdeschena/s-flm tinygsm/duo.ckpt --local-dir ./checkpoints
# Results: ${OUTPUT_DIR} (GSM8K accuracy in the log and the generated answers).

CKPT_PATH="${CKPT_PATH:-${PWD}/checkpoints/tinygsm/duo.ckpt}"
SAMPLER="${SAMPLER:-sid}"             # sid / dfm / ancestral
STEPS="${STEPS:-32}"                  # number of sampling steps
BATCH_SIZE="${BATCH_SIZE:-32}"
NOISE="${NOISE:-log-linear}"          # log-linear / cosine-squared
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/eval_runs/tinygsm/duo_${SAMPLER}_${NOISE}_steps${STEPS}}"

apptainer exec --nv --env WANDB_MODE=offline --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 sfm.sif \
    python -u -m main \
    mode=gsm8k_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    data=gsm8k-test \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir="${PWD}/data_cache" \
    model=small \
    model.length=512 \
    algo=duo-base \
    sampler="${SAMPLER}" \
    sampler.steps="${STEPS}" \
    noise="${NOISE}" \
    loader.eval_batch_size="${BATCH_SIZE}" \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    gsm8k.output_dir="${OUTPUT_DIR}" \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"
