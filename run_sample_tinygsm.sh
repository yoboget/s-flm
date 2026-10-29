#!/bin/sh
#SBATCH --job-name tinygsm-duo-sid
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --cpus-per-task 4
#SBATCH --mem=32000
#SBATCH --partition private-kalousis-gpu,private-cui-gpu,public-gpu,shared-gpu

#SBATCH --time 06:00:00

# Evaluate the pretrained TinyGSM DUO (single mode) on the GSM8K test set with the sid sampler,
# inside the apptainer image. Same setup as scripts/sample/tinygsm/duo.sh, with sampler=sid.
# Run from the repo root:
#   Baobab: `mkdir -p sbatch && sbatch run_sample_tinygsm.sh`
#   Local:  `sh run_sample_tinygsm.sh` (lower BATCH_SIZE if the GPU runs out of memory)
# Override e.g. with `STEPS=64 SAMPLER=ancestral sh run_sample_tinygsm.sh`.
# Download the checkpoint first (see README):
#   huggingface-cli download jdeschena/s-flm tinygsm/duo.ckpt --local-dir ./checkpoints
# and build the GSM8K test file once (needs internet):
#   apptainer exec sfm.sif python scripts/prepare_gsm8k_test.py data/gsm8k_test.json
# Results: ${OUTPUT_DIR} (GSM8K accuracy in the log and the generated answers).

CKPT_PATH="${CKPT_PATH:-${PWD}/checkpoints/tinygsm/duo.ckpt}"
GSM8K_PATH="${GSM8K_PATH:-${PWD}/data/gsm8k_test.json}"
SAMPLER="${SAMPLER:-sid}"             # sid / sid_partial / ancestral_sid / ancestral_restart / dfm / ancestral
RHO="${RHO:-0.1}"                     # sid_partial: proportion of the positions resampled at each step
LAM="${LAM:-0.75}"                    # ancestral_restart: proportion of the steps of the ancestral pass (restart at t = 1 - LAM)
RESTART_ALPHA="${RESTART_ALPHA:-}"    # ancestral_restart: restart at this alpha instead of t = 1 - LAM (empty: 1 - LAM)
X0_PARAM="${X0_PARAM:-elbo}"          # ancestral / ancestral_sid / ancestral_restart: elbo (DUO checkpoint) / posterior (cross-entropy finetuned checkpoint)
SID_X0_PARAM="${SID_X0_PARAM:-posterior}"  # sid steps (sid / sid_partial / ancestral_sid / ancestral_restart): posterior (output used as p(x0 | xt)) / elbo (DUO x_theta converted to p(x0 | xt))
STEPS="${STEPS:-1024}"                  # number of sampling steps
BATCH_SIZE="${BATCH_SIZE:-16}"
NOISE="${NOISE:-log-linear}"          # log-linear / cosine-squared (the denoiser only sees alpha_t)
NOISE_POWER="${NOISE_POWER:-2}"       # cosine-squared only: alpha_t = cos(pi/2 t)^NOISE_POWER (< 2 more steps near the data, > 2 near the noise)
SAMPLER_NAME="${SAMPLER}"
[ "${SAMPLER}" = "sid_partial" ] && SAMPLER_NAME="${SAMPLER}_rho${RHO}"
[ "${SAMPLER}" = "ancestral_restart" ] && SAMPLER_NAME="${SAMPLER}_lam${LAM}"
[ "${SAMPLER}" = "ancestral_restart" ] && [ -n "${RESTART_ALPHA}" ] && SAMPLER_NAME="${SAMPLER_NAME}_ra${RESTART_ALPHA}"
case "${SAMPLER}" in
  ancestral|ancestral_sid|ancestral_restart)
    [ "${X0_PARAM}" = "posterior" ] && SAMPLER_NAME="${SAMPLER_NAME}_x0post" ;;
esac
case "${SAMPLER}" in
  sid|sid_partial|ancestral_sid|ancestral_restart)
    [ "${SID_X0_PARAM}" = "elbo" ] && SAMPLER_NAME="${SAMPLER_NAME}_sidelbo" ;;
esac
NOISE_NAME="${NOISE}"
[ "${NOISE}" = "cosine-squared" ] && NOISE_NAME="cosine-p${NOISE_POWER}"
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/eval_runs/tinygsm/duo_${SAMPLER_NAME}_${NOISE_NAME}_steps${STEPS}}"
[ "${SAMPLER}" = "sid_partial" ] && SAMPLER_ARGS="sampler.rho=${RHO}"
[ "${SAMPLER}" = "ancestral_restart" ] && SAMPLER_ARGS="sampler.lam=${LAM}"
[ "${SAMPLER}" = "ancestral_restart" ] && [ -n "${RESTART_ALPHA}" ] && SAMPLER_ARGS="${SAMPLER_ARGS} sampler.restart_alpha=${RESTART_ALPHA}"
[ "${NOISE}" = "cosine-squared" ] && NOISE_ARGS="noise.power=${NOISE_POWER}"

apptainer exec --nv --env WANDB_MODE=offline --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 sfm.sif \
    python -u -m main \
    mode=gsm8k_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    data=gsm8k-test \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir="${PWD}/data_cache" \
    data.data_path="${GSM8K_PATH}" \
    model=small \
    model.length=512 \
    algo=duo-base \
    sampler="${SAMPLER}" ${SAMPLER_ARGS} \
    ++sampler.x0_param="${X0_PARAM}" \
    ++sampler.sid_x0_param="${SID_X0_PARAM}" \
    sampler.steps="${STEPS}" \
    noise="${NOISE}" ${NOISE_ARGS} \
    loader.eval_batch_size="${BATCH_SIZE}" \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    gsm8k.output_dir="${OUTPUT_DIR}" \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"
