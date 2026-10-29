#!/bin/sh
#SBATCH --job-name owt-duo-sid
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
# One GPU with at least 24 GB of memory, any model except Blackwell (the image's torch 2.7 cu126
# has no sm_120 kernels)
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
# Set the run in the settings below.
# Download the checkpoint first:
#   huggingface-cli download jdeschena/s-flm owt/duo.ckpt --local-dir ./checkpoints
# Results: ${OUTPUT_DIR}/samples.json (texts, gen. perplexity, entropy, and entropy_trajectory: the
# entropy of xt and of x0 ~ p(x0 | xt) every ENTROPY_EVERY of the steps); the metrics are also in the log.

#CKPT_PATH="${PWD}/checkpoints/owt/duo.ckpt"
CKPT_PATH='outputs/owt/owt_duo_ce_finetune/checkpoints/best.ckpt'  # finetuned with run_finetune_owt_ce.sh
# Hydra runs from hydra.run.dir, so a relative path would be resolved there: make it absolute
case "${CKPT_PATH}" in
  /*) ;;
  *) CKPT_PATH="${PWD}/${CKPT_PATH}" ;;
esac
SAMPLER="sid"       # sid / sid_partial / ancestral_sid / ancestral_restart / dfm / ancestral
RHO="0.1"                     # sid_partial: proportion of the positions resampled at each step
LAM="0.8"                    # ancestral_restart: proportion of the steps of the ancestral pass and restart noise level alpha (any schedule; log-linear: t = 1 - LAM)
X0_PARAM="elbo"        # ancestral / ancestral_sid / ancestral_restart: elbo (DUO checkpoint) / posterior (cross-entropy finetuned checkpoint)
SID_X0_PARAM="posterior"  # sid steps (sid / sid_partial / ancestral_sid / ancestral_restart): posterior (output used as p(x0 | xt)) / elbo (DUO x_theta converted to p(x0 | xt))
STEPS="128"                  # number of sampling steps
BATCH_SIZE="8"
NUM_BATCHES="4"       # total samples = NUM_BATCHES * BATCH_SIZE
ANC_NOISE="log-linear"      # schedule of the ancestral steps (ancestral, ancestral_restart): log-linear / cosine; the denoiser only sees alpha_t, so it need not match the training
ANC_NOISE_POWER="2"  # cosine only: alpha_t = cos(pi/2 t)^power (< 2 more steps near the data, > 2 near the noise)
SID_NOISE="cosine"      # schedule of the sid steps (sid, sid_partial, ancestral_restart with RESTART_SAMPLER=sid): log-linear / cosine
SID_NOISE_POWER="0.5"  # cosine only
NOISE="log-linear"      # ancestral_sid / dfm (one schedule for both kernels): log-linear / cosine
NOISE_POWER="2"  # cosine only
RESTART_SAMPLER="sid"  # ancestral_restart: kernel after the restart, sid / ancestral (second ancestral pass, ablates the sid refinement)
ENTROPY_EVERY="0.1"  # entropy of xt and x0 every 10% of the steps (null to disable)
SAMPLER_NAME="${SAMPLER}"
[ "${SAMPLER}" = "sid_partial" ] && SAMPLER_NAME="${SAMPLER}_rho${RHO}"
[ "${SAMPLER}" = "ancestral_restart" ] && SAMPLER_NAME="${SAMPLER}_lam${LAM}"
[ "${X0_PARAM}" = "posterior" ] && SAMPLER_NAME="${SAMPLER_NAME}_x0post"
case "${SAMPLER}" in
  sid|sid_partial|ancestral_sid|ancestral_restart)
    [ "${SID_X0_PARAM}" = "elbo" ] && SAMPLER_NAME="${SAMPLER_NAME}_sidelbo" ;;
esac
[ "${SAMPLER}" = "ancestral_restart" ] && [ "${RESTART_SAMPLER}" = "ancestral" ] && SAMPLER_NAME="${SAMPLER_NAME}_rsanc"
# Schedule(s) of the run, in the output directory name
sched_name() { if [ "$1" = "cosine" ]; then echo "cosine-p$2"; else echo "$1"; fi; }
case "${SAMPLER}" in
  ancestral) RUN_NOISE=$(sched_name "${ANC_NOISE}" "${ANC_NOISE_POWER}") ;;
  sid|sid_partial) RUN_NOISE=$(sched_name "${SID_NOISE}" "${SID_NOISE_POWER}") ;;
  ancestral_restart)
    RUN_NOISE=$(sched_name "${ANC_NOISE}" "${ANC_NOISE_POWER}")
    [ "${RESTART_SAMPLER}" = "sid" ] && RUN_NOISE="${RUN_NOISE}+$(sched_name "${SID_NOISE}" "${SID_NOISE_POWER}")" ;;
  *) RUN_NOISE=$(sched_name "${NOISE}" "${NOISE_POWER}") ;;
esac
NOISE_NAME="${RUN_NOISE}"
OUTPUT_DIR="${PWD}/eval_runs/owt/ce_denoiser_${SAMPLER_NAME}_${NOISE_NAME}_steps${STEPS}"
[ "${SAMPLER}" = "sid_partial" ] && SAMPLER_ARGS="sampler.rho=${RHO}"
[ "${SAMPLER}" = "ancestral_restart" ] && SAMPLER_ARGS="sampler.lam=${LAM}"
[ "${SAMPLER}" = "ancestral_restart" ] && SAMPLER_ARGS="${SAMPLER_ARGS} ++sampler.restart_sampler=${RESTART_SAMPLER}"
[ "${NOISE}" = "cosine" ] && NOISE_ARGS="noise.power=${NOISE_POWER}"

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
    ++sampler.x0_param="${X0_PARAM}" \
    ++sampler.sid_x0_param="${SID_X0_PARAM}" \
    sampler.steps="${STEPS}" \
    sampler.num_sample_batches="${NUM_BATCHES}" \
    sampler.temperature=1 \
    noise="${NOISE}" ${NOISE_ARGS} \
    ++sampler.ancestral_noise.type="${ANC_NOISE}" \
    ++sampler.ancestral_noise.power="${ANC_NOISE_POWER}" \
    ++sampler.sid_noise.type="${SID_NOISE}" \
    ++sampler.sid_noise.power="${SID_NOISE_POWER}" \
    loader.eval_batch_size="${BATCH_SIZE}" \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    +wandb.offline=True \
    hydra.run.dir="${OUTPUT_DIR}"
