#!/bin/sh
# Local (no Slurm): evaluate a DUO trained with run.sh on sudoku: fills in the
# validation puzzles given their clues and reports the accuracy, cell and blank cell accuracy.
# Run from the repo root: `sh run_sample_local.sh`, inside sfm.sif if it is there, otherwise with
# the active python (e.g. after `conda activate sfm`, or PYTHON=/path/to/python).
# Override e.g. with
#   `SAMPLER=dfm HYP_SELECTION=sample STEPS=8,32 sh run_sample_local.sh`
#   `SAMPLER=ancestral_restart LAM=0.75 RESTART_ALPHA=0.8 NOISE_POWER=1 sh run_sample_local.sh`
# Every number of sampling steps in STEPS is evaluated in the same launch.
# Results: ${OUTPUT_DIR}/summary.json (accuracy, cell and blank cell accuracy per number of steps)
# and ${OUTPUT_DIR}/steps_<n>/results.json (+ every generated grid).

DIFFICULTY="${DIFFICULTY:-hard}"      # easy / medium / hard
NUM_MODES="${NUM_MODES:-1}"          # must match the checkpoint
HYP_LOSS="${HYP_LOSS:-mixture}"       # only used to find the training run
TRAIN_LOSS="${TRAIN_LOSS:-none}"  # only used to find the training run
TRAIN_RUN="${TRAIN_RUN:-sudoku_duo_sid_M${NUM_MODES}_${HYP_LOSS}_${TRAIN_LOSS}_${DIFFICULTY}}"
CKPT="${CKPT:-best}"                   # best (lowest val/nll) / last / e.g. 106-20000 (step checkpoint)
CKPT_PATH="${CKPT_PATH:-${PWD}/outputs/sudoku/${TRAIN_RUN}/checkpoints/${CKPT}.ckpt}"

SAMPLER="${SAMPLER:-sid}"       # sid / sid_partial / ancestral_sid / ancestral_restart / dfm / ancestral
RHO="${RHO:-0.1}"                     # sid_partial: proportion of the positions resampled at each step
LAM="${LAM:-0.75}"                    # ancestral_restart: proportion of the steps of the ancestral pass (restart at t = 1 - LAM)
RESTART_ALPHA="${RESTART_ALPHA:-}"    # ancestral_restart: restart at this alpha instead of t = 1 - LAM (empty: 1 - LAM)
# ancestral / ancestral_sid / ancestral_restart, uniform diffusion: what the denoiser outputs, the DUO
# x_theta (elbo) or p(x0 | xt) (posterior, cross-entropy training); defaults to posterior when
# TRAIN_LOSS starts with "none" (loss_weighting=none), elbo otherwise
case "${TRAIN_LOSS}" in
  none*) X0_PARAM_DEFAULT=posterior ;;
  *) X0_PARAM_DEFAULT=elbo ;;
esac
X0_PARAM="${X0_PARAM:-${X0_PARAM_DEFAULT}}"
SID_X0_PARAM="${SID_X0_PARAM:-posterior}"  # sid steps (sid / sid_partial / ancestral_sid / ancestral_restart): posterior (output used as p(x0 | xt)) / elbo (DUO x_theta converted to p(x0 | xt))
ALSO_GREEDY="${ALSO_GREEDY:-True}"     # also decode the last step with argmax, from the same state: results_greedy.json and summary.json[steps].greedy
HYP_SELECTION="${HYP_SELECTION:-sample}"  # NUM_MODES > 1: sample / average
STEPS="${STEPS:-8, 16, 32, 64, 128}"      # comma-separated numbers of sampling steps
BATCH_SIZE="${BATCH_SIZE:-256}"       # lower it if the GPU runs out of memory
NOISE="${NOISE:-cosine-squared}"      # sampling schedule; the denoiser only sees alpha_t, so it need not match the training
NOISE_POWER="${NOISE_POWER:-1}"       # cosine-squared only: alpha_t = cos(pi/2 t)^NOISE_POWER (< 2 more steps near the data, > 2 near the noise)
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
# cosine-squared with the default power keeps the previous output directories
[ "${NOISE}" = "cosine-squared" ] && [ "${NOISE_POWER}" != "2" ] && SAMPLER_NAME="${SAMPLER_NAME}_cosine-p${NOISE_POWER}"
[ "${NOISE}" = "log-linear" ] && SAMPLER_NAME="${SAMPLER_NAME}_log-linear"
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/eval_runs/sudoku/${TRAIN_RUN}/${CKPT}/${SAMPLER_NAME}_${HYP_SELECTION}}"
[ "${SAMPLER}" = "sid_partial" ] && SAMPLER_ARGS="sampler.rho=${RHO}"
[ "${SAMPLER}" = "ancestral_restart" ] && SAMPLER_ARGS="sampler.lam=${LAM}"
[ "${SAMPLER}" = "ancestral_restart" ] && [ -n "${RESTART_ALPHA}" ] && SAMPLER_ARGS="${SAMPLER_ARGS} sampler.restart_alpha=${RESTART_ALPHA}"
[ "${NOISE}" = "cosine-squared" ] && NOISE_ARGS="noise.power=${NOISE_POWER}"

export WANDB_MODE=offline TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
if [ -f sfm.sif ]; then
  RUN="apptainer exec --nv sfm.sif python"
else
  RUN="${PYTHON:-python}"
fi

${RUN} -u -m main \
    mode=sudoku_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    data=sudoku \
    data.cache_dir="${PWD}/data_cache" \
    data.difficulty="${DIFFICULTY}" \
    model=tiny \
    algo=duo-base \
    algo.num_modes="${NUM_MODES}" \
    algo.hyp_loss="${HYP_LOSS}" \
    sampler="${SAMPLER}" ${SAMPLER_ARGS} \
    ++sampler.x0_param="${X0_PARAM}" \
    ++sampler.sid_x0_param="${SID_X0_PARAM}" \
    ++eval.also_greedy="${ALSO_GREEDY}" \
    sudoku.steps_list="[${STEPS}]" \
    sampler.hyp_selection="${HYP_SELECTION}" \
    noise="${NOISE}" ${NOISE_ARGS} \
    sudoku.batch_size="${BATCH_SIZE}" \
    loader.eval_batch_size="${BATCH_SIZE}" \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    sudoku.output_dir="${OUTPUT_DIR}" \
    hydra.run.dir="${OUTPUT_DIR}"
