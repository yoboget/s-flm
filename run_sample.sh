#!/bin/sh
#SBATCH --job-name sudoku-duo-sample
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --cpus-per-task 4
#SBATCH --mem=32000
#SBATCH --partition private-kalousis-gpu,private-cui-gpu,public-gpu,shared-gpu

#SBATCH --time 04:00:00

# Baobab: evaluate a denoiser trained with run.sh on sudoku: fills in the
# validation puzzles given their clues and reports the accuracy (fully solved grids).
# Submit from the repo root: `mkdir -p sbatch && sbatch run_sample.sh`
# Set the run in the settings below (the training settings must match run.sh).
# Every number of sampling steps in STEPS is evaluated in the same launch.
# Results: ${OUTPUT_DIR}/summary.json (accuracy, cell and blank cell accuracy per number of steps)
# and ${OUTPUT_DIR}/steps_<n>/results.json (+ every generated grid).

DIFFICULTY="hard"      # easy / medium / hard
ALGO="unsid"                 # unsid / duo-base / mdlm, must match the checkpoint
NUM_MODES="32"          # must match the checkpoint
HYP_LOSS="mixture"       # only used to find the training run
LOSS_WEIGHTING="none"  # duo-base / mdlm: only used to find the training run
if [ "${ALGO}" = "unsid" ]; then
  TRAIN_RUN="sudoku_duo_unsid_M${NUM_MODES}_${HYP_LOSS}_${DIFFICULTY}"
else
  TRAIN_RUN="sudoku_duo_sid_M${NUM_MODES}_${HYP_LOSS}_${LOSS_WEIGHTING}_${DIFFICULTY}"
fi
CKPT="best"                   # best (lowest val/nll) / last / e.g. 106-20000 (step checkpoint)
CKPT_PATH="${PWD}/outputs/sudoku/${TRAIN_RUN}/checkpoints/${CKPT}.ckpt"

SAMPLER="sid"             # sid / sid_partial / ancestral_sid / ancestral_restart / dfm / ancestral (unsid: sid only)
RHO="0.1"                     # sid_partial: proportion of the positions resampled at each step
LAM="0.75"                    # ancestral_restart: proportion of the steps of the ancestral pass and restart noise level alpha (any schedule; log-linear: t = 1 - LAM)
RESTART_SAMPLER="sid"  # ancestral_restart: kernel after the restart, sid / ancestral (second ancestral pass, ablates the sid refinement)
# ancestral / ancestral_sid / ancestral_restart, uniform diffusion: what the denoiser outputs, the DUO
# x_theta (elbo) or p(x0 | xt) (posterior, cross-entropy training); defaults to the training loss
X0_PARAM_DEFAULT=elbo
[ "${LOSS_WEIGHTING}" = "none" ] && X0_PARAM_DEFAULT=posterior
X0_PARAM="${X0_PARAM_DEFAULT}"
SID_X0_PARAM="posterior"  # sid steps (sid / sid_partial / ancestral_sid / ancestral_restart): posterior (output used as p(x0 | xt)) / elbo (DUO x_theta converted to p(x0 | xt))
ALSO_GREEDY="True"     # also decode the last step with argmax, from the same state: results_greedy.json and summary.json[steps].greedy
HYP_SELECTION="average"  # NUM_MODES > 1: sample / average
STEPS="8,16,32,64,128"      # comma-separated numbers of sampling steps
ANC_NOISE="log-linear"      # schedule of the ancestral steps (ancestral, ancestral_restart): log-linear / cosine; the denoiser only sees alpha_t, so it need not match the training
ANC_NOISE_POWER="2"  # cosine only: alpha_t = cos(pi/2 t)^power (< 2 more steps near the data, > 2 near the noise)
SID_NOISE="cosine"      # schedule of the sid steps (sid, sid_partial, ancestral_restart with RESTART_SAMPLER=sid): log-linear / cosine
SID_NOISE_POWER="0.5"  # cosine only
NOISE="log-linear"      # ancestral_sid / dfm (one schedule for both kernels): log-linear / cosine
NOISE_POWER="2"  # cosine only
SAMPLER_NAME="${SAMPLER}"
[ "${SAMPLER}" = "sid_partial" ] && SAMPLER_NAME="${SAMPLER}_rho${RHO}"
[ "${SAMPLER}" = "ancestral_restart" ] && SAMPLER_NAME="${SAMPLER}_lam${LAM}"
[ "${SAMPLER}" = "ancestral_restart" ] && [ "${RESTART_SAMPLER}" = "ancestral" ] && SAMPLER_NAME="${SAMPLER_NAME}_rsanc"
case "${SAMPLER}" in
  ancestral|ancestral_sid|ancestral_restart)
    [ "${X0_PARAM}" = "posterior" ] && SAMPLER_NAME="${SAMPLER_NAME}_x0post" ;;
esac
case "${SAMPLER}" in
  sid|sid_partial|ancestral_sid|ancestral_restart)
    [ "${SID_X0_PARAM}" = "elbo" ] && SAMPLER_NAME="${SAMPLER_NAME}_sidelbo" ;;
esac
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
[ "${RUN_NOISE}" != "log-linear" ] && SAMPLER_NAME="${SAMPLER_NAME}_${RUN_NOISE}"
OUTPUT_DIR="${PWD}/eval_runs/sudoku/${TRAIN_RUN}/${CKPT}/${SAMPLER_NAME}_${HYP_SELECTION}"
[ "${SAMPLER}" = "sid_partial" ] && SAMPLER_ARGS="sampler.rho=${RHO}"
[ "${SAMPLER}" = "ancestral_restart" ] && SAMPLER_ARGS="sampler.lam=${LAM}"
[ "${SAMPLER}" = "ancestral_restart" ] && SAMPLER_ARGS="${SAMPLER_ARGS} ++sampler.restart_sampler=${RESTART_SAMPLER}"
[ "${NOISE}" = "cosine" ] && NOISE_ARGS="noise.power=${NOISE_POWER}"

srun apptainer exec --nv --env WANDB_MODE=online --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 sfm.sif \
    python -u -m main \
    mode=sudoku_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    data=sudoku \
    data.cache_dir="${PWD}/data_cache" \
    data.difficulty="${DIFFICULTY}" \
    model=tiny \
    algo="${ALGO}" \
    algo.num_modes="${NUM_MODES}" \
    algo.hyp_loss="${HYP_LOSS}" \
    sampler="${SAMPLER}" ${SAMPLER_ARGS} \
    ++sampler.x0_param="${X0_PARAM}" \
    ++sampler.sid_x0_param="${SID_X0_PARAM}" \
    ++eval.also_greedy="${ALSO_GREEDY}" \
    sudoku.steps_list="[${STEPS}]" \
    sampler.hyp_selection="${HYP_SELECTION}" \
    noise="${NOISE}" ${NOISE_ARGS} \
    ++sampler.ancestral_noise.type="${ANC_NOISE}" \
    ++sampler.ancestral_noise.power="${ANC_NOISE_POWER}" \
    ++sampler.sid_noise.type="${SID_NOISE}" \
    ++sampler.sid_noise.power="${SID_NOISE_POWER}" \
    sudoku.batch_size=256 \
    loader.eval_batch_size=256 \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    sudoku.output_dir="${OUTPUT_DIR}" \
    hydra.run.dir="${OUTPUT_DIR}"
