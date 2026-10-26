#!/bin/sh
#SBATCH --job-name sudoku-uncond-sample
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --cpus-per-task 4
#SBATCH --mem=32000
#SBATCH --partition private-kalousis-gpu,private-cui-gpu,public-gpu,shared-gpu

#SBATCH --time 04:00:00

ALGO="${ALGO:-duo-base}"              # duo-base / mdlm / unsid, must match the checkpoint
NUM_MODES="${NUM_MODES:-1}"           # must match the checkpoint
HYP_LOSS="${HYP_LOSS:-mixture}"       # only used to find the training run
LOSS_WEIGHTING="${LOSS_WEIGHTING:-none}"  # only used to find the training run
NOISE="${NOISE:-log-linear}"          # must match the training
TRAIN_RUN="${TRAIN_RUN:-sudoku_uncond_${ALGO}_M${NUM_MODES}_${HYP_LOSS}_${LOSS_WEIGHTING}}2"
CKPT="${CKPT:-best}"                   # best (lowest val/nll) / last / e.g. 106-20000 (step checkpoint)
CKPT_PATH="${CKPT_PATH:-${PWD}/outputs/sudoku/${TRAIN_RUN}/checkpoints/${CKPT}.ckpt}"

SAMPLER="${SAMPLER:-sid}"             # sid / sid_partial / ancestral_sid / ancestral_restart / dfm / ancestral (unsid: sid only)
RHO="${RHO:-0.1}"                     # sid_partial: proportion of the positions resampled at each step
LAM="${LAM:-0.75}"                    # ancestral_restart: proportion of the steps of the ancestral pass (restart at t = 1 - LAM)
HYP_SELECTION="${HYP_SELECTION:-sample}"  # NUM_MODES > 1: sample / average
STEPS="${STEPS:-8,16,32,64,128}"      # comma-separated numbers of sampling steps
NUM_SAMPLES="${NUM_SAMPLES:-1024}"    # grids generated for each number of steps
ENTROPY_EVERY="${ENTROPY_EVERY:-0.1}" # entropy trajectory every 10% of the steps (null to disable)
SAMPLER_NAME="${SAMPLER}"
[ "${SAMPLER}" = "sid_partial" ] && SAMPLER_NAME="${SAMPLER}_rho${RHO}"
[ "${SAMPLER}" = "ancestral_restart" ] && SAMPLER_NAME="${SAMPLER}_lam${LAM}"
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/eval_runs/sudoku/${TRAIN_RUN}/${CKPT}/${SAMPLER_NAME}_${HYP_SELECTION}}"
[ "${SAMPLER}" = "sid_partial" ] && SAMPLER_ARGS="sampler.rho=${RHO}"
[ "${SAMPLER}" = "ancestral_restart" ] && SAMPLER_ARGS="sampler.lam=${LAM}"

srun apptainer exec --nv --env WANDB_MODE=offline --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 sfm.sif \
    python -u -m main \
    mode=sudoku_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.entropy_every="${ENTROPY_EVERY}" \
    data=sudoku \
    data.unconditional=True \
    data.cache_dir="${PWD}/data_cache" \
    model=tiny \
    model.length=90 \
    algo="${ALGO}" \
    algo.num_modes="${NUM_MODES}" \
    algo.hyp_loss="${HYP_LOSS}" \
    sampler="${SAMPLER}" ${SAMPLER_ARGS} \
    sampler.hyp_selection="${HYP_SELECTION}" \
    sudoku.steps_list="[${STEPS}]" \
    sudoku.num_samples="${NUM_SAMPLES}" \
    noise="${NOISE}" \
    sudoku.batch_size=256 \
    loader.eval_batch_size=256 \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    sudoku.output_dir="${OUTPUT_DIR}" \
    hydra.run.dir="${OUTPUT_DIR}"
