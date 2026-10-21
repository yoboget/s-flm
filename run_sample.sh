#!/bin/sh
#SBATCH --job-name sudoku-duo-sample
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task 4
#SBATCH --mem=32000
#SBATCH --partition private-kalousis-gpu,private-cui-gpu

#SBATCH --time 04:00:00

# Baobab: evaluate a multi-mode DUO trained with run.sh on sudoku: fills in the
# validation puzzles given their clues and reports the accuracy (fully solved grids).
# Submit from the repo root: `mkdir -p sbatch && sbatch run_sample.sh`
# The defaults match run.sh; override e.g. with
#   `SAMPLER=dfm HYP_SELECTION=average STEPS=8,32,1024 sbatch run_sample.sh`
# Every number of sampling steps in STEPS is evaluated in the same launch.
# Results: ${OUTPUT_DIR}/summary.json (accuracy, cell and blank cell accuracy per number of steps)
# and ${OUTPUT_DIR}/steps_<n>/results.json (+ every generated grid).

DIFFICULTY="${DIFFICULTY:-hard}"      # easy / medium / hard
NUM_MODES="${NUM_MODES:-32}"          # must match the checkpoint
HYP_LOSS="${HYP_LOSS:-mixture}"       # only used to find the training run
TRAIN_RUN="sudoku_duo_sid_M${NUM_MODES}_${HYP_LOSS}_${DIFFICULTY}"
CKPT_PATH="${CKPT_PATH:-${PWD}/outputs/sudoku/${TRAIN_RUN}/checkpoints/last.ckpt}"

SAMPLER="${SAMPLER:-sid}"             # sid / dfm / ancestral
HYP_SELECTION="${HYP_SELECTION:-sample}"  # sample / average
STEPS="${STEPS:-8,16,32,64,128}"      # comma-separated numbers of sampling steps
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/eval_runs/sudoku/${TRAIN_RUN}/${SAMPLER}_${HYP_SELECTION}}"

srun apptainer exec --nv --env WANDB_MODE=online --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 sfm.sif \
    python -u -m main \
    mode=sudoku_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    data=sudoku \
    data.cache_dir="${PWD}/data_cache" \
    data.difficulty="${DIFFICULTY}" \
    model=tiny \
    algo=duo-base \
    algo.num_modes="${NUM_MODES}" \
    algo.hyp_loss="${HYP_LOSS}" \
    sampler="${SAMPLER}" \
    sudoku.steps_list="[${STEPS}]" \
    sampler.hyp_selection="${HYP_SELECTION}" \
    noise=log-linear \
    sudoku.batch_size=256 \
    loader.eval_batch_size=256 \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    sudoku.output_dir="${OUTPUT_DIR}" \
    hydra.run.dir="${OUTPUT_DIR}"
