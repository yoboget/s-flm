#!/bin/sh
# Local (no Slurm): evaluate a multi-mode DUO trained with run.sh on sudoku: fills in the
# validation puzzles given their clues and reports the accuracy, cell and blank cell accuracy.
# Run from the repo root: `sh run_sample_local.sh`
# Override e.g. with
#   `SAMPLER=dfm HYP_SELECTION=sample STEPS=8,32 sh run_sample_local.sh`
# Every number of sampling steps in STEPS is evaluated in the same launch.
# Results: ${OUTPUT_DIR}/summary.json (accuracy, cell and blank cell accuracy per number of steps)
# and ${OUTPUT_DIR}/steps_<n>/results.json (+ every generated grid).

DIFFICULTY="${DIFFICULTY:-hard}"      # easy / medium / hard
NUM_MODES="${NUM_MODES:-1}"          # must match the checkpoint
HYP_LOSS="${HYP_LOSS:-mixture}"       # only used to find the training run
TRAIN_RUN="sudoku_duo_sid_M${NUM_MODES}_${HYP_LOSS}_${DIFFICULTY}"
CKPT="${CKPT:-best}"                   # best (lowest val/nll) / last / e.g. 106-20000 (step checkpoint)
CKPT_PATH="${CKPT_PATH:-${PWD}/outputs/sudoku/${TRAIN_RUN}/checkpoints/${CKPT}.ckpt}"

SAMPLER="${SAMPLER:-sid}"             # sid / dfm / ancestral
HYP_SELECTION="${HYP_SELECTION:-sample}"  # sample / average
STEPS="${STEPS:-8, 16, 32, 64, 128}"      # comma-separated numbers of sampling steps
BATCH_SIZE="${BATCH_SIZE:-256}"       # lower it if the GPU runs out of memory
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/eval_runs/sudoku/${TRAIN_RUN}/${CKPT}/${SAMPLER}_${HYP_SELECTION}}"

apptainer exec --nv --env WANDB_MODE=offline --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 sfm.sif \
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
    noise=cosine-squared \
    sudoku.batch_size="${BATCH_SIZE}" \
    loader.eval_batch_size="${BATCH_SIZE}" \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    sudoku.output_dir="${OUTPUT_DIR}" \
    hydra.run.dir="${OUTPUT_DIR}"
