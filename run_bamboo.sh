#!/bin/sh
#SBATCH --job-name sudoku-sfm
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task 4
#SBATCH --mem=32000
#SBATCH --partition shared-gpu,public-gpu

#SBATCH --time 12:00:00

# Train S-FLM on sudoku inside the apptainer image (see apptainer/sfm.def).
# Submit from the repo root: `mkdir -p sbatch && sbatch run_bamboo.sh`
# Override the difficulty with: `DIFFICULTY=hard sbatch run_bamboo.sh`
# Resubmitting with the same difficulty resumes from the last checkpoint.

DIFFICULTY="${DIFFICULTY:-easy}"      # easy / medium / hard
RUN_NAME="sudoku_sfm_${DIFFICULTY}"

srun apptainer exec --nv --env WANDB_MODE=online sfm.sif \
    python -u -m main \
    data=sudoku \
    data.cache_dir="${PWD}/data_cache" \
    data.difficulty="${DIFFICULTY}" \
    model=tiny-sphere-dit \
    algo=sfm \
    algo.invert_time_convention=false \
    noise=log-linear \
    loader.global_batch_size=256 \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    loader.num_workers=4 \
    eval.generate_samples=False \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.val_check_interval=20_000 \
    trainer.limit_val_batches=0 \
    trainer.max_steps=20_000 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=1_000 \
    wandb.project=s-flm \
    +wandb.name="${RUN_NAME}" \
    hydra.run.dir="${PWD}/outputs/sudoku/${RUN_NAME}"
