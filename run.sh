#!/bin/sh
#SBATCH --job-name sudoku-duo-modes
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --cpus-per-task 4
#SBATCH --mem=32000
#SBATCH --partition private-kalousis-gpu,private-cui-gpu,public-gpu,shared-gpu

#SBATCH --time 12:00:00

# Baobab: train a multi-mode DUO (uniform-state diffusion, sid sampler) on sudoku inside the apptainer image (see apptainer/sfm.def).
# Submit from the repo root: `mkdir -p sbatch && sbatch run.sh`
# Override the difficulty with: `DIFFICULTY=hard sbatch run.sh`
# Resubmitting with the same difficulty resumes from the last checkpoint.

DIFFICULTY="${DIFFICULTY:-hard}"      # easy / medium / hard
NUM_MODES="${NUM_MODES:-32}"          # 1: standard single-mode denoiser
HYP_LOSS="${HYP_LOSS:-mixture}"       # mixture / wta / tempered
RUN_NAME="sudoku_duo_unsid_M${NUM_MODES}_${HYP_LOSS}_${DIFFICULTY}"

srun apptainer exec --nv --env WANDB_MODE=online sfm.sif \
    python -u -m main \
    data=sudoku \
    data.cache_dir="${PWD}/data_cache" \
    data.difficulty="${DIFFICULTY}" \
    data.sudoku_num_workers=4 \
    model=tiny \
    algo=unsid \
    algo.num_modes="${NUM_MODES}" \
    algo.hyp_loss="${HYP_LOSS}" \
    sampler=sid \
    sampler.hyp_selection=sample \
    noise=log-linear \
    loader.global_batch_size=256 \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    loader.num_workers=4 \
    eval.generate_samples=True \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.val_check_interval=20_000 \
    trainer.limit_val_batches=4 \
    trainer.max_steps=20_000 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=1_000 \
    wandb.project=s-flm \
    +wandb.name="${RUN_NAME}" \
    hydra.run.dir="${PWD}/outputs/sudoku/${RUN_NAME}"
