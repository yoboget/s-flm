#!/bin/sh
#SBATCH --job-name owt-prepare
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
#SBATCH --partition shared-cpu
#SBATCH --cpus-per-task 32
#SBATCH --mem=96000
#SBATCH --exclude=cpu[350,352]

#SBATCH --time 08:00:00

# Baobab, CPU only: download OpenWebText and tokenize it into CACHE_DIR, where
# run_finetune_owt_ce.sh (same CACHE_DIR) reads it. Inside the apptainer image.
# Submit from the repo root: `mkdir -p sbatch && sbatch prepare_owt.sh`
# 32 cores / 96 GB fit the Cascade Lake (36 cores, 187 GB) and EPYC nodes. If the job is
# interrupted, resubmitting reuses the finished download and the cached map steps.
# Disk: roughly 250 GB in CACHE_DIR at the end (raw arrow, intermediate map caches and
# the final .dat files); only the two *_wrapped.dat directories are needed for training.

CACHE_DIR="${CACHE_DIR:-${HOME}/scratch/data_cache}"
HF_HOME="${HF_HOME:-${CACHE_DIR}/hf}"   # raw download (not the default ~/.cache)
NUM_PROC="${NUM_PROC:-${SLURM_CPUS_PER_TASK:-32}}"

mkdir -p "${CACHE_DIR}" "${HF_HOME}"
# ~/scratch is a link to /srv/beegfs/scratch/...: use the real paths and bind them, since
# apptainer only mounts $HOME and the link target is not visible inside the container.
CACHE_DIR="$(readlink -f "${CACHE_DIR}")"
HF_HOME="$(readlink -f "${HF_HOME}")"

# One thread per process: the parallelism comes from the NUM_PROC dataset workers.
srun apptainer exec \
    --bind "${CACHE_DIR}" \
    --env HF_HOME="${HF_HOME}" \
    --env TOKENIZERS_PARALLELISM=false \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    sfm.sif \
    python -u scripts/prepare_owt.py \
    data.cache_dir="${CACHE_DIR}" \
    loader.num_workers="${NUM_PROC}"
