#!/bin/sh
#SBATCH --job-name owt-sweep
#SBATCH --error sbatch/error.e%A_%a
#SBATCH --output sbatch/out.o%A_%a
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --mem=32000
#SBATCH --partition public-gpu,shared-gpu

#SBATCH --time 8:00:00

# Sweep of the OWT sampling space, one configuration per array task: unconditional samples scored by generative
# perplexity (gpt2-large) and sample entropy, for
#   duo         + sid        (DUO x_theta converted to p(x0 | xt): sid_x0_param=elbo)
#   ce_denoiser + sid        (cross-entropy model, its output is already p(x0 | xt))
#   duo         + ancestral  (DUO x_theta used as is: x0_param=elbo)
# and every number of steps in STEPS.
# Checkpoints: checkpoints/owt/duo.ckpt and checkpoints/owt/ce_denoiser.ckpt.
# Run from the repo root: `mkdir -p sbatch && sbatch run_sample_owt_sweep.sh` submits the three
# configurations as array tasks 0, 1, 2 (one GPU each, in parallel if GPUs are free). One only:
# `sbatch --array=1 run_sample_owt_sweep.sh`, or without Slurm `CONFIG=1 sh run_sample_owt_sweep.sh`
# (CONFIG is the index or the name, e.g. CONFIG=ce_denoiser_sid).
# Override e.g. with `STEPS="32 64" sbatch run_sample_owt_sweep.sh`
# Results:
#   ${SWEEP_DIR}/<model>_<sampler>_<noise>_steps<n>/samples.json  (texts + metrics, one per run)
#   ${SWEEP_DIR}/summary.json  (metrics only, every run of every configuration found in SWEEP_DIR;
#   rebuilt after each run, by whichever task finishes it)
# A run whose samples.json already exists is skipped, so resubmitting the job resumes the sweep.

STEPS="${STEPS:-32 64 128 256}"        # space-separated numbers of sampling steps
BATCH_SIZE=8                           # 16 runs out of memory on a 24 GB GPU
NUM_BATCHES=64                         # 512 samples = NUM_BATCHES * BATCH_SIZE
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-8}"  # sequences scored at once by gpt2-large
NOISE="${NOISE:-log-linear}"
ENTROPY_EVERY="${ENTROPY_EVERY:-0.1}"  # entropy of xt and x0 every 10% of the steps (null to disable)
CKPT_DIR="${CKPT_DIR:-${PWD}/checkpoints/owt}"
SWEEP_DIR="${SWEEP_DIR:-${PWD}/eval_runs/owt/sweep}"

# index name checkpoint sampler x0_param sid_x0_param
CONFIGS="0 duo_sid_sidelbo duo sid elbo elbo
1 ce_denoiser_sid ce_denoiser sid posterior posterior
2 duo_ancestral duo ancestral elbo posterior"
CONFIG="${CONFIG:-${SLURM_ARRAY_TASK_ID:-}}"
[ -n "${CONFIG}" ] || { echo "Set CONFIG (0, 1, 2 or a name) or submit as an array job"; exit 1; }
SELECTED=$(echo "${CONFIGS}" | awk -v c="${CONFIG}" '$1 == c || $2 == c')
[ -n "${SELECTED}" ] || { echo "Unknown CONFIG ${CONFIG}"; exit 1; }

export WANDB_MODE=offline TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -f sfm.sif ]; then
  RUN="apptainer exec --nv sfm.sif python"
else
  RUN="${PYTHON:-python}"
fi

ckpt=$(echo "${SELECTED}" | awk '{print $3}')
[ -f "${CKPT_DIR}/${ckpt}.ckpt" ] || { echo "Missing checkpoint ${CKPT_DIR}/${ckpt}.ckpt"; exit 1; }
mkdir -p "${SWEEP_DIR}"

sample() {  # name ckpt sampler x0_param sid_x0_param steps output_dir
  ${RUN} -u -m main \
      mode=sample_eval \
      eval.checkpoint_path="${CKPT_DIR}/$2.ckpt" \
      eval.compute_generative_perplexity=True \
      eval.perplexity_batch_size="${PPL_BATCH_SIZE}" \
      eval.results_json_path="$7/samples.json" \
      eval.entropy_every="${ENTROPY_EVERY}" \
      data=openwebtext-split \
      data.cache_dir="${PWD}/data_cache" \
      model=small \
      algo=duo-base \
      sampler="$3" \
      ++sampler.x0_param="$4" \
      ++sampler.sid_x0_param="$5" \
      sampler.steps="$6" \
      sampler.num_sample_batches="${NUM_BATCHES}" \
      sampler.temperature=1 \
      noise="${NOISE}" \
      loader.eval_batch_size="${BATCH_SIZE}" \
      loader.num_workers=4 \
      trainer.num_nodes=1 \
      trainer.devices=1 \
      +wandb.offline=True \
      hydra.run.dir="$7" < /dev/null
}

summarize() {
  ${RUN} - "${SWEEP_DIR}" <<'EOF'
import glob, json, os, sys
sweep_dir = sys.argv[1]
runs = []
for path in sorted(glob.glob(os.path.join(sweep_dir, '*', 'samples.json'))):
  with open(path) as f:
    s = json.load(f)
  c = s['config']
  runs.append(dict(
    run=os.path.basename(os.path.dirname(path)),
    checkpoint=s['ckpt_name'],
    sampler=c['sampler']['predictor'],
    x0_param=c['sampler'].get('x0_param'),
    sid_x0_param=c['sampler'].get('sid_x0_param'),
    noise=c['noise']['type'],
    steps=c['sampler']['steps'],
    batch_size=c['loader']['eval_batch_size'],
    num_samples=len(s['text']),
    gen_ppl=s['gen_ppl_first_chunk_retok'],
    entropy=s['entropy'],
    avg_nfe=s['avg_nfe'],
    entropy_trajectory=s.get('entropy_trajectory')))
# The array tasks share the summary: write a temporary file and rename it (atomic), so a reader
# never sees a half-written file; the last task to finish a run rewrites it with every run.
tmp = os.path.join(sweep_dir, f'.summary.{os.getpid()}.json')
with open(tmp, 'w') as f:
  json.dump(runs, f, indent=2)
os.replace(tmp, os.path.join(sweep_dir, 'summary.json'))
for r in runs:
  print(f"{r['run']:45s} gen_ppl {r['gen_ppl']:9.2f}  entropy {r['entropy']:.3f}  n {r['num_samples']}")
EOF
}

echo "${SELECTED}" | while read -r index name ckpt sampler x0 sidx0; do
  for steps in ${STEPS}; do
    out="${SWEEP_DIR}/${name}_${NOISE}_steps${steps}"
    if [ -f "${out}/samples.json" ]; then
      echo "=== ${name} steps ${steps}: already done, skipped"
      continue
    fi
    echo "=== ${name} steps ${steps} (batch ${BATCH_SIZE} x ${NUM_BATCHES})"
    sample "${name}" "${ckpt}" "${sampler}" "${x0}" "${sidx0}" "${steps}" "${out}" \
      || echo "=== ${name} steps ${steps} failed, skipped"
    summarize
  done
done
