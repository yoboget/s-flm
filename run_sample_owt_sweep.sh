#!/bin/sh
#SBATCH --job-name owt-sweep
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
# Any GPU but the Blackwell ones (RTX PRO 6000, sm_120): the image's torch 2.7 cu126 has no
# kernels for them. On Bamboo, Ampere means an RTX 3090 (24 GB) or an A100.
# On Baobab, replace the two lines below with e.g. `#SBATCH --gres=gpu:nvidia_rtx_a5000:1`.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --mem=32000
#SBATCH --partition public-gpu,shared-gpu

#SBATCH --time 12:00:00

# Sweep of the OWT sampling space in one job: unconditional samples scored by generative
# perplexity (gpt2-large) and sample entropy, for
#   duo         + sid        (DUO x_theta converted to p(x0 | xt): sid_x0_param=elbo)
#   ce_denoiser + sid        (cross-entropy model, its output is already p(x0 | xt))
#   duo         + ancestral  (DUO x_theta used as is: x0_param=elbo)
# and every number of steps in STEPS.
# Checkpoints: checkpoints/owt/duo.ckpt and checkpoints/owt/ce_denoiser.ckpt.
# Run from the repo root: `mkdir -p sbatch && sbatch run_sample_owt_sweep.sh`
# Override e.g. with `STEPS="32 64" BATCH_SIZE=8 NUM_BATCHES=64 sbatch run_sample_owt_sweep.sh`
# Results:
#   ${SWEEP_DIR}/<model>_<sampler>_<noise>_steps<n>/samples.json  (texts + metrics, one per run)
#   ${SWEEP_DIR}/summary.json  (metrics only, every run; rewritten after each run)
# A run whose samples.json already exists is skipped, so resubmitting the job resumes the sweep.
# If a run fails (e.g. out of GPU memory), it is retried once with half the batch size and twice
# the number of batches (same number of samples); the batch size used is in the summary.

STEPS="${STEPS:-32 64 128 256}"        # space-separated numbers of sampling steps
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_BATCHES="${NUM_BATCHES:-32}"       # total samples = NUM_BATCHES * BATCH_SIZE
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-8}"  # sequences scored at once by gpt2-large
NOISE="${NOISE:-log-linear}"
ENTROPY_EVERY="${ENTROPY_EVERY:-0.1}"  # entropy of xt and x0 every 10% of the steps (null to disable)
CKPT_DIR="${CKPT_DIR:-${PWD}/checkpoints/owt}"
SWEEP_DIR="${SWEEP_DIR:-${PWD}/eval_runs/owt/sweep}"

# name checkpoint sampler x0_param sid_x0_param
CONFIGS="duo_sid_sidelbo duo sid elbo elbo
ce_denoiser_sid ce_denoiser sid posterior posterior
duo_ancestral duo ancestral elbo posterior"

export WANDB_MODE=offline TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -f sfm.sif ]; then
  RUN="apptainer exec --nv sfm.sif python"
else
  RUN="${PYTHON:-python}"
fi

for ckpt in duo ce_denoiser; do
  [ -f "${CKPT_DIR}/${ckpt}.ckpt" ] || { echo "Missing checkpoint ${CKPT_DIR}/${ckpt}.ckpt"; exit 1; }
done
mkdir -p "${SWEEP_DIR}"

sample() {  # name ckpt sampler x0_param sid_x0_param steps output_dir batch_size num_batches
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
      sampler.num_sample_batches="$9" \
      sampler.temperature=1 \
      noise="${NOISE}" \
      loader.eval_batch_size="$8" \
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
with open(os.path.join(sweep_dir, 'summary.json'), 'w') as f:
  json.dump(runs, f, indent=2)
for r in runs:
  print(f"{r['run']:45s} gen_ppl {r['gen_ppl']:9.2f}  entropy {r['entropy']:.3f}  n {r['num_samples']}")
EOF
}

echo "${CONFIGS}" | while read -r name ckpt sampler x0 sidx0; do
  for steps in ${STEPS}; do
    out="${SWEEP_DIR}/${name}_${NOISE}_steps${steps}"
    if [ -f "${out}/samples.json" ]; then
      echo "=== ${name} steps ${steps}: already done, skipped"
      continue
    fi
    echo "=== ${name} steps ${steps} (batch ${BATCH_SIZE} x ${NUM_BATCHES})"
    if ! sample "${name}" "${ckpt}" "${sampler}" "${x0}" "${sidx0}" "${steps}" "${out}" \
        "${BATCH_SIZE}" "${NUM_BATCHES}"; then
      half=$((BATCH_SIZE / 2))
      echo "=== ${name} steps ${steps} failed, retrying with batch ${half} x $((NUM_BATCHES * 2))"
      sample "${name}" "${ckpt}" "${sampler}" "${x0}" "${sidx0}" "${steps}" "${out}" \
        "${half}" "$((NUM_BATCHES * 2))" || echo "=== ${name} steps ${steps} failed again, skipped"
    fi
    summarize
  done
done
