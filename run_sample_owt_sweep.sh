#!/bin/sh
#SBATCH --job-name owt-sweep
#SBATCH --error sbatch/error.e%A_%a
#SBATCH --output sbatch/out.o%A_%a
#SBATCH --array=0-4
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
#   ce_denoiser + ancestral_restart, LAM (ancestral pass on the CE output converted to x_theta,
#                 x0_param=posterior; sid after the restart on the CE output as is, sid_x0_param=posterior)
#   ce_denoiser + ancestral  (CE output converted to x_theta: x0_param=posterior; control of the restart)
# and every number of steps in STEPS. Each run decodes the last step twice from the same state:
# sampled (noise_removal=ancestral) and argmax (greedy), each scored on its own.
# Checkpoints: checkpoints/owt/duo.ckpt and checkpoints/owt/${CE_CKPT}.ckpt.
# Run from the repo root: `mkdir -p sbatch && sbatch run_sample_owt_sweep.sh` submits the
# configurations as array tasks 0, 1, 2, 3, 4 (one GPU each, in parallel if GPUs are free). One only:
# `sbatch --array=1 run_sample_owt_sweep.sh`; without Slurm, set CONFIG below (the index or the
# name, e.g. ce_denoiser_sid). Set the run in the settings below.
# Results:
#   ${SWEEP_DIR}/<model>_<sampler>_<noise>_steps<n>/samples.json  (texts + metrics, sampled last step;
#   <noise>: the schedule of the sampler, SID_NOISE for sid, ANC_NOISE for ancestral and
#   ANC_NOISE+SID_NOISE for ancestral_restart, as log-linear / cosine-p<power>)
#   ${SWEEP_DIR}/<model>_<sampler>_<noise>_steps<n>/samples_greedy.json  (same, argmax last step)
#   ${SWEEP_DIR}/summary.json  (metrics only, both last steps of every run found in SWEEP_DIR;
#   rebuilt after each run, by whichever task finishes it)
# A run whose samples.json and samples_greedy.json already exist is skipped, so resubmitting the
# job resumes the sweep.

STEPS="32 64 128 256"        # space-separated numbers of sampling steps
BATCH_SIZE=8                           # 16 runs out of memory on a 24 GB GPU
NUM_BATCHES=64                         # 512 samples = NUM_BATCHES * BATCH_SIZE
PPL_BATCH_SIZE="8"  # sequences scored at once by gpt2-large
ANC_NOISE="log-linear"      # schedule of the ancestral runs and of the ancestral pass of ancestral_restart: log-linear / cosine
ANC_NOISE_POWER="2"   # cosine only: alpha_t = cos(pi/2 t)^power (< 2 more steps near the data, > 2 near the noise)
SID_NOISE="cosine"          # schedule of the sid runs and of the sid steps after the restart: log-linear / cosine
SID_NOISE_POWER="0.5" # cosine only
LAM="0.8"                    # ancestral_restart: proportion of the steps of the ancestral pass and restart noise level alpha
sched_name() { if [ "$1" = "cosine" ]; then echo "cosine-p$2"; else echo "$1"; fi; }
ENTROPY_EVERY="0.1"  # entropy of xt and x0 every 10% of the steps (null to disable)
CKPT_DIR="${PWD}/checkpoints/owt"
CE_CKPT="ce_denoiser_step22000"           # cross-entropy checkpoint (file name in CKPT_DIR, without .ckpt); ce_denoiser: step 40000
SWEEP_DIR="${PWD}/eval_runs/owt/sweep_ce_step22000"  # one folder (and summary.json) per CE checkpoint; eval_runs/owt/sweep: step 40000

# index name checkpoint sampler x0_param sid_x0_param
# ancestral_restart restarts with sid (restart_sampler=sid) at LAM; the run names get _lam<LAM>
CONFIGS="0 duo_sid_sidelbo duo sid elbo elbo
1 ce_denoiser_sid ${CE_CKPT} sid posterior posterior
2 duo_ancestral duo ancestral elbo posterior
3 ce_denoiser_restart ${CE_CKPT} ancestral_restart posterior posterior
4 ce_denoiser_ancestral ${CE_CKPT} ancestral posterior posterior"
CONFIG="${SLURM_ARRAY_TASK_ID}"        # the array task; without Slurm, set the index or the name here
[ -n "${CONFIG}" ] || { echo "Set CONFIG (0, 1, 2, 3, 4 or a name) or submit as an array job"; exit 1; }
SELECTED=$(echo "${CONFIGS}" | awk -v c="${CONFIG}" '$1 == c || $2 == c')
[ -n "${SELECTED}" ] || { echo "Unknown CONFIG ${CONFIG}"; exit 1; }

export WANDB_MODE=offline TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -f sfm.sif ]; then
  RUN="apptainer exec --nv sfm.sif python"
else
  RUN="python"  # the active python, e.g. after `conda activate sfm`
fi

ckpt=$(echo "${SELECTED}" | awk '{print $3}')
[ -f "${CKPT_DIR}/${ckpt}.ckpt" ] || { echo "Missing checkpoint ${CKPT_DIR}/${ckpt}.ckpt"; exit 1; }
mkdir -p "${SWEEP_DIR}"

sample() {  # name ckpt sampler x0_param sid_x0_param steps output_dir
  RESTART_ARGS=""
  [ "$3" = "ancestral_restart" ] && RESTART_ARGS="sampler.lam=${LAM} ++sampler.restart_sampler=sid"
  ${RUN} -u -m main \
      mode=sample_eval \
      eval.checkpoint_path="${CKPT_DIR}/$2.ckpt" \
      eval.compute_generative_perplexity=True \
      eval.perplexity_batch_size="${PPL_BATCH_SIZE}" \
      eval.results_json_path="$7/samples.json" \
      eval.entropy_every="${ENTROPY_EVERY}" \
      ++eval.also_greedy=True \
      data=openwebtext-split \
      data.cache_dir="${PWD}/data_cache" \
      model=small \
      algo=duo-base \
      sampler="$3" ${RESTART_ARGS} \
      ++sampler.x0_param="$4" \
      ++sampler.sid_x0_param="$5" \
      sampler.steps="$6" \
      sampler.num_sample_batches="${NUM_BATCHES}" \
      sampler.temperature=1 \
      ++sampler.ancestral_noise.type="${ANC_NOISE}" \
      ++sampler.ancestral_noise.power="${ANC_NOISE_POWER}" \
      ++sampler.sid_noise.type="${SID_NOISE}" \
      ++sampler.sid_noise.power="${SID_NOISE_POWER}" \
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
for path in sorted(glob.glob(os.path.join(sweep_dir, '*', 'samples*.json'))):
  with open(path) as f:
    s = json.load(f)
  c = s['config']
  # The schedule of the sampler's kernel; runs made before the per-kernel schedules used noise
  key = 'sid_noise' if c['sampler']['predictor'] == 'sid' else 'ancestral_noise'
  noise = c['sampler'].get(key) or c['noise']
  noise_type = 'cosine' if noise['type'] == 'cosine-squared' else noise['type']
  restart = c['sampler']['predictor'] == 'ancestral_restart'
  sid_noise = c['sampler'].get('sid_noise') if restart else None
  runs.append(dict(
    run=os.path.basename(os.path.dirname(path)),
    last_step='greedy' if path.endswith('_greedy.json') else 'sample',
    checkpoint=s['ckpt_name'],
    sampler=c['sampler']['predictor'],
    x0_param=c['sampler'].get('x0_param'),
    sid_x0_param=c['sampler'].get('sid_x0_param'),
    noise=noise_type,
    noise_power=noise.get('power') if noise_type == 'cosine' else None,
    # ancestral_restart: noise / noise_power are the ancestral pass, sid_noise the steps after the restart
    lam=c['sampler'].get('lam') if restart else None,
    restart_sampler=c['sampler'].get('restart_sampler') if restart else None,
    sid_noise=sid_noise['type'] if sid_noise else None,
    sid_noise_power=sid_noise.get('power') if sid_noise and sid_noise['type'] == 'cosine' else None,
    steps=c['sampler']['steps'],
    batch_size=c['loader']['eval_batch_size'],
    num_samples=len(s['text']),
    gen_ppl=s['gen_ppl_first_chunk_retok'],
    entropy=s['entropy'],
    frac_tokens_changed=s.get('frac_tokens_changed'),  # greedy: vs the sampled last step
    avg_nfe=s['avg_nfe'],
    # Clock time of the sampling (seconds, same for both last steps; runs before the timing: None)
    sampling_time_s=s.get('timing', {}).get('sampling_time_s'),
    batch_time_mean_without_first_s=s.get('timing', {}).get('batch_time_mean_without_first_s'),
    gpu_name=s.get('timing', {}).get('gpu_name'),
    hostname=s.get('timing', {}).get('hostname'),
    entropy_trajectory=s.get('entropy_trajectory')))
# The array tasks share the summary: write a temporary file and rename it (atomic), so a reader
# never sees a half-written file; the last task to finish a run rewrites it with every run.
tmp = os.path.join(sweep_dir, f'.summary.{os.getpid()}.json')
with open(tmp, 'w') as f:
  json.dump(runs, f, indent=2)
os.replace(tmp, os.path.join(sweep_dir, 'summary.json'))
for r in runs:
  print(f"{r['run']:45s} {r['last_step']:6s} gen_ppl {r['gen_ppl']:9.2f}  entropy {r['entropy']:.3f}  n {r['num_samples']}")
EOF
}

echo "${SELECTED}" | while read -r index name ckpt sampler x0 sidx0; do
  [ "${sampler}" = "ancestral_restart" ] && name="${name}_lam${LAM}"
  for steps in ${STEPS}; do
    case "${sampler}" in
      sid) noise_name=$(sched_name "${SID_NOISE}" "${SID_NOISE_POWER}") ;;
      ancestral_restart) noise_name="$(sched_name "${ANC_NOISE}" "${ANC_NOISE_POWER}")+$(sched_name "${SID_NOISE}" "${SID_NOISE_POWER}")" ;;
      *) noise_name=$(sched_name "${ANC_NOISE}" "${ANC_NOISE_POWER}") ;;
    esac
    out="${SWEEP_DIR}/${name}_${noise_name}_steps${steps}"
    if [ -f "${out}/samples.json" ] && [ -f "${out}/samples_greedy.json" ]; then
      echo "=== ${name} steps ${steps}: already done, skipped"
      continue
    fi
    echo "=== ${name} steps ${steps} (batch ${BATCH_SIZE} x ${NUM_BATCHES})"
    sample "${name}" "${ckpt}" "${sampler}" "${x0}" "${sidx0}" "${steps}" "${out}" \
      || echo "=== ${name} steps ${steps} failed, skipped"
    summarize
  done
done
