#!/bin/sh
#SBATCH --job-name tinygsm-sweep
#SBATCH --error sbatch/error.e%A_%a
#SBATCH --output sbatch/out.o%A_%a
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --mem=32000
#SBATCH --partition public-gpu,shared-gpu

#SBATCH --time 9:00:00

# Sweep of the TinyGSM sampling space with the pretrained DUO, one configuration per array task:
# GSM8K test accuracy (1319 problems, one answer each), for
#   duo + sid                (DUO x_theta converted to p(x0 | xt): sid_x0_param=elbo)
#   duo + ancestral          (DUO x_theta used as is: x0_param=elbo)
#   duo + ancestral_restart, LAM (ancestral pass on x_theta as is, x0_param=elbo; sid after the
#                            restart on x_theta converted to p(x0 | xt), sid_x0_param=elbo)
# and every number of steps in STEPS.
# Batch size: each run tries BATCH_SIZE first and halves it when the GPU runs out of memory
# (down to 1); the batch size that fits is kept for the next runs of the task.
# Checkpoint: checkpoints/tinygsm/duo.ckpt, download it with
#   wget -c -O checkpoints/tinygsm/duo.ckpt https://huggingface.co/jdeschena/s-flm/resolve/main/tinygsm/duo.ckpt
# and build the GSM8K test file once (needs internet):
#   apptainer exec sfm.sif python scripts/prepare_gsm8k_test.py data/gsm8k_test.json
# Run from the repo root: `mkdir -p sbatch && sbatch run_sample_tinygsm_sweep.sh` submits the
# configurations as array tasks 0, 1, 2 (one GPU each, in parallel if GPUs are free). One only:
# `sbatch --array=1 run_sample_tinygsm_sweep.sh`; without Slurm, set CONFIG below (the index or
# the name, e.g. duo_sid_sidelbo). Set the run in the settings below.
# Results:
#   ${SWEEP_DIR}/<model>_<sampler>_<noise>[_T<temperature>]_steps<n>/results.json  (accuracy, config,
#   timing and every answer; <noise>: the schedule of the sampler, SID_NOISE for sid, ANC_NOISE for
#   ancestral and ANC_NOISE+SID_NOISE for ancestral_restart, as log-linear / cosine-p<power>)
#   (_T<temperature> only when TEMPERATURE is not 1, so each temperature has its own runs)
#   ${SWEEP_DIR}/summary.json  (metrics only, every run found in SWEEP_DIR; rebuilt after each run,
#   by whichever task finishes it)
# A run whose results.json already exists is skipped, so resubmitting the job resumes the sweep.

STEPS="32 64 128 256 512"        # space-separated numbers of sampling steps
BATCH_SIZE="32"           # first batch size tried; halved on out of memory
ANC_NOISE="log-linear"      # schedule of the ancestral runs and of the ancestral pass of ancestral_restart: log-linear / cosine
ANC_NOISE_POWER="2"   # cosine only: alpha_t = cos(pi/2 t)^power (< 2 more steps near the data, > 2 near the noise)
SID_NOISE="cosine"          # schedule of the sid runs and of the sid steps after the restart: log-linear / cosine
SID_NOISE_POWER="0.5" # cosine only
TEMPERATURE="1"            # sampling temperature (all steps); the run names get _T<TEMPERATURE> when it is not 1
LAM="0.8"                    # ancestral_restart: proportion of the steps of the ancestral pass and restart noise level alpha
sched_name() { if [ "$1" = "cosine" ]; then echo "cosine-p$2"; else echo "$1"; fi; }
CKPT_PATH="${PWD}/checkpoints/tinygsm/duo.ckpt"
GSM8K_PATH="${PWD}/data/gsm8k_test.json"
SWEEP_DIR="${PWD}/eval_runs/tinygsm/sweep"

# index name sampler x0_param sid_x0_param
# DUO only: the sid steps always convert x_theta (sid_x0_param=elbo), the ancestral steps never
# (x0_param=elbo); ancestral_restart restarts with sid (restart_sampler=sid) at LAM, the run names get _lam<LAM>
CONFIGS="0 duo_sid_sidelbo sid elbo elbo
1 duo_ancestral ancestral elbo elbo
2 duo_restart ancestral_restart elbo elbo"
CONFIG="${SLURM_ARRAY_TASK_ID}"        # the array task; without Slurm, set the index or the name here
[ -n "${CONFIG}" ] || { echo "Set CONFIG (0, 1, 2 or a name) or submit as an array job"; exit 1; }
SELECTED=$(echo "${CONFIGS}" | awk -v c="${CONFIG}" '$1 == c || $2 == c')
[ -n "${SELECTED}" ] || { echo "Unknown CONFIG ${CONFIG}"; exit 1; }

export WANDB_MODE=offline TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -f sfm.sif ]; then
  RUN="apptainer exec --nv sfm.sif python"
else
  RUN="python"  # the active python, e.g. after `conda activate sfm`
fi

[ -f "${CKPT_PATH}" ] || { echo "Missing checkpoint ${CKPT_PATH}"; exit 1; }
[ -f "${GSM8K_PATH}" ] || { echo "Missing GSM8K test file ${GSM8K_PATH}"; exit 1; }
mkdir -p "${SWEEP_DIR}"

sample() {  # sampler x0_param sid_x0_param steps output_dir batch_size
  RESTART_ARGS=""
  [ "$1" = "ancestral_restart" ] && RESTART_ARGS="sampler.lam=${LAM} ++sampler.restart_sampler=sid"
  mkdir -p "$5"
  # stderr also goes to a file, to tell an out of memory from another failure
  ${RUN} -u -m main \
      mode=gsm8k_eval \
      eval.checkpoint_path="${CKPT_PATH}" \
      data=gsm8k-test \
      data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
      data.cache_dir="${PWD}/data_cache" \
      data.data_path="${GSM8K_PATH}" \
      model=small \
      model.length=512 \
      algo=duo-base \
      sampler="$1" ${RESTART_ARGS} \
      ++sampler.x0_param="$2" \
      ++sampler.sid_x0_param="$3" \
      sampler.steps="$4" \
      sampler.temperature="${TEMPERATURE}" \
      ++sampler.ancestral_noise.type="${ANC_NOISE}" \
      ++sampler.ancestral_noise.power="${ANC_NOISE_POWER}" \
      ++sampler.sid_noise.type="${SID_NOISE}" \
      ++sampler.sid_noise.power="${SID_NOISE_POWER}" \
      loader.eval_batch_size="$6" \
      loader.num_workers=4 \
      trainer.num_nodes=1 \
      trainer.devices=1 \
      gsm8k.output_dir="$5" \
      +wandb.offline=True \
      hydra.run.dir="$5" < /dev/null 2> "$5/stderr.log"
  status=$?
  cat "$5/stderr.log" >&2
  return ${status}
}

summarize() {
  ${RUN} - "${SWEEP_DIR}" <<'EOF'
import glob, json, os, sys
sweep_dir = sys.argv[1]
runs = []
for path in sorted(glob.glob(os.path.join(sweep_dir, '*', 'results.json'))):
  with open(path) as f:
    s = json.load(f)
  c = s.get('config')
  if c is None:  # results.json written before the config was saved
    continue
  # The schedule of the sampler's kernel
  key = 'sid_noise' if c['sampler']['predictor'] == 'sid' else 'ancestral_noise'
  noise = c['sampler'].get(key) or c['noise']
  restart = c['sampler']['predictor'] == 'ancestral_restart'
  sid_noise = c['sampler'].get('sid_noise') if restart else None
  timing = s.get('timing', {})
  runs.append(dict(
    run=os.path.basename(os.path.dirname(path)),
    checkpoint=s['ckpt_name'],
    sampler=c['sampler']['predictor'],
    x0_param=c['sampler'].get('x0_param'),
    sid_x0_param=c['sampler'].get('sid_x0_param'),
    noise=noise['type'],
    noise_power=noise.get('power') if noise['type'] == 'cosine' else None,
    # ancestral_restart: noise / noise_power are the ancestral pass, sid_noise the steps after the restart
    lam=c['sampler'].get('lam') if restart else None,
    restart_sampler=c['sampler'].get('restart_sampler') if restart else None,
    sid_noise=sid_noise['type'] if sid_noise else None,
    sid_noise_power=sid_noise.get('power') if sid_noise and sid_noise['type'] == 'cosine' else None,
    steps=c['sampler']['steps'],
    temperature=c['sampler']['temperature'],
    batch_size=c['loader']['eval_batch_size'],
    accuracy=s['accuracy'],
    num_correct=s['num_correct'],
    num_total=s['num_total'],
    avg_nfe=s.get('avg_nfe'),
    # Clock time of the sampling (seconds)
    sampling_time_s=timing.get('sampling_time_s'),
    batch_time_mean_without_first_s=timing.get('batch_time_mean_without_first_s'),
    # per problem, to compare runs with different batch sizes
    sample_time_s=(timing['batch_time_mean_without_first_s'] / timing['batch_size']
                   if timing.get('batch_time_mean_without_first_s') else None),
    gpu_name=timing.get('gpu_name'),
    hostname=timing.get('hostname')))
# The array tasks share the summary: write a temporary file and rename it (atomic), so a reader
# never sees a half-written file; the last task to finish a run rewrites it with every run.
tmp = os.path.join(sweep_dir, f'.summary.{os.getpid()}.json')
with open(tmp, 'w') as f:
  json.dump(runs, f, indent=2)
os.replace(tmp, os.path.join(sweep_dir, 'summary.json'))
for r in runs:
  print(f"{r['run']:50s} accuracy {100 * r['accuracy']:6.2f}%  ({r['num_correct']}/{r['num_total']})")
EOF
}

echo "${SELECTED}" | while read -r index name sampler x0 sidx0; do
  [ "${sampler}" = "ancestral_restart" ] && name="${name}_lam${LAM}"
  for steps in ${STEPS}; do
    case "${sampler}" in
      sid) noise_name=$(sched_name "${SID_NOISE}" "${SID_NOISE_POWER}") ;;
      ancestral_restart) noise_name="$(sched_name "${ANC_NOISE}" "${ANC_NOISE_POWER}")+$(sched_name "${SID_NOISE}" "${SID_NOISE_POWER}")" ;;
      *) noise_name=$(sched_name "${ANC_NOISE}" "${ANC_NOISE_POWER}") ;;
    esac
    temp_name=""
    case "${TEMPERATURE}" in 1|1.0|1.00) ;; *) temp_name="_T${TEMPERATURE}" ;; esac
    out="${SWEEP_DIR}/${name}_${noise_name}${temp_name}_steps${steps}"
    if [ -f "${out}/results.json" ]; then
      echo "=== ${name} steps ${steps}: already done (${out}), skipped"
      continue
    fi
    # Halve the batch on out of memory; the batch that fits is kept for the next runs
    while :; do
      echo "=== ${name} steps ${steps} (batch ${BATCH_SIZE})"
      if sample "${sampler}" "${x0}" "${sidx0}" "${steps}" "${out}" "${BATCH_SIZE}"; then
        break
      fi
      if grep -qi "out of memory" "${out}/stderr.log" && [ "${BATCH_SIZE}" -gt 1 ]; then
        BATCH_SIZE=$((BATCH_SIZE / 2))
        echo "=== ${name} steps ${steps}: out of memory, retrying with batch ${BATCH_SIZE}"
      else
        echo "=== ${name} steps ${steps} failed (see ${out}/stderr.log), skipped"
        break
      fi
    done
    summarize
  done
done
