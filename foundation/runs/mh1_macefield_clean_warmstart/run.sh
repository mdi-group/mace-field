#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUN_ROOT}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES
IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
if [ "${#GPU_LIST[@]}" -ne 4 ]; then
  echo "Expected four visible GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun is required for the four-GPU launch" >&2
  exit 2
fi

CONFIG="${RUN_ROOT}/config/mh1_macefield_heads_strict.yaml"
CHECKPOINT="${RUN_ROOT}/checkpoints/mh1_macefield_clean_warmstart_run-123_epoch-35.pt"
REPLAY_TRAIN="foundation/data/mh1_true_replay_10000.xyz"
REPLAY_VALID="foundation/data/mh1_true_replay_10000_valid.xyz"
for required_path in "${CONFIG}" "${CHECKPOINT}" "${REPLAY_TRAIN}" "${REPLAY_VALID}"; do
  if [ ! -s "${required_path}" ]; then
    echo "Required warm-start input does not exist or is empty: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/models" "${RUN_ROOT}/results" "${RUN_ROOT}/logs" "${RUN_ROOT}/work"

# This is a separate tag and output root.  --restart_latest therefore loads
# only the copied epoch-35 warm-start checkpoint and cannot overwrite the
# paused production run's checkpoint, log, model, or result files.
exec torchrun \
  --standalone \
  --nproc_per_node=4 \
  --log-dir="${RUN_ROOT}/logs/torchrun" \
  --redirects=3 \
  --tee=3 \
  mace/cli/run_train.py \
  --distributed \
  --launcher=torchrun \
  --name=mh1_macefield_clean_warmstart \
  --model=MACEField \
  --loss=universal_field \
  --config="${CONFIG}" \
  --foundation_model=mh-1 \
  --foundation_head=omat_pbe \
  --E0s=estimated \
  --num_samples_pt=10000 \
  --force_mh_ft_lr=True \
  --lr="${MACE_WARMSTART_LR:-1e-4}" \
  --multiheads_finetuning=True \
  --pt_train_file="${REPLAY_TRAIN}" \
  --pt_valid_file="${REPLAY_VALID}" \
  --pseudolabel_replay=True \
  --pseudolabel_replay_compute_stress=True \
  --compute_forces=True \
  --compute_stress=True \
  --compute_polarization=True \
  --compute_becs=True \
  --compute_polarizability=True \
  --energy_weight=1.0 \
  --forces_weight=100.0 \
  --stress_weight=1.0 \
  --polarization_weight=10.0 \
  --becs_weight=100.0 \
  --polarizability_weight=1.0 \
  --polarizability_loss_mode=standardized_symmetric_huber \
  --polarizability_huber_delta=1.0 \
  --polarizability_scale_core_max_norm=1000.0 \
  --weight_decay=0.0 \
  --field_weight_decay=5e-7 \
  --default_dtype=float32 \
  --device=cuda \
  --batch_size=1 \
  --valid_batch_size=1 \
  --max_num_epochs="${MACE_WARMSTART_MAX_EPOCHS:-2048}" \
  --model_dir="${RUN_ROOT}/models" \
  --checkpoints_dir="${RUN_ROOT}/checkpoints" \
  --results_dir="${RUN_ROOT}/results" \
  --log_dir="${RUN_ROOT}/logs" \
  --work_dir="${RUN_ROOT}/work" \
  --clip_grad=1.0 \
  --ema_decay=0.9999 \
  --ema \
  --restart_latest \
  "$@"
