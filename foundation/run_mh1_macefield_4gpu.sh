#!/usr/bin/env bash
set -euo pipefail

# Run from any directory: all paths are resolved against this checkout.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# Keep allocator fragmentation from consuming the headroom needed by the
# response derivatives.  Respect an explicitly supplied allocator setting.
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

mkdir -p foundation/logs foundation/models foundation/results

# MACE-MH-1 is a plain MACE foundation model.  It supplies replay E/F/stress
# pseudolabels; field labels come from the real field-capable heads above.
torchrun \
  --standalone \
  --nproc_per_node=4 \
  --log-dir=foundation/logs/torchrun \
  --redirects=3 \
  --tee=3 \
  mace/cli/run_train.py \
  --distributed \
  --launcher=torchrun \
  --name=mh1_macefield \
  --model=MACEField \
  --loss=universal_field \
  --config=foundation/configs/mh1_macefield_heads.yaml \
  --foundation_model=mh-1 \
  --foundation_head=omat_pbe \
  --E0s=foundation \
  --multiheads_finetuning=True \
  --pt_train_file="${MACE_REPLAY_TRAIN_FILE:-foundation/data/mh1_replay.xyz}" \
  --pt_valid_file="${MACE_REPLAY_VALID_FILE:-foundation/data/mh1_replay_valid.xyz}" \
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
  --polarizability_weight=10.0 \
  --default_dtype=float64 \
  --device=cuda \
  --batch_size="${MACE_BATCH_SIZE:-2}" \
  --valid_batch_size="${MACE_VALID_BATCH_SIZE:-1}" \
  --macefield_max_atoms_per_batch="${MACE_MAX_ATOMS_PER_BATCH:-288}" \
  --max_num_updates="${MACE_MAX_UPDATES:-0}" \
  --max_num_epochs="${MACE_MAX_EPOCHS:-2048}" \
  --model_dir=foundation/models \
  --checkpoints_dir=foundation/models \
  --results_dir=foundation/results \
  --log_dir=foundation/logs \
  --work_dir=foundation \
  "$@"
