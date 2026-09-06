#!/usr/bin/env bash
# The paper's main comparison: 7 PDEs x 5 schemes x 5 seeds, with the seed-1
# solution fields.
#
#   GPUS="0,1,2"  comma-separated device ids to shard over   (default: 0)
#   JOBS=4        concurrent runs per device                 (default: 4)
#
# Usage:  GPUS="0,1,2" JOBS=4 bash release/experiments/reproduce.sh
# Finished runs are skipped, so re-running resumes an interrupted sweep.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-python}
# swap in a runner that also dumps the frozen system for the analyses
RUNNER=${RUNNER:-release/experiments/run.py}
GPUS=${GPUS:-0}
JOBS=${JOBS:-4}
IFS=',' read -ra DEV <<< "$GPUS"
NDEV=${#DEV[@]}
SLOTS=$(( NDEV * JOBS ))
mkdir -p results

PDES="klein_gordon helmholtz heat convection_diffusion burgers advection allen_cahn"
METHODS="vanilla ntk_trace lr_anneal sa_pinn relobralo"

i=0
for seed in 1 0 2 3 4; do                       # seed 1 first: it carries the field dumps
  for pde in $PDES; do
    for method in $METHODS; do
      dev=${DEV[$(( i % NDEV ))]}
      extra=""; [ "$seed" -eq 1 ] && extra="--field"
      CUDA_VISIBLE_DEVICES=$dev XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 \
        $PY $RUNNER --pde "$pde" --method "$method" --seed "$seed" \
            $extra >> results/reproduce.log 2>&1 &
      i=$(( i + 1 ))
      # refill a slot as soon as any run ends; runs differ by 4x in length, so a
      # wait-for-all barrier would leave most of the pool idle
      while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do wait -n; done
    done
  done
done
wait
echo "runs $(ls results/runs/*.json | wc -l)/175," \
     "fields $(ls results/fields/*.npz | wc -l)/35"
