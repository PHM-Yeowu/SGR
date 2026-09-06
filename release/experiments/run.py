#!/usr/bin/env python3
"""Run one (pde, method, seed) and write results/runs/{pde}_{method}_seed{seed}.json.

    python release/experiments/run.py --pde helmholtz --method vanilla --seed 0
"""
# DETERMINISTIC
import os, sys
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_deterministic_ops=true'
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # release/
sys.path.insert(0, os.path.join(_R, 'src'))
os.chdir(os.path.join(_R, '..'))                                          # repository root
import argparse
import json

import jax
jax.config.update('jax_default_matmul_precision', 'highest')

import pdes
import train

METHODS = ['vanilla', 'ntk_trace', 'lr_anneal', 'sa_pinn', 'relobralo']


def execute(pde, method, seed=0, iters=30000, coef=None, field=False, also=None):
    """Train one configuration and write its record.

    also(state, result, tag) runs on the trained state before the record commits,
    so an analysis can hang its own dump off it; on a resume it is probed as
    also(None, None, tag) and answers False when that dump is missing. Returns the
    tag, or None when the run was already on disk.
    """
    name = pdes.short(pde)
    tag = f'{name}_{method}_seed{seed}'
    os.makedirs('results/runs', exist_ok=True)
    out = f'results/runs/{tag}.json'
    # the record is written last, so it doubles as the commit sentinel; a resume
    # needs every artifact this invocation asks for, not the record alone, or an
    # earlier plain run would suppress the dumps a later sweep needs
    if os.path.exists(out) and (not field or os.path.exists(f'results/fields/{tag}.npz')):
        if also is None or also(None, None, tag) is not False:
            print(f'skip (exists): {out}')
            return None

    if field:
        os.makedirs('results/fields', exist_ok=True)
    field_path = f'results/fields/{tag}.npz' if field else None
    result, state = train.run(pde, method, seed=seed, iters=iters, coef=coef,
                              dump_field=field_path, return_state=True)
    if also is not None:
        also(state, result, tag)

    # via .tmp so a killed sweep never leaves a half-written record behind
    with open(out + '.tmp', 'w') as f:
        json.dump(result, f, indent=1)
    os.replace(out + '.tmp', out)
    print(f"{name} {method} seed{seed}: "
          f"pre {result['pre_L2']:.3e} -> post {result['post_L2']:.3e}  ({out})")
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pde', required=True, choices=pdes.PDES + list(pdes.ALIAS))
    ap.add_argument('--method', required=True, choices=METHODS)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iters', type=int, default=30000)
    ap.add_argument('--coef', type=float, default=None)
    ap.add_argument('--field', action='store_true',
                    help='also dump the before/after solution fields')
    args = ap.parse_args()
    execute(args.pde, args.method, args.seed, args.iters, args.coef, args.field)


if __name__ == '__main__':
    main()
