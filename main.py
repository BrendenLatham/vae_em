#!/usr/bin/env python3
"""
Entry point: VAE vs. Approximate-EM on MNIST and PBMC 10k RNA-seq.

Usage:
    python main.py --mode mnist
    python main.py --mode rnaseq
    python main.py --mode full

Each --mode wipes its own outputs/<mode>/ subtree at startup. The data
cache (--data-dir, default ./data) is treated as upstream input and is
never touched.
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch


def _resolve_device(arg):
    if arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if arg == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device cuda requested but CUDA is not available.')
    return arg


def _parse_args():
    p = argparse.ArgumentParser(
        description='VAE vs. Approximate-EM on MNIST and PBMC 10k RNA-seq.',
    )
    p.add_argument('--mode', required=True, choices=['mnist', 'rnaseq', 'full'],
                   help='Which experiment(s) to run.')
    p.add_argument('--output-dir', default='outputs',
                   help='Base output directory (default: outputs).')
    p.add_argument('--data-dir', default='./data',
                   help='Cache for raw datasets (MNIST, PBMC h5). Not wiped.')
    p.add_argument('--seed', type=int, default=42,
                   help='Seed for torch + numpy RNG (default: 42).')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                   help='Compute device (default: auto).')
    p.add_argument('--epochs-vae', type=int, default=30,
                   help='VAE training epochs (default: 30).')
    p.add_argument('--epochs-em', type=int, default=30,
                   help='Approximate-EM training epochs (default: 30).')
    p.add_argument('--langevin-steps', type=int, default=30,
                   help='Inner-loop Langevin steps in EM E-step (default: 30). '
                        'Also used as the training-cutoff line in the convergence plot.')
    p.add_argument('--K', type=int, default=1000,
                   help='Number of prior samples for log p(x) and ESS (default: 1000).')
    return p.parse_args()


def main():
    args = _parse_args()

    # Resolve device before anything else (so the log line is informative).
    args.device = _resolve_device(args.device)

    # Seed RNGs.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # Decide which subtrees to wipe.
    modes_to_run = ['mnist', 'rnaseq'] if args.mode == 'full' else [args.mode]

    # Lazy import after argparse so --help is fast and so we can wire up
    # logging before the experiment modules emit anything.
    from src import io_utils
    paths = io_utils.reset_mode_dirs(args.output_dir, modes_to_run)
    log_path = os.path.join(args.output_dir, 'run.log')
    io_utils.setup_logging(log_path)

    log = logging.getLogger('main')
    log.info(f'CLI args: {vars(args)}')
    log.info(f'Output directory: {os.path.abspath(args.output_dir)}')
    log.info(f'Data directory:   {os.path.abspath(args.data_dir)}')
    log.info(f'Modes to run: {modes_to_run}')
    log.info(f'Device: {args.device}')

    if 'mnist' in modes_to_run:
        from src.experiment_mnist import run_mnist
        run_mnist(args, paths['mnist'])

    if 'rnaseq' in modes_to_run:
        from src.experiment_rnaseq import run_rnaseq
        run_rnaseq(args, paths['rnaseq'])

    log.info('All requested experiments complete.')


if __name__ == '__main__':
    sys.exit(main())
