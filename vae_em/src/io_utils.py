"""
I/O helpers: output directory reset, structured logging, CSV/JSON/npz
writers, and model checkpointing.

Reset policy: when --mode foo is run, only outputs/foo/ is wiped and
recreated. The top-level run.log is always overwritten. The data cache
(MNIST raw files, PBMC h5) is never touched -- it is upstream input.
"""

import csv
import json
import logging
import os
import shutil

import numpy as np
import torch

log = logging.getLogger(__name__)


def reset_mode_dirs(base_output_dir, mode):
    """Wipe and recreate outputs/<mode>/{plots,metrics,checkpoints} for every
    mode in the given list. `mode` may be a single string or a list of strings.
    Returns a dict mapping mode_name -> {'plots': path, 'metrics': path, 'checkpoints': path}.
    """
    if isinstance(mode, str):
        modes = [mode]
    else:
        modes = list(mode)

    paths = {}
    for m in modes:
        mode_dir = os.path.join(base_output_dir, m)
        if os.path.isdir(mode_dir):
            shutil.rmtree(mode_dir)
        sub = {}
        for sub_name in ('plots', 'metrics', 'checkpoints'):
            p = os.path.join(mode_dir, sub_name)
            os.makedirs(p, exist_ok=False)
            sub[sub_name] = p
        paths[m] = sub
    return paths


def setup_logging(log_path, level=logging.INFO):
    """Root logger writes to both stdout and `log_path`. Truncates the file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Truncate the log file.
    open(log_path, 'w').close()

    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    root = logging.getLogger()
    # Clear any handlers from earlier runs / libraries.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)


def save_history_csv(history, path):
    """history: dict of (column_name -> list of floats), all same length."""
    keys = list(history.keys())
    n = len(history[keys[0]])
    for k in keys:
        if len(history[k]) != n:
            raise ValueError(f'history columns differ in length: {[(k, len(v)) for k, v in history.items()]}')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch'] + keys)
        for i in range(n):
            w.writerow([i + 1] + [history[k][i] for k in keys])
    log.info(f'  wrote metrics: {path}')


def save_metrics_json(metrics, path):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    log.info(f'  wrote metrics: {path}')


def save_latents_npz(path, **arrays):
    """Save named numpy arrays in a single .npz file."""
    np.savez(path, **arrays)
    log.info(f'  wrote latents: {path}')


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    log.info(f'  wrote checkpoint: {path}')
