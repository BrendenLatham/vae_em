"""
Orchestrator for the PBMC 10k RNA-seq experiment.

Sequence:
  1. Load data (downloads + scanpy preprocessing on first run).
  2. Train VAE on the log-normalized HVG matrix; save history + checkpoint
     + training curve.
  3. Deepcopy VAE -> EM; train; save history + checkpoint.
  4. Extract latents (encoder mean for VAE, Langevin posterior for EM),
     run t-SNE, build a true-vs-recon scatter on a single test cell,
     save combined plot + npz.
  5. Estimate log p(x) on test for both; save JSON.
"""

import copy
import logging
import os

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.manifold import TSNE

from .model import VAE
from .training import train_vae, train_em
from .evaluation import (
    estimate_marginal_log_likelihood,
    extract_latents,
)
from .langevin import langevin_sample
from .data_rnaseq import get_rnaseq_loaders
from . import plotting
from . import io_utils

log = logging.getLogger(__name__)


def run_rnaseq(args, paths):
    """Args is the parsed argparse Namespace. paths is the dict from
    reset_mode_dirs for the 'rnaseq' subtree."""
    device = args.device

    log.info('=' * 60)
    log.info('RNA-seq experiment (PBMC 10k)')
    log.info('=' * 60)

    # 1. Data.
    train_loader, test_loader, n_genes = get_rnaseq_loaders(
        data_dir=args.data_dir, batch_size=128, seed=args.seed,
    )

    # 2. VAE.
    vae = VAE(input_dim=n_genes, latent_dim=10, hidden_dim=400).to(device)
    vae_history = train_vae(vae, train_loader, device,
                            epochs=args.epochs_vae, lr=1e-3)
    io_utils.save_history_csv(
        vae_history, os.path.join(paths['metrics'], 'vae_history.csv'))
    io_utils.save_checkpoint(
        vae, os.path.join(paths['checkpoints'], 'vae.pt'))

    # 3. EM, warm-started from VAE.
    em = copy.deepcopy(vae)
    em_history = train_em(em, train_loader, device,
                          epochs=args.epochs_em, lr=1e-4,
                          langevin_steps=args.langevin_steps,
                          langevin_lr=0.005, m_steps=args.m_steps)
    io_utils.save_history_csv(
        em_history, os.path.join(paths['metrics'], 'em_history.csv'))
    io_utils.save_checkpoint(
        em, os.path.join(paths['checkpoints'], 'em.pt'))

    # Combined training-curve plot.
    plotting.plot_rnaseq_training_curves(
        vae_history, em_history,
        os.path.join(paths['plots'], 'training_curves.png'),
    )

    # 4. Latents + t-SNE + recon scatter.
    log.info('Extracting VAE latents (encoder mean)...')
    z_vae, y_vae = extract_latents(vae, test_loader, device, mode='encoder')
    log.info('Extracting EM latents (Langevin from prior init)...')
    z_em, y_em = extract_latents(em, test_loader, device, mode='langevin',
                                 langevin_steps=100, langevin_lr=0.005)

    log.info('Running t-SNE on both...')
    z_vae_2d = TSNE(n_components=2, perplexity=30, init='pca',
                    random_state=0).fit_transform(z_vae)
    z_em_2d = TSNE(n_components=2, perplexity=30, init='pca',
                   random_state=0).fit_transform(z_em)

    # Reconstruction scatter on a single test cell.
    real_x, _ = next(iter(test_loader))
    sample_x = real_x[0].to(device)

    with torch.no_grad():
        z_mu, _ = vae.encode(sample_x.unsqueeze(0))
        vae_recon, _ = vae.decode(z_mu)

    z_init = torch.randn(1, em.latent_dim, device=device)
    with torch.enable_grad():
        z_em_sample = langevin_sample(em, sample_x.unsqueeze(0), z_init, num_steps=100)
    with torch.no_grad():
        em_recon, _ = em.decode(z_em_sample)

    sample_x_np = sample_x.cpu().numpy()
    vae_recon_np = vae_recon.squeeze(0).cpu().numpy()
    em_recon_np = em_recon.squeeze(0).cpu().numpy()
    r_vae, _ = pearsonr(sample_x_np, vae_recon_np)
    r_em, _ = pearsonr(sample_x_np, em_recon_np)
    log.info(f'  Pearson r (VAE recon vs true, single cell): {r_vae:.3f}')
    log.info(f'  Pearson r (EM  recon vs true, single cell): {r_em:.3f}')

    io_utils.save_latents_npz(
        os.path.join(paths['metrics'], 'latents.npz'),
        z_vae=z_vae, y_vae=y_vae, z_em=z_em, y_em=y_em,
        z_vae_2d=z_vae_2d, z_em_2d=z_em_2d,
        sample_x=sample_x_np, vae_recon=vae_recon_np, em_recon=em_recon_np,
    )
    plotting.plot_rnaseq_evaluation(
        z_vae_2d, y_vae, z_em_2d, y_em,
        sample_x_np, vae_recon_np, em_recon_np,
        r_vae, r_em,
        os.path.join(paths['plots'], 'tsne_and_recon.png'),
    )

    # 5. Marginal log-likelihood.
    log.info(f'Estimating log p(x) on test (K={args.K})...')
    ll_vae = estimate_marginal_log_likelihood(vae, test_loader, device, K=args.K)
    ll_em = estimate_marginal_log_likelihood(em, test_loader, device, K=args.K)
    log.info(f'  VAE: {ll_vae:.3f} nats / cell')
    log.info(f'  EM : {ll_em:.3f} nats / cell')
    log.info(f'  Delta (EM - VAE): {ll_em - ll_vae:+.3f}')
    io_utils.save_metrics_json(
        {'K': args.K,
         'log_p_x_vae': ll_vae,
         'log_p_x_em': ll_em,
         'delta_em_minus_vae': ll_em - ll_vae,
         'recon_pearson_r_vae_single_cell': float(r_vae),
         'recon_pearson_r_em_single_cell': float(r_em)},
        os.path.join(paths['metrics'], 'log_likelihood.json'),
    )

    log.info('RNA-seq experiment complete.')
