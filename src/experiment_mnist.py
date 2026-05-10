"""
Orchestrator for the MNIST experiment.

Sequence:
  1. Load data.
  2. Train VAE; save history + checkpoint + training curve.
  3. Deepcopy VAE -> EM; train; save history + checkpoint + training curve.
  4. Extract latents (encoder mean for VAE, Langevin from prior init for EM),
     run t-SNE, save npz + plot.
  5. Estimate log p(x) on test for both; save JSON.
  6. Generate sample / reconstruction / slerp grid; save plot.
  7. Compute ESS + best-prior MSE on test; save JSON.
  8. Run a long Langevin chain on the EM model from prior init; save trace
     plot.
"""

import copy
import logging
import os

import numpy as np
import torch
from sklearn.manifold import TSNE

from .model import VAE
from .training import train_vae, train_em
from .evaluation import (
    estimate_marginal_log_likelihood,
    extract_latents,
    compute_ess_and_mse,
    compute_langevin_trace,
    slerp,
)
from .langevin import langevin_sample
from .data_mnist import get_mnist_loaders
from . import plotting
from . import io_utils

log = logging.getLogger(__name__)


def run_mnist(args, paths):
    """Args is the parsed argparse Namespace. paths is the dict from
    reset_mode_dirs for the 'mnist' subtree."""
    device = args.device

    log.info('=' * 60)
    log.info('MNIST experiment')
    log.info('=' * 60)

    # 1. Data.
    train_loader, test_loader, input_dim = get_mnist_loaders(
        data_dir=args.data_dir,
        n_train=10_000, n_test=2_000,
        batch_size=128, seed=args.seed,
    )

    # 2. VAE.
    vae = VAE(input_dim=input_dim, latent_dim=10, hidden_dim=400).to(device)
    vae_history = train_vae(vae, train_loader, device,
                            epochs=args.epochs_vae, lr=1e-3)
    io_utils.save_history_csv(
        vae_history, os.path.join(paths['metrics'], 'vae_history.csv'))
    io_utils.save_checkpoint(
        vae, os.path.join(paths['checkpoints'], 'vae.pt'))
    plotting.plot_vae_training(
        vae_history, os.path.join(paths['plots'], 'vae_training.png'),
        title='VAE training (MNIST)',
    )

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
    plotting.plot_em_training(
        em_history, os.path.join(paths['plots'], 'em_training.png'),
        title='Approximate EM (M-step loss, MNIST)',
    )

    # 4. Latents + t-SNE.
    log.info('Extracting VAE latents (encoder mean)...')
    z_vae, y_vae = extract_latents(vae, test_loader, device, mode='encoder')
    log.info('Extracting EM latents (Langevin from prior init)...')
    z_em, y_em = extract_latents(em, test_loader, device, mode='langevin',
                                 langevin_steps=100, langevin_lr=0.005)

    log.info('Running t-SNE on both (this can take ~30s)...')
    z_vae_2d = TSNE(n_components=2, perplexity=30, init='pca',
                    random_state=0).fit_transform(z_vae)
    z_em_2d = TSNE(n_components=2, perplexity=30, init='pca',
                   random_state=0).fit_transform(z_em)

    io_utils.save_latents_npz(
        os.path.join(paths['metrics'], 'latents.npz'),
        z_vae=z_vae, y_vae=y_vae, z_em=z_em, y_em=y_em,
        z_vae_2d=z_vae_2d, z_em_2d=z_em_2d,
    )
    plotting.plot_tsne_latents_mnist(
        z_vae_2d, y_vae, z_em_2d, y_em,
        os.path.join(paths['plots'], 'tsne_latents.png'),
    )

    # 5. Marginal log-likelihood.
    log.info(f'Estimating log p(x) on test (K={args.K})...')
    ll_vae = estimate_marginal_log_likelihood(vae, test_loader, device, K=args.K)
    ll_em = estimate_marginal_log_likelihood(em, test_loader, device, K=args.K)
    log.info(f'  VAE: {ll_vae:.3f} nats / image')
    log.info(f'  EM : {ll_em:.3f} nats / image')
    log.info(f'  Delta (EM - VAE): {ll_em - ll_vae:+.3f}')
    io_utils.save_metrics_json(
        {'K': args.K, 'log_p_x_vae': ll_vae, 'log_p_x_em': ll_em,
         'delta_em_minus_vae': ll_em - ll_vae},
        os.path.join(paths['metrics'], 'log_likelihood.json'),
    )

    # 6. Sample / recon / slerp visualization.
    log.info('Generating sample / reconstruction / slerp grid...')
    real_x_np, vae_recon_np, em_recon_np, interp_np = \
        _generate_visualizations(vae, em, test_loader, device, num_samples=8)
    plotting.plot_mnist_samples_recon_slerp(
        real_x_np, vae_recon_np, em_recon_np, interp_np,
        os.path.join(paths['plots'], 'samples_recon_slerp.png'),
        num_samples=8, slerp_steps=8,
    )

    # 7. ESS and best-prior MSE.
    log.info(f'Computing ESS and best-prior MSE (K={args.K})...')
    mse_vae, ess_vae = compute_ess_and_mse(vae, test_loader, device, K=args.K)
    mse_em, ess_em = compute_ess_and_mse(em, test_loader, device, K=args.K)
    log.info(f'  VAE - MSE: {mse_vae:.4f}, Mean ESS: {ess_vae:.2f} (out of {args.K})')
    log.info(f'  EM  - MSE: {mse_em:.4f}, Mean ESS: {ess_em:.2f} (out of {args.K})')
    io_utils.save_metrics_json(
        {'K': args.K,
         'vae': {'mse_per_pixel': mse_vae, 'mean_ess': ess_vae},
         'em':  {'mse_per_pixel': mse_em,  'mean_ess': ess_em}},
        os.path.join(paths['metrics'], 'ess_mse.json'),
    )

    # 8. Langevin convergence trace.
    log.info('Computing Langevin convergence trace on EM model...')
    trace = compute_langevin_trace(
        em, test_loader, device,
        max_steps=200, step_size=0.005, num_chains=16,
    )
    plotting.plot_langevin_convergence(
        trace, os.path.join(paths['plots'], 'langevin_convergence.png'),
        training_cutoff=args.langevin_steps,
    )
    np.save(os.path.join(paths['metrics'], 'langevin_trace.npy'), trace)
    log.info(f'  wrote trace: {os.path.join(paths["metrics"], "langevin_trace.npy")}')

    log.info('MNIST experiment complete.')


def _generate_visualizations(vae_model, em_model, test_loader, device, num_samples=8):
    """Returns (real_x_np, vae_recon_np, em_recon_np, interp_np) all shape
    (num_samples, 784) / (slerp_steps, 784)."""
    vae_model.eval()
    em_model.eval()

    real_x, _ = next(iter(test_loader))
    real_x = real_x[:num_samples].to(device)

    with torch.no_grad():
        # VAE: encoder mean -> decode.
        vae_z_mu, _ = vae_model.encode(real_x)
        vae_recon, _ = vae_model.decode(vae_z_mu)

    # EM: Langevin posterior samples -> decode.
    z_init = torch.randn(num_samples, em_model.latent_dim, device=device)
    with torch.enable_grad():
        em_z = langevin_sample(em_model, real_x, z_init, num_steps=100)
    with torch.no_grad():
        em_recon, _ = em_model.decode(em_z)

    # Slerp between the first two EM latents.
    steps = 8
    z1, z2 = em_z[0:1], em_z[1:2]
    alphas = torch.linspace(0, 1, steps, device=device)
    interp_z = torch.cat([slerp(a, z1, z2) for a in alphas])
    with torch.no_grad():
        interp_mu, _ = em_model.decode(interp_z)

    return (real_x.cpu().numpy(),
            vae_recon.cpu().numpy(),
            em_recon.cpu().numpy(),
            interp_mu.cpu().numpy())
