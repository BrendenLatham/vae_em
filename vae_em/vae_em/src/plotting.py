"""
All plotting functions. Each takes pre-computed inputs and an output path,
calls savefig + close, and never calls plt.show. The Agg backend is set
before pyplot is imported so this module can run on a headless server.
"""

import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

_DPI = 150


def _save(fig, path):
    fig.savefig(path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    log.info(f'  wrote plot: {path}')


def plot_vae_training(history, path, title='VAE training'):
    fig = plt.figure(figsize=(9, 3.5))
    plt.plot(history['total'], label='-ELBO / sample')
    plt.plot(history['recon'], label='Recon NLL', ls='--')
    plt.plot(history['kl'], label='KL', ls=':')
    plt.xlabel('epoch')
    plt.ylabel('loss / sample')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, path)


def plot_em_training(history, path, title='Approximate EM (M-step loss)'):
    fig = plt.figure(figsize=(9, 3.5))
    plt.plot(history['m_loss'], color='tab:green')
    plt.xlabel('epoch')
    plt.ylabel('-log p(x|z) / sample')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, path)


def plot_tsne_latents_mnist(z_vae_2d, y_vae, z_em_2d, y_em, path,
                            num_classes=10, suptitle='MNIST: t-SNE of 10-D latents'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    cmap = plt.get_cmap('tab10')
    for c in range(num_classes):
        m1 = y_vae == c
        m2 = y_em == c
        axes[0].scatter(z_vae_2d[m1, 0], z_vae_2d[m1, 1],
                        s=8, alpha=0.6, color=cmap(c), label=str(c))
        axes[1].scatter(z_em_2d[m2, 0], z_em_2d[m2, 1],
                        s=8, alpha=0.6, color=cmap(c), label=str(c))
    axes[0].set_title('VAE  —  encoder mean')
    axes[0].legend(fontsize=7, markerscale=1.5)
    axes[1].set_title('Approximate EM  —  Langevin posterior')
    axes[1].legend(fontsize=7, markerscale=1.5)
    plt.suptitle(suptitle)
    plt.tight_layout()
    _save(fig, path)


def plot_mnist_samples_recon_slerp(real_x, vae_recon, em_recon, interp_mu, path,
                                   num_samples=8, slerp_steps=8):
    """real_x, vae_recon, em_recon: (num_samples, 784) numpy arrays.
    interp_mu: (slerp_steps, 784) numpy array."""
    cols = max(num_samples, slerp_steps)
    fig, axes = plt.subplots(4, cols, figsize=(12, 6))
    for ax in axes.flatten():
        ax.axis('off')

    for i in range(num_samples):
        axes[0, i].imshow(real_x[i].reshape(28, 28), cmap='gray')
        if i == 0:
            axes[0, i].set_title('Real')
        axes[1, i].imshow(vae_recon[i].reshape(28, 28), cmap='gray')
        if i == 0:
            axes[1, i].set_title('VAE Recon')
        axes[2, i].imshow(em_recon[i].reshape(28, 28), cmap='gray')
        if i == 0:
            axes[2, i].set_title('EM Recon')

    for i in range(slerp_steps):
        axes[3, i].imshow(interp_mu[i].reshape(28, 28), cmap='gray')
        if i == 0:
            axes[3, i].set_title('EM Slerp')

    plt.tight_layout()
    _save(fig, path)


def plot_langevin_convergence(trace, path, training_cutoff=30,
                              title='Langevin Chain Convergence (Unadjusted)'):
    """trace: (max_steps, num_chains) numpy array of log p(x, z)."""
    fig = plt.figure(figsize=(8, 4))
    for i in range(trace.shape[1]):
        plt.plot(trace[:, i], alpha=0.4, color='tab:blue')
    plt.plot(trace.mean(axis=1), color='black', linewidth=2, label='Mean Log-Joint')
    plt.axvline(x=training_cutoff, color='red', linestyle='--',
                label=f'Training Cutoff ({training_cutoff} steps)')
    plt.title(title)
    plt.xlabel('Langevin Steps')
    plt.ylabel('log p(x, z)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, path)


def plot_rnaseq_training_curves(vae_history, em_history, path,
                                title_vae='PBMC 10k: VAE Training',
                                title_em='PBMC 10k: Approximate EM'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(vae_history['total'], label='-ELBO')
    axes[0].plot(vae_history['recon'], label='Recon NLL', ls='--')
    axes[0].set_title(title_vae)
    axes[0].legend()

    axes[1].plot(em_history['m_loss'], color='tab:green')
    axes[1].set_title(title_em)
    axes[1].set_ylabel('-log p(x|z)')
    plt.tight_layout()
    _save(fig, path)


def plot_rnaseq_evaluation(z_vae_2d, y_vae, z_em_2d, y_em,
                           sample_x_np, vae_recon_np, em_recon_np,
                           r_vae, r_em, path):
    """Three-panel: VAE t-SNE, EM t-SNE, true-vs-recon scatter."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.get_cmap('tab20')

    for c in np.unique(y_vae):
        axes[0].scatter(z_vae_2d[y_vae == c, 0], z_vae_2d[y_vae == c, 1],
                        s=8, alpha=0.6, color=cmap(c % 20), label=f'Cluster {c}')
        axes[1].scatter(z_em_2d[y_em == c, 0], z_em_2d[y_em == c, 1],
                        s=8, alpha=0.6, color=cmap(c % 20))

    axes[0].set_title('VAE Latents (Encoder Mean)')
    axes[0].legend(fontsize=7, loc='best')
    axes[1].set_title('Approximate EM Latents (Langevin Posterior)')

    axes[2].scatter(sample_x_np, vae_recon_np, alpha=0.2, s=5, label=f'VAE (r={r_vae:.3f})')
    axes[2].scatter(sample_x_np, em_recon_np, alpha=0.2, s=5, label=f'EM (r={r_em:.3f})')
    min_val, max_val = sample_x_np.min(), sample_x_np.max()
    axes[2].plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
    axes[2].set_title('Gene Expression: True vs Recon')
    axes[2].set_xlabel('True log-counts')
    axes[2].set_ylabel('Reconstructed log-counts')
    axes[2].legend()

    plt.tight_layout()
    _save(fig, path)
