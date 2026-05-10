"""
VAE (ELBO) and approximate-EM (Langevin E-step + decoder M-step) trainers.

The encoder phi is frozen during EM. The optimizer is constructed over
decoder parameters only, so phi cannot drift even if requires_grad were
re-enabled elsewhere.
"""

import logging
import torch

from .model import elbo_loss
from .langevin import langevin_sample

log = logging.getLogger(__name__)


def train_vae(model, loader, device, epochs=30, lr=1e-3, grad_clip=5.0):
    """Standard ELBO training over all VAE parameters (phi and theta)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'total': [], 'recon': [], 'kl': []}

    log.info('Training VAE (ELBO)...')
    for epoch in range(epochs):
        model.train()
        ep_total = ep_recon = ep_kl = 0.0
        n = 0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            dec_mu, dec_logvar, enc_mu, enc_logvar, _ = model(x)
            loss, recon, kl = elbo_loss(x, dec_mu, dec_logvar, enc_mu, enc_logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ep_total += loss.item()
            ep_recon += recon
            ep_kl += kl
            n += x.size(0)

        history['total'].append(ep_total / n)
        history['recon'].append(ep_recon / n)
        history['kl'].append(ep_kl / n)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            log.info(
                f'  ep {epoch+1:3d}/{epochs} | -ELBO/sample {ep_total/n:.3f} '
                f'| recon {ep_recon/n:.3f} | KL {ep_kl/n:.3f}'
            )
    return history


def train_em(model, loader, device, epochs=30, lr=1e-4,
             langevin_steps=30, langevin_lr=0.005, m_steps=1, grad_clip=5.0):
    """Approximate EM. Encoder is frozen; only decoder parameters are updated.

    Per-minibatch:
        E-step: sample z ~ p_theta_old(z | x) via Langevin, warm-started at mu_phi(x).
        M-step: m_steps gradient ascent steps on log p_theta(x | z) wrt theta only.
    """
    # Freeze encoder parameters explicitly.
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in [model.enc_mu.weight, model.enc_mu.bias,
              model.enc_logvar.weight, model.enc_logvar.bias]:
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.decoder_parameters(), lr=lr)
    history = {'m_loss': []}

    log.info('Training Approximate EM (Langevin E-step + decoder M-step)...')
    for epoch in range(epochs):
        ep_loss = 0.0
        n = 0
        last_neg_log_pxz = 0.0
        for x, _ in loader:
            x = x.to(device)

            # E-step.
            with torch.no_grad():
                z_init, _ = model.encode(x)
            with torch.enable_grad():
                z_samples = langevin_sample(
                    model, x, z_init,
                    num_steps=langevin_steps, step_size=langevin_lr,
                )

            # M-step.
            for _ in range(m_steps):
                optimizer.zero_grad()
                dec_mu, dec_logvar = model.decode(z_samples.detach())
                neg_log_pxz = 0.5 * torch.sum(
                    dec_logvar + (x - dec_mu).pow(2) * (-dec_logvar).exp()
                )
                neg_log_pxz.backward()
                torch.nn.utils.clip_grad_norm_(model.decoder_parameters(), grad_clip)
                optimizer.step()

            last_neg_log_pxz = neg_log_pxz.item()
            ep_loss += last_neg_log_pxz
            n += x.size(0)

        history['m_loss'].append(ep_loss / n)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            log.info(f'  ep {epoch+1:3d}/{epochs} | -log p(x|z)/sample {ep_loss/n:.3f}')
    return history
