"""
Evaluation utilities. Each function returns numbers or arrays; plotting
lives in plotting.py so we can re-plot without re-computing.

Marginal likelihood is estimated as
    log p_theta(x) ~= log( (1/K) sum_k p_theta(x | z_k) ),  z_k ~ N(0, I)
with logsumexp for numerical stability and the d/2 * log(2*pi) constant
restored (it was dropped during training).
"""

import math
import logging
import numpy as np
import torch
from tqdm.auto import tqdm

from .langevin import log_joint, langevin_sample

log = logging.getLogger(__name__)


def estimate_marginal_log_likelihood(model, loader, device, K=1000, max_batches=None):
    """MC estimate of mean log p(x) via prior importance sampling.

    Includes the full Gaussian normalization constant that elbo_loss drops.
    """
    model.eval()
    d = model.input_dim
    log_norm = -0.5 * d * math.log(2.0 * math.pi)
    total_ll = 0.0
    n = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(loader, desc=f'log p(x), K={K}')):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            B = x.size(0)
            z = torch.randn(B, K, model.latent_dim, device=device).view(B * K, -1)
            dec_mu, dec_logvar = model.decode(z)
            dec_mu = dec_mu.view(B, K, d)
            dec_logvar = dec_logvar.view(B, K, d)
            x_exp = x.unsqueeze(1)  # (B, 1, d)
            log_pxz = (-0.5 * torch.sum(
                dec_logvar + (x_exp - dec_mu).pow(2) * (-dec_logvar).exp(), dim=2
            ) + log_norm)  # (B, K)
            log_px = torch.logsumexp(log_pxz, dim=1) - math.log(K)
            total_ll += log_px.sum().item()
            n += B
    return total_ll / n


def extract_latents(model, loader, device, mode='encoder',
                    langevin_steps=100, langevin_lr=0.005):
    """mode='encoder' returns mu_phi(x); mode='langevin' returns posterior samples.

    The two modes match how each method uses its latents at sample time:
    VAE uses the encoder mean, EM uses Langevin posterior samples.
    """
    assert mode in ('encoder', 'langevin')
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        x = x.to(device)
        if mode == 'encoder':
            with torch.no_grad():
                mu, _ = model.encode(x)
            zs.append(mu.cpu())
        else:
            z_init = torch.randn(x.size(0), model.latent_dim, device=device)
            with torch.enable_grad():
                z = langevin_sample(model, x, z_init,
                                    num_steps=langevin_steps, step_size=langevin_lr)
            zs.append(z.cpu())
        ys.append(y)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


def compute_ess_and_mse(model, loader, device, K=1000):
    """Effective sample size of prior IS, and best-prior-sample MSE per pixel.

    MSE here is the per-pixel reconstruction error using the prior sample
    with the largest IS weight -- a quick proxy that does not require the
    encoder, so it is comparable across VAE and EM models.
    """
    model.eval()
    d = model.input_dim
    log_norm = -0.5 * d * math.log(2.0 * math.pi)

    total_mse = 0.0
    total_ess = 0.0
    n = 0

    with torch.no_grad():
        for x, _ in tqdm(loader, desc='Computing ESS & MSE'):
            x = x.to(device)
            B = x.size(0)

            # IS log-weights from prior samples.
            z = torch.randn(B, K, model.latent_dim, device=device).view(B * K, -1)
            dec_mu_is, dec_logvar_is = model.decode(z)
            dec_mu_is = dec_mu_is.view(B, K, d)
            dec_logvar_is = dec_logvar_is.view(B, K, d)

            x_exp = x.unsqueeze(1)
            log_w = (-0.5 * torch.sum(
                dec_logvar_is + (x_exp - dec_mu_is).pow(2) * (-dec_logvar_is).exp(),
                dim=2,
            ) + log_norm)

            log_sum_w = torch.logsumexp(log_w, dim=1, keepdim=True)
            log_tilde_w = log_w - log_sum_w
            ess = torch.exp(-torch.logsumexp(2 * log_tilde_w, dim=1))
            total_ess += ess.sum().item()

            best_idx = torch.argmax(log_w, dim=1)
            best_dec_mu = dec_mu_is[torch.arange(B), best_idx, :]
            total_mse += torch.nn.functional.mse_loss(best_dec_mu, x, reduction='sum').item()

            n += B

    return total_mse / (n * d), total_ess / n


def compute_langevin_trace(model, loader, device, max_steps=200, step_size=0.005,
                           num_chains=16, grad_clip=100.0):
    """Track log_joint over a long Langevin run starting from prior init.

    Returns an (max_steps, num_chains) array. Plotting is done elsewhere.
    """
    x, _ = next(iter(loader))
    x = x[:num_chains].to(device)

    z = torch.randn(x.size(0), model.latent_dim, device=device)
    z = z.requires_grad_(True)

    trace = []
    for _ in range(max_steps):
        lj = log_joint(z, x, model)
        trace.append(lj.detach().cpu().numpy())

        grad_z, = torch.autograd.grad(lj.sum(), z)
        grad_z = grad_z.clamp(-grad_clip, grad_clip)

        with torch.no_grad():
            noise = torch.randn_like(z)
            z = z + step_size * grad_z + math.sqrt(2.0 * step_size) * noise
        z = z.detach().requires_grad_(True)

    return np.array(trace)


def slerp(val, low, high):
    """Spherical linear interpolation between two single-row latent vectors.

    Falls back to linear interpolation when the two vectors are colinear.
    Inputs are torch tensors of shape (1, latent_dim); val is a 0-d tensor
    in [0, 1].
    """
    omega = torch.acos(
        (low / torch.norm(low, dim=1, keepdim=True)
         * high / torch.norm(high, dim=1, keepdim=True)).sum(1).clamp(-1, 1)
    )
    so = torch.sin(omega)
    if so.item() == 0:
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so * low
            + torch.sin(val * omega) / so * high)
