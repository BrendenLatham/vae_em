"""
Unadjusted Langevin sampler for the posterior p_theta(z | x).

The score is:
    grad_z log p_theta(z|x) = -z + grad_z log p_theta(x|z)
where p(z) = N(0, I) gives -z and the second term comes from autograd
through the decoder. p_theta(x) is z-independent and drops out.
"""

import math
import torch


def log_joint(z, x, model):
    """log p(z) + log p_theta(x|z) per sample, up to a z-independent constant.

    Returns a (batch,) tensor; .sum() over the batch is what the sampler
    differentiates w.r.t. z.
    """
    log_pz = -0.5 * z.pow(2).sum(dim=1)
    dec_mu, dec_logvar = model.decode(z)  # decoder applies the logvar clamp
    log_pxz = -0.5 * torch.sum(
        dec_logvar + (x - dec_mu).pow(2) * (-dec_logvar).exp(), dim=1
    )
    return log_pz + log_pxz


def langevin_sample(model, x, z_init, num_steps=50, step_size=0.005, grad_clip=100.0):
    """ULA chain: z_{t+1} = z_t + eps * grad_z log p(z|x) + sqrt(2*eps) * noise.

    Decoder parameters are not differentiated through; we only take d/dz.
    Per-coordinate gradient clamp prevents one bad sample from blowing up
    the chain. Stationary distribution is biased by O(eps); acceptable for
    the "approximate" EM the spec asks for.
    """
    z = z_init.clone().detach().requires_grad_(True)
    for _ in range(num_steps):
        lj = log_joint(z, x, model).sum()
        grad_z, = torch.autograd.grad(lj, z)
        grad_z = grad_z.clamp(-grad_clip, grad_clip)
        with torch.no_grad():
            noise = torch.randn_like(z)
            z = z + step_size * grad_z + math.sqrt(2.0 * step_size) * noise
        z = z.detach().requires_grad_(True)
    return z.detach()
