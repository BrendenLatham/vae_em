"""
VAE with learned per-pixel decoder variance, plus the Gaussian ELBO.

Mirrors the notebook's cells 4 and 6. The logvar clamp range is applied
uniformly across training, sampling, and evaluation -- changing it in only
one place would mean the sampler targets a different distribution than the
one being trained.
"""

import torch
import torch.nn as nn

# Variance bounds: ~e^-6 to ~e^2. Applied uniformly in train / sample / eval.
LOGVAR_MIN = -6.0
LOGVAR_MAX = 2.0


class VAE(nn.Module):
    """VAE with learned per-pixel decoder variance: p(x|z) = N(mu(z), sigma^2(z))."""

    def __init__(self, input_dim=784, latent_dim=10, hidden_dim=400):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ---- Encoder phi: x -> q_phi(z | x) ----
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ---- Decoder theta: z -> p_theta(x | z) ----
        self.decoder_body = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.dec_mu = nn.Linear(hidden_dim, input_dim)
        self.dec_logvar = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.enc_mu(h), self.enc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.decoder_body(z)
        mu = self.dec_mu(h)
        logvar = self.dec_logvar(h).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar

    def forward(self, x):
        enc_mu, enc_logvar = self.encode(x)
        z = self.reparameterize(enc_mu, enc_logvar)
        dec_mu, dec_logvar = self.decode(z)
        return dec_mu, dec_logvar, enc_mu, enc_logvar, z

    def decoder_parameters(self):
        """Theta only -- used by the M-step optimizer."""
        return (list(self.decoder_body.parameters())
                + list(self.dec_mu.parameters())
                + list(self.dec_logvar.parameters()))


def elbo_loss(x, dec_mu, dec_logvar, enc_mu, enc_logvar):
    """Returns (-ELBO summed over batch, per-batch recon NLL, per-batch KL).

    The d/2 * log(2*pi) constant of the Gaussian normalization is dropped
    here (it does not affect gradients) and re-added in the marginal-
    likelihood estimator at evaluation time.
    """
    # Reconstruction NLL with per-pixel variance.
    recon = 0.5 * torch.sum(dec_logvar + (x - dec_mu).pow(2) * (-dec_logvar).exp())
    # KL( N(mu_phi, sigma^2_phi) || N(0, I) ) closed form.
    kl = 0.5 * torch.sum(enc_mu.pow(2) + enc_logvar.exp() - 1.0 - enc_logvar)
    return recon + kl, recon.item(), kl.item()
