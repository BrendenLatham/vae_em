# VAE vs. Approximate-EM on MNIST and PBMC 10k

Both experiments train a Gaussian-decoder VAE with per-pixel learned
variance, then warm-start an approximate-EM model from the trained VAE
weights and run Langevin E-steps + decoder-only M-steps.

## Usage

```bash
pip install -r requirements.txt

python main.py --mode mnist        # MNIST experiment only
python main.py --mode rnaseq       # PBMC 10k experiment only
python main.py --mode full         # both
```

Common flags:

```
--output-dir DIR        # default: outputs
--data-dir   DIR        # default: ./data  (MNIST + PBMC h5 cache; never wiped)
--seed       INT        # default: 42
--device     auto|cuda|cpu
--epochs-vae INT        # default: 30
--epochs-em  INT        # default: 30
--langevin-steps INT    # default: 30 (E-step inner steps; also the cutoff line in the trace plot)
--K          INT        # default: 1000  (prior samples for log p(x) + ESS)
--m-step
```

## Output layout

`--mode mnist` wipes only `outputs/mnist/`. `--mode rnaseq` wipes only
`outputs/rnaseq/`. `--mode full` wipes both. `outputs/run.log` is always
overwritten. The dataset cache in `--data-dir` is never touched.

```
outputs/
├── run.log
├── mnist/
│   ├── plots/
│   │   ├── vae_training.png
│   │   ├── em_training.png
│   │   ├── tsne_latents.png
│   │   ├── samples_recon_slerp.png
│   │   └── langevin_convergence.png
│   ├── metrics/
│   │   ├── vae_history.csv
│   │   ├── em_history.csv
│   │   ├── log_likelihood.json
│   │   ├── ess_mse.json
│   │   ├── latents.npz             # z_vae, y_vae, z_em, y_em, z_vae_2d, z_em_2d
│   │   └── langevin_trace.npy
│   └── checkpoints/
│       ├── vae.pt
│       └── em.pt
└── rnaseq/
    ├── plots/
    │   ├── training_curves.png
    │   └── tsne_and_recon.png
    ├── metrics/
    │   ├── vae_history.csv
    │   ├── em_history.csv
    │   ├── log_likelihood.json
    │   └── latents.npz
    └── checkpoints/
        ├── vae.pt
        └── em.pt
```

## Module map

```
main.py                       # CLI entry (argparse + dispatch)
src/
├── model.py                  # VAE, elbo_loss, LOGVAR_MIN/MAX
├── langevin.py               # log_joint, langevin_sample (ULA)
├── training.py               # train_vae, train_em
├── evaluation.py             # log p(x), extract_latents, ESS/MSE,
│                             #   compute_langevin_trace, slerp
├── plotting.py               # all figures; Agg backend; savefig + close
├── data_mnist.py             # torchvision loaders
├── data_rnaseq.py            # PBMC 10k via scanpy (lazy-imported)
├── experiment_mnist.py       # MNIST orchestrator
├── experiment_rnaseq.py      # RNA-seq orchestrator
└── io_utils.py               # output reset, logging, CSV/JSON/npz/ckpt writers
```
