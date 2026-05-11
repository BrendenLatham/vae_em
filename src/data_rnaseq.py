"""
PBMC 10k (10x Genomics) loader for the RNA-seq experiment.

Lazy-imports scanpy / igraph / louvain so users running --mode mnist do not
need them installed. The h5 file is cached at `data_dir/pbmc_10k_v3_*.h5`
and is treated as an upstream input, not an output of this program.
"""

import logging
import os
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

PBMC_FILENAME = 'pbmc_10k_v3_filtered_feature_bc_matrix.h5'
PBMC_URL = (
    'http://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/'
    'pbmc_10k_v3_filtered_feature_bc_matrix.h5'
)


class RNASeqDataset(Dataset):
    def __init__(self, data_matrix, labels):
        self.data = torch.tensor(data_matrix, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def _download_pbmc(h5_path):
    if os.path.exists(h5_path):
        log.info(f'PBMC h5 already present: {h5_path}')
        return
    log.info(f'Downloading PBMC 10k dataset from 10x Genomics (~80MB) -> {h5_path}')
    # Spoof User-Agent to bypass 10x's bot filter.
    req = urllib.request.Request(PBMC_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(h5_path, 'wb') as out_file:
        out_file.write(response.read())


def get_rnaseq_loaders(data_dir='./data', batch_size=128, seed=42):
    """Returns (train_loader, test_loader, input_dim).
    """
    try:
        import scanpy as sc
    except ImportError as e:
        raise ImportError(
            'RNA-seq mode requires scanpy. Install with: '
            'pip install scanpy igraph louvain'
        ) from e

    os.makedirs(data_dir, exist_ok=True)
    h5_path = os.path.join(data_dir, PBMC_FILENAME)
    _download_pbmc(h5_path)

    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    log.info('Preprocessing PBMC 10k...')
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

    adata = adata[:, adata.var.highly_variable]

    # Dense log-normalized HVG matrix used for VAE training (pre-scaling).
    X_rna = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    # Separate scale -> PCA -> neighbors -> Louvain just for cluster labels.
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.louvain(adata)
    y_rna = adata.obs['louvain'].cat.codes.values

    rng = np.random.RandomState(seed)
    train_idx = rng.choice(len(X_rna), int(len(X_rna) * 0.8), replace=False)
    test_idx = np.setdiff1d(np.arange(len(X_rna)), train_idx)

    n_genes = X_rna.shape[1]
    log.info(f'PBMC 10k: {len(X_rna)} cells, {n_genes} HVGs, '
             f'train={len(train_idx)} test={len(test_idx)}')

    train_loader = DataLoader(
        RNASeqDataset(X_rna[train_idx], y_rna[train_idx]),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        RNASeqDataset(X_rna[test_idx], y_rna[test_idx]),
        batch_size=batch_size, shuffle=False,
    )
    return train_loader, test_loader, n_genes
