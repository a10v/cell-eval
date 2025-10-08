# from advanced_dataloader import AdvancedPerturbationDataModule

import numpy as np
import scanpy as sc
import h5py
from tqdm import trange

H5AD = "/home/dennis/Parse_10M_PBMC_cytokines_hvg.h5ad"

# Load data in backed mode for subsampling
adata_b = sc.read_h5ad(H5AD, backed="r")

print(adata_b.obs["gene_count"].shape)

# print(adata_b["obsm/hvg"].shape)

# print(adata_b.var["highly_variable"].shape)

# print(adata_b.uns["hvg"].keys())

print(adata_b.obsm.keys())

print(adata_b.var.columns)  # shows gene-level metadata columns
print(adata_b.obs.columns)  # shows cell-level metadata columns
print(adata_b.uns.keys())   # shows unstructured data keys