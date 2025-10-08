# CPA Training Setup Guide for cell-eval

This directory contains the CPA (Compositional Perturbation Autoencoder) training code, copied from the state-benchmark repository.

## Quick Start

### 1. Install Dependencies

First, make sure you have Python 3.11+ installed. Then install the required packages:

```bash
cd /home/dennis/cell-eval/src/baselines
pip install -r requirements.txt
```

**Note**: You may also need to install:
- `flash-attn==1.0.9` (optional, for faster training)
- `torch-scatter` (optional, for some models)

### 2. Prepare Your Data

Make sure your h5ad file exists and has the required fields:
- Expression data in `adata.obsm['X_hvg']` (or another key you specify)
- Perturbation column (e.g., `adata.obs['cytokine']`)
- Cell type column (e.g., `adata.obs['cell_type']`)
- Batch column (e.g., `adata.obs['donor']`)

### 3. Run CPA Training

Use the example script provided:

```bash
cd /home/dennis/cell-eval/src/baselines
./run_cpa_example.sh
```

Or run directly with Python:

```bash
python -m state_sets_reproduce.train \
    model=cpa \
    data.kwargs.data_path=/path/to/your/data.h5ad \
    data.kwargs.embed_key=X_hvg \
    data.kwargs.pert_col=cytokine \
    data.kwargs.cell_type_key=cell_type \
    data.kwargs.batch_col=donor \
    data.kwargs.control_pert=PBS \
    training.batch_size=128 \
    training.max_steps=10000 \
    output_dir=/path/to/output
```

## Configuration

The training is configured using Hydra. Key configuration files are in:
- `state_sets_reproduce/configs/config.yaml` - Main config
- `state_sets_reproduce/configs/model/cpa.yaml` - CPA model config
- `state_sets_reproduce/configs/training/cpa.yaml` - CPA training config

You can override any config parameter via command line:

```bash
python -m state_sets_reproduce.train \
    model=cpa \
    model.kwargs.n_latent=128 \
    training.lr=0.001 \
    ...
```

## Directory Structure

```
baselines/
├── __init__.py                    # Package init
├── advanced_dataloader.py         # Data loading utilities
├── requirements.txt               # Python dependencies
├── README.md                      # Full documentation
├── SETUP.md                       # This file
├── run_cpa_example.sh            # Example training script
└── state_sets_reproduce/
    ├── __init__.py
    ├── callbacks/                 # Training callbacks
    ├── configs/                   # Hydra configurations
    │   ├── config.yaml           # Main config
    │   ├── model/                # Model configs
    │   │   └── cpa.yaml         # CPA specific
    │   ├── training/             # Training configs
    │   └── data/                 # Data configs
    ├── models/                   # Model implementations
    │   ├── base.py
    │   ├── utils.py
    │   ├── cpa/                  # CPA model
    │   │   ├── _model.py
    │   │   ├── _module.py
    │   │   ├── _task.py
    │   │   └── ...
    │   └── scvi/                 # scVI model (if needed)
    └── train/                    # Training scripts
        └── __main__.py           # Main training entry point
```

## Key Features

1. **Batch-aware control mapping**: Automatically matches control cells from the same batch/donor
2. **HVG support**: Works with highly variable genes (X_hvg)
3. **Flexible configuration**: Easily adjust hyperparameters via command line
4. **PyTorch Lightning**: Built on Lightning for robust training
5. **Checkpointing**: Automatic model checkpointing and resumption

## Common Issues

### Import Errors
If you get import errors, make sure you're running from the baselines directory:
```bash
cd /home/dennis/cell-eval/src/baselines
python -m state_sets_reproduce.train ...
```

### Path Issues
Always use **full absolute paths** in configs, not `~`:
```bash
# ❌ Bad
data.kwargs.data_path=~/data.h5ad

# ✅ Good
data.kwargs.data_path=/home/dennis/data.h5ad
```

### Memory Issues
If you run out of memory:
- Reduce `training.batch_size` (try 64 or 32)
- Reduce `data.kwargs.num_workers` (try 2 or 1)

## For More Information

See the comprehensive [README.md](README.md) for:
- Detailed configuration options
- Advanced splitting strategies (few-shot, zero-shot)
- Model architecture details
- Hyperparameter tuning tips
- WandB integration

## Related Files

- Original implementation: `/home/dennis/state-benchmark/baselines/`
- Data file: `/home/dennis/Parse_10M_PBMC_cytokines.h5ad`

