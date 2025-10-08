# CPA Training Quick Start

## What Was Copied

The following files and directories were copied from `/home/dennis/state-benchmark/baselines/` to `/home/dennis/cell-eval/src/baselines/`:

### Core Files
- ✅ `advanced_dataloader.py` - Data loading utilities for perturbation data
- ✅ `requirements.txt` - Python dependencies
- ✅ `README.md` - Comprehensive documentation
- ✅ `example_config.toml` - Example TOML configuration

### State Sets Reproduce Package
- ✅ `state_sets_reproduce/` - Main package directory
  - `__init__.py`
  - `models/` - Model implementations
    - `cpa/` - CPA model implementation
    - `scvi/` - scVI model (bonus)
    - `base.py`, `utils.py`, `__init__.py`
  - `train/` - Training scripts
    - `__main__.py` - Main training entry point
    - `main2.py`, `utils.py`
  - `configs/` - Hydra configuration files
    - `config.yaml` - Main config
    - `model/cpa.yaml` - CPA model config
    - `training/cpa.yaml` - CPA training config
    - `data/perturbation.yaml` - Data config
    - `splits/` - Example split configs
  - `callbacks/` - PyTorch Lightning callbacks
  - `cpa_parse/` - Example CPA configs for Parse dataset

## How to Run CPA Training

### Option 1: Use the Example Script

```bash
cd /home/dennis/cell-eval/src/baselines
./run_cpa_example.sh
```

### Option 2: Run Directly with Python

```bash
cd /home/dennis/cell-eval/src/baselines

python3 -m state_sets_reproduce.train \
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

### Option 3: Create Your Own Shell Script

```bash
#!/bin/bash
cd /home/dennis/cell-eval/src/baselines

OUTPUT_DIR="/home/dennis/cell-eval/outputs/my_cpa_run"
DATA_PATH="/home/dennis/Parse_10M_PBMC_cytokines.h5ad"

python3 -m state_sets_reproduce.train \
    model=cpa \
    data.kwargs.data_path="${DATA_PATH}" \
    data.kwargs.embed_key=X_hvg \
    data.kwargs.pert_col=cytokine \
    data.kwargs.cell_type_key=cell_type \
    data.kwargs.batch_col=donor \
    data.kwargs.control_pert=PBS \
    training.max_steps=50000 \
    output_dir="${OUTPUT_DIR}"
```

## Prerequisites

1. **Install Dependencies** (if not already installed):
   ```bash
   cd /home/dennis/cell-eval/src/baselines
   pip install -r requirements.txt
   ```

2. **Prepare Your Data**:
   - Your h5ad file should have:
     - Expression data in `adata.obsm['X_hvg']` (or specify a different key)
     - Perturbation column (e.g., `adata.obs['cytokine']`)
     - Cell type column (e.g., `adata.obs['cell_type']`)
     - Batch/donor column (e.g., `adata.obs['donor']`)

3. **Working Directory**:
   - Always run from `/home/dennis/cell-eval/src/baselines/` directory

## Key Configuration Parameters

### Data Parameters
- `data.kwargs.data_path` - Path to h5ad file
- `data.kwargs.embed_key` - Key in adata.obsm (e.g., "X_hvg")
- `data.kwargs.pert_col` - Perturbation column name
- `data.kwargs.cell_type_key` - Cell type column name
- `data.kwargs.batch_col` - Batch/donor column name
- `data.kwargs.control_pert` - Control condition label
- `data.kwargs.output_space` - "gene" or "all"
- `data.kwargs.basal_mapping_strategy` - "batch", "random", "cell_type", or "batch_cell_type"

### Model Parameters
- `model.kwargs.n_latent` - Latent dimension (default: 64)
- `model.kwargs.hidden_dim` - Hidden layer size (default: 256)
- `model.kwargs.n_layers_encoder` - Number of encoder layers (default: 4)
- `model.kwargs.n_layers_decoder` - Number of decoder layers (default: 3)
- `model.kwargs.dropout_rate_encoder` - Encoder dropout (default: 0.15)
- `model.kwargs.dropout_rate_decoder` - Decoder dropout (default: 0.15)
- `model.kwargs.recon_loss` - Loss type: "gauss", "mse", or "nb"

### Training Parameters
- `training.batch_size` - Batch size (default: 128)
- `training.max_steps` - Maximum training steps (default: 50000)
- `training.val_freq` - Validation frequency (default: 1000)
- `training.lr` - Learning rate (default: 0.0003)
- `training.train_seed` - Random seed (default: 42)

### Output Parameters
- `output_dir` - Where to save results (use absolute path)
- `name` - Experiment name
- `use_wandb` - Enable WandB logging (true/false)

## Output Structure

After training, you'll find:
```
output_dir/
└── experiment_name/
    ├── checkpoints/
    │   ├── last.ckpt          # Best checkpoint
    │   ├── final.ckpt         # Final checkpoint
    │   └── step=*.ckpt        # Periodic checkpoints
    ├── config.yaml            # Full config used
    ├── data_module.pkl        # Saved data module
    └── version_0/
        └── metrics.csv        # Training metrics
```

## Troubleshooting

### Import Errors
Make sure you're in the right directory:
```bash
cd /home/dennis/cell-eval/src/baselines
python3 -m state_sets_reproduce.train ...
```

### Memory Issues
Reduce batch size:
```bash
training.batch_size=64  # or 32
```

### Path Issues
Use full paths, not `~`:
```bash
# ❌ Wrong
data.kwargs.data_path=~/data.h5ad

# ✅ Correct
data.kwargs.data_path=/home/dennis/data.h5ad
```

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Prepare your data (h5ad file with required fields)
3. Run the example: `./run_cpa_example.sh`
4. Check outputs in the specified output directory
5. For detailed documentation, see `README.md`

## Original Location

Original files are in: `/home/dennis/state-benchmark/baselines/`

## Notes

- ❌ `.h5ad` data files were NOT copied (as requested)
- ✅ All scripts, configs, and model code were copied
- ✅ The training is fully runnable from this directory
- ✅ No modifications to code were needed - it works as-is!

