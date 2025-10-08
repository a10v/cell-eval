# State-Sets Baselines Reproduction

This directory contains the implementation of the baselines from the State-Sets paper.

## Overview

This project implements and evaluates several baseline models for perturbation modeling:
- LowRankLinear (LRM)
- CPA (Compositional Perturbation Autoencoder)
- scGPT
- scVI

The models are evaluated on two major datasets:
- Replogle dataset (with 4-fold cross-validation)
- Tahoe dataset

## Installation

### 1. Pre-requisites

#### PyTorch and CUDA Setup
First, install the appropriate PyTorch version. This project was tested with:
- PyTorch 2.4.1
- CUDA 12.1

Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) to get the correct installation command for your system.

### 2. Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/Arcinstitute/state-sets-reproduce.git
cd state-sets-reproduce/baselines
```

2. Create and activate a virtual environment using uv:
```bash
uv venv --python 3.11.6
source .venv/bin/activate
```

3. Install dependencies:

```bash
uv pip install -r requirements.txt
```

Install flash attention and torch-scatter
```bash
uv pip install flash-attn==1.0.9
uv pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
```
In case of any issues, please refer to the [flash-attention](https://github.com/Dao-AILab/flash-attention) repository.

## Usage

### Training Baselines

The repository includes training scripts for each model-dataset combination. The scripts are located in the `scripts/` directory.

To train the baselines, you can use the `train.sh` script. The script takes three arguments:
- `<model>`: the model to train (e.g. `lrlm`, `cpa`, `scgpt`, `scvi`)
- `<dataset>`: the dataset to train on
- `<fold_id>`: the fold id to train on (only for Replogle)

For tahoe, the `<fold_id>` is not needed.

sh scripts/train.sh lrlm parse

## Tutorial: Training Models with Hydra Configs

This section provides a comprehensive guide on how to train models using Hydra configuration files, with **CPA (Compositional Perturbation Autoencoder)** as the primary example.

### 📋 Understanding the Config Structure

Your training configuration is organized hierarchically using **Hydra**. The main config file is located at:
```
state_sets_reproduce/configs/config.yaml
```

The config structure consists of several main sections:

```yaml
# Basic run settings
output_dir: /path/to/output      # Where to save results
name: my_experiment               # Experiment name
overwrite: false                  # Overwrite existing runs?
use_wandb: true                   # Enable WandB logging?

# WandB settings (if use_wandb: true)
wandb:
  project: perturbation-prediction
  entity: your-wandb-username
  local_wandb_dir: /path/to/wandb
  tags: [parse, cpa]

# Data configuration
data:
  kwargs:
    data_path: /path/to/data.h5ad
    embed_key: X_hvg
    pert_col: cytokine
    cell_type_key: cell_type
    batch_col: donor
    control_pert: PBS
    # ... more data options

# Model configuration
model:
  name: cpa
  kwargs:
    # Model-specific hyperparameters
    hidden_dim: 256
    n_latent: 64
    # ... more model options

# Training configuration
training:
  batch_size: 128
  max_steps: 50000
  val_freq: 1000
  lr: 0.0003
  train_seed: 42
  # ... more training options
```

### 🚀 Quick Start: Training CPA on Parse Dataset

#### Step 1: Prepare Your Data

Make sure your data file exists:
```bash
# Example: Parse PBMC cytokines dataset
DATA_PATH="/home/dennis/Parse_10M_PBMC_cytokines.h5ad"
```

#### Step 2: Create a TOML Config (Optional - for advanced splitting)

Create `configs/parse.toml` for dataset organization:
```toml
[datasets]
parse = "/home/dennis/Parse_10M_PBMC_cytokines.h5ad"

[training]
parse = "train"

# Few-shot: Hold out specific cytokines for validation/testing
[fewshot."parse.Donor1"]
val = ["G-CSF", "IFN-beta", "M-CSF", "IGF-1"]
test = ["IL-2", "IL-4", "IL-6", "TNF", "IFN-gamma"]
```

#### Step 3: Run Training with Hydra Overrides

The simplest way to train is using command-line overrides:

```bash
# Basic training command
python -m state_sets_reproduce.train \
    model=cpa \
    data.kwargs.data_path=/home/dennis/Parse_10M_PBMC_cytokines.h5ad \
    data.kwargs.embed_key=X_hvg \
    data.kwargs.pert_col=cytokine \
    data.kwargs.cell_type_key=cell_type \
    data.kwargs.batch_col=donor \
    data.kwargs.control_pert=PBS \
    training.batch_size=128 \
    training.max_steps=50000 \
    output_dir=/path/to/output/cpa_parse
```

#### Step 4: Using a Shell Script (Recommended)

Create `run_cpa_parse.sh`:
```bash
#!/bin/bash

# Set paths (NO TILDE ~, use full paths!)
OUTPUT_DIR="/home/dennis/state-benchmark/baselines/state_sets_reproduce/cpa_parse"
DATA_PATH="/home/dennis/Parse_10M_PBMC_cytokines.h5ad"

# Run training with Hydra overrides
python -m state_sets_reproduce.train \
    model=cpa \
    data.kwargs.data_path="${DATA_PATH}" \
    data.kwargs.embed_key=X_hvg \
    data.kwargs.pert_col=cytokine \
    data.kwargs.cell_type_key=cell_type \
    data.kwargs.batch_col=donor \
    data.kwargs.control_pert=PBS \
    data.kwargs.output_space=gene \
    data.kwargs.num_workers=4 \
    data.kwargs.basal_mapping_strategy=batch \
    training.batch_size=128 \
    training.max_steps=50000 \
    training.val_freq=1000 \
    training.lr=0.0003 \
    training.gradient_clip_val=10 \
    output_dir="${OUTPUT_DIR}" \
    use_wandb=true
```

Then run:
```bash
chmod +x run_cpa_parse.sh
./run_cpa_parse.sh
```

### 🔧 Understanding CPA Model Configuration

The CPA model has several important hyperparameters in `configs/model/cpa.yaml`:

```yaml
name: CPA
kwargs:
  # Latent space dimension
  n_latent: 64
  
  # Architecture
  hidden_dim: 256
  n_hidden_encoder: 768
  n_layers_encoder: 4
  n_hidden_decoder: 768
  n_layers_decoder: 3
  
  # Regularization
  dropout_rate_encoder: 0.15
  dropout_rate_decoder: 0.15
  use_batch_norm: decoder
  use_layer_norm: encoder
  
  # Loss function
  recon_loss: gauss  # or 'mse', 'nb'
  
  # Adversarial component
  n_hidden_adv: 64
  n_layers_adv: 2
  dropout_rate_adv: 0.2
  
  # Other options
  variational: false
  seed: 2025
```

### 📊 Data Configuration Options

#### Basic Data Settings
```yaml
data:
  kwargs:
    # Data source
    data_path: /path/to/data.h5ad          # Single file
    # OR
    toml_config_path: ./configs/parse.toml  # Multi-dataset with splits
    
    # Column names in AnnData
    embed_key: X_hvg              # Embedding in adata.obsm
    pert_col: cytokine            # Perturbation column
    cell_type_key: cell_type      # Cell type column
    batch_col: donor              # Batch/donor column
    control_pert: PBS             # Control label
    
    # Output configuration
    output_space: gene            # 'gene' or 'all'
    
    # Control cell mapping
    basal_mapping_strategy: batch # 'random', 'batch', 'cell_type', 'batch_cell_type'
    n_basal_samples: 1            # Number of controls per perturbed cell
    
    # Data loading
    num_workers: 4
    train_split: 0.8
    val_split: 0.1
    test_split: 0.1
    
    # Optional: Pre-computed embeddings
    perturbation_features_file: /path/to/gene_embeddings.pt
    hvg_indices_file: /path/to/hvg_indices.npy
```

#### Control Cell Mapping Strategies

**Important**: This affects how control cells are matched to perturbed cells:

1. **`random`**: Random control cells (fastest, but may have batch effects)
2. **`batch`**: Controls from same batch/donor (recommended for Parse dataset)
3. **`cell_type`**: Controls from same cell type
4. **`batch_cell_type`**: Controls from same batch AND cell type (most specific)

Example:
```bash
# Use batch-aware controls (recommended)
python -m state_sets_reproduce.train \
    model=cpa \
    data.kwargs.basal_mapping_strategy=batch \
    ...
```

### 🎯 Advanced: Few-Shot and Zero-Shot Splits

#### Using TOML for Sophisticated Splits

Create `configs/parse_fewshot.toml`:
```toml
[datasets]
parse = "/home/dennis/Parse_10M_PBMC_cytokines.h5ad"

[training]
parse = "train"

# Hold out specific cytokines for each donor
[fewshot."parse.Donor1"]
val = ["G-CSF", "IFN-beta"]
test = ["IL-2", "IL-4", "IL-6", "TNF"]

[fewshot."parse.Donor4"]
val = ["G-CSF", "IFN-beta"]
test = ["IL-2", "IL-4", "IL-6", "TNF"]

# Zero-shot: Hold out entire donors
[zeroshot]
"parse.Donor12" = "test"  # Use Donor12 for zero-shot testing
```

Then train with:
```bash
python -m state_sets_reproduce.train \
    model=cpa \
    data.kwargs.toml_config_path=./configs/parse_fewshot.toml \
    data.kwargs.embed_key=X_hvg \
    ...
```

### 💡 Common Hydra Override Patterns

#### 1. Override Single Values
```bash
python -m state_sets_reproduce.train model=cpa training.batch_size=256
```

#### 2. Override Nested Values
```bash
python -m state_sets_reproduce.train \
    model=cpa \
    data.kwargs.batch_col=donor \
    model.kwargs.hidden_dim=512
```

#### 3. Multiple Runs with Different Seeds
```bash
# Run 1
python -m state_sets_reproduce.train model=cpa training.train_seed=42

# Run 2
python -m state_sets_reproduce.train model=cpa training.train_seed=123

# Run 3
python -m state_sets_reproduce.train model=cpa training.train_seed=999
```

#### 4. Experiment with Different Learning Rates
```bash
# Low LR
python -m state_sets_reproduce.train model=cpa training.lr=0.0001

# Medium LR
python -m state_sets_reproduce.train model=cpa training.lr=0.001

# High LR
python -m state_sets_reproduce.train model=cpa training.lr=0.01
```

### 🐛 Common Issues and Solutions

#### Issue 1: Tilde (~) in Paths
**Error**: `LexerNoViableAltException: output_dir=~/...`

**Solution**: Use full paths, not `~`
```bash
# ❌ BAD
output_dir=~/my/path

# ✅ GOOD
output_dir=/home/username/my/path
```

#### Issue 2: Missing Config Values
**Error**: `KeyError: 'some_config'`

**Solution**: Provide defaults or check config files
```bash
# Use .get() with defaults in code
batch_size = cfg["training"].get("batch_size", 128)
```

#### Issue 3: File Already Open Error
**Error**: `OSError: Unable to synchronously open file (file is already open for read-only)`

**Solution**: This is fixed in the advanced dataloader by properly closing files before writing

### 📈 Monitoring Training with WandB

Enable WandB logging:
```bash
python -m state_sets_reproduce.train \
    model=cpa \
    use_wandb=true \
    wandb.project=my-perturbation-project \
    wandb.entity=my-username \
    wandb.tags=[parse,cpa,experiment1] \
    ...
```

WandB will track:
- Training and validation loss
- Learning rate
- Gradient norms
- Model hyperparameters
- System metrics (GPU, CPU, memory)

### 📁 Output Structure

After training, your output directory will contain:
```
output_dir/
├── experiment_name/
│   ├── checkpoints/
│   │   ├── last.ckpt              # Best checkpoint (symlink)
│   │   ├── step=1000.ckpt         # Periodic checkpoints
│   │   ├── step=2000.ckpt
│   │   └── final.ckpt             # Final checkpoint
│   ├── config.yaml                # Full config used
│   ├── data_module.pkl            # Saved data module
│   ├── wandb_path.txt             # WandB run path (if enabled)
│   └── version_0/
│       └── metrics.csv            # CSV logs
```

### 🔄 Complete Example Workflow

Here's a complete example for training CPA on Parse dataset:

```bash
#!/bin/bash
# complete_cpa_training.sh

# 1. Set up paths
OUTPUT_DIR="/home/dennis/experiments/cpa_parse_$(date +%Y%m%d_%H%M%S)"
DATA_PATH="/home/dennis/Parse_10M_PBMC_cytokines.h5ad"

# 2. Train CPA model
python -m state_sets_reproduce.train \
    model=cpa \
    name=cpa_parse_batch_aware \
    \
    data.kwargs.data_path="${DATA_PATH}" \
    data.kwargs.embed_key=X_hvg \
    data.kwargs.pert_col=cytokine \
    data.kwargs.cell_type_key=cell_type \
    data.kwargs.batch_col=donor \
    data.kwargs.control_pert=PBS \
    data.kwargs.output_space=gene \
    data.kwargs.basal_mapping_strategy=batch \
    data.kwargs.num_workers=8 \
    \
    model.kwargs.n_latent=64 \
    model.kwargs.hidden_dim=256 \
    model.kwargs.n_layers_encoder=4 \
    model.kwargs.n_layers_decoder=3 \
    model.kwargs.dropout_rate_encoder=0.15 \
    model.kwargs.dropout_rate_decoder=0.15 \
    model.kwargs.recon_loss=gauss \
    \
    training.batch_size=128 \
    training.max_steps=50000 \
    training.val_freq=1000 \
    training.lr=0.0003 \
    training.gradient_clip_val=10 \
    training.train_seed=42 \
    \
    output_dir="${OUTPUT_DIR}" \
    use_wandb=true \
    wandb.project=parse-perturbation \
    wandb.tags=[parse,cpa,production]

echo "Training complete! Results saved to: ${OUTPUT_DIR}"
```

### 📚 Additional Resources

- **Advanced Dataloader Documentation**: See `baselines/README_DATALOADER.md`
- **Model Implementations**: `state_sets_reproduce/models/`
- **Config Files**: `state_sets_reproduce/configs/`
- **Example Configs**: `baselines/example_config.toml`

### 🎓 Next Steps

1. Start with the basic CPA example above
2. Experiment with different hyperparameters
3. Try few-shot splits using TOML configs
4. Compare different control mapping strategies
5. Monitor results with WandB

For questions or issues, please open an issue on GitHub!

