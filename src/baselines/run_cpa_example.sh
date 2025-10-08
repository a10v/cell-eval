#!/bin/bash
# Example script to train CPA model on Parse dataset
# This script demonstrates how to run CPA training from the cell-eval baselines directory

# Set paths (use FULL PATHS, not ~)
OUTPUT_DIR="/home/dennis/cell-eval/outputs/cpa_parse_example"
DATA_PATH="/home/dennis/Parse_10M_PBMC_cytokines.h5ad"

# Make sure we're in the right directory
cd /home/dennis/cell-eval/src/baselines

# Run training with Hydra overrides
python -m state_sets_reproduce.train \
    model=cpa \
    name=cpa_parse_example \
    \
    data.kwargs.data_path="${DATA_PATH}" \
    data.kwargs.embed_key=X_hvg \
    data.kwargs.pert_col=cytokine \
    data.kwargs.cell_type_key=cell_type \
    data.kwargs.batch_col=donor \
    data.kwargs.control_pert=PBS \
    data.kwargs.output_space=gene \
    data.kwargs.basal_mapping_strategy=batch \
    data.kwargs.num_workers=4 \
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
    training.max_steps=10000 \
    training.val_freq=1000 \
    training.lr=0.0003 \
    training.train_seed=42 \
    \
    output_dir="${OUTPUT_DIR}" \
    use_wandb=false

echo "Training complete! Results saved to: ${OUTPUT_DIR}"

