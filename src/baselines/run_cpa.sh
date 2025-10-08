#!/bin/bash

#SBATCH --job-name=hvg
#SBATCH --partition=job_a100
#SBATCH --time=08:00:00        # must be <= 02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --output=cpa_train.out

OUTPUT_DIR="/home/dennis/cell-eval/src/baselines/outputs/cpa_parse_example"
DATA_PATH="/home/dennis/Parse_10M_PBMC_cytokines_hvg.h5ad"

cd /home/dennis/cell-eval/src/baselines

source /home/dennis/state-benchmark/baselines/.venv/bin/activate


# Run training with Hydra overrides
python3 -m state_sets_reproduce.train \
    model=cpa \
    name=cpa_parse_example \
    \
    data.kwargs.toml_config_path=/home/dennis/cell-eval/src/baselines/state_sets_reproduce/configs/parse.toml \
    data.kwargs.embed_key=hvg \
    data.kwargs.pert_col=cytokine \
    data.kwargs.cell_type_key=donor \
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




