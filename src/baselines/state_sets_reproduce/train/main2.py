import json
import os
from pathlib import Path
import pickle
import shutil
import re
from os.path import join, exists
from typing import List

import hydra
import torch
import sys

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.plugins.precision import MixedPrecision
from advanced_dataloader import AdvancedPerturbationDataModule

from state_sets_reproduce.models import (
    CPAPerturbationModel,
    SCVIPerturbationModel,
)
from state_sets_reproduce.callbacks import BatchSpeedMonitorCallback

import logging


logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("medium")


os.environ["WANDB_API_KEY"] = "eb06269b6054eee7bba6309cbc96a7624274cb32"


def get_lightning_module(model_type: str, data_config: dict, model_config: dict, training_config: dict, var_dims: dict):
    """Create model instance based on config."""
    # combine the model config and training config
    module_config = {**model_config, **training_config}
    module_config["embed_key"] = data_config["embed_key"]
    module_config["output_space"] = data_config["output_space"]
    module_config["gene_names"] = var_dims["gene_names"]
    
    print(f"Found {len(module_config['gene_names'])} genes in the data")
    module_config["batch_size"] = training_config["batch_size"]

    if data_config["output_space"] == "gene":
        gene_dim = var_dims["hvg_dim"]
    else:
        gene_dim = var_dims["gene_dim"]

    if model_type.lower() == "cpa":
        return CPAPerturbationModel(
            input_dim=var_dims["input_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            gene_dim=gene_dim,
            **module_config,
        )

    elif model_type.lower() == "scvi":
        return SCVIPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")



def get_latest_step_checkpoint(directory):
    # Get all checkpoint files
    files = os.listdir(directory)
    
    # Extract step numbers using regex, excluding files with 'val_loss'
    step_numbers = []
    for f in files:
        if f.startswith('step=') and 'val_loss' not in f:
            # Extract the number between 'step=' and '.ckpt'
            match = re.search(r'step=(\d+)(?:-v\d+)?\.ckpt', f)
            if match:
                step_numbers.append(int(match.group(1)))
    
    if not step_numbers:
        raise ValueError("No checkpoint files found")
        
    # Get the maximum step number
    max_step = max(step_numbers)
    
    # Construct the checkpoint path
    checkpoint_path = join(directory, f"step={max_step}.ckpt")
    
    return checkpoint_path

def get_loggers(
    output_dir: str,
    name: str,
    wandb_project: str,
    wandb_entity: str,
    local_wandb_dir: str,
    use_wandb: bool = False,
    cfg: dict = None,
) -> List:
    """Set up logging to local CSV and optionally WandB."""
    # Always use CSV logger
    csv_logger = CSVLogger(save_dir=output_dir, name=name, version=0)
    loggers = [csv_logger]

    # Add WandB if requested
    if use_wandb:
        wandb_logger = WandbLogger(
            name=name,
            project=wandb_project,
            entity=wandb_entity,
            dir=local_wandb_dir,
            tags=cfg["wandb"].get("tags", []) if cfg else [],
        )
        if cfg is not None:
            wandb_logger.experiment.config.update(cfg)
        loggers.append(wandb_logger)

    return loggers


def get_checkpoint_callbacks(output_dir: str, name: str, val_freq: int, ckpt_every_n_steps: int) -> List[ModelCheckpoint]:
    """Create checkpoint callbacks based on validation frequency."""
    checkpoint_dir = join(output_dir, name, "checkpoints")
    callbacks = []

    # Save best checkpoint based on validation loss
    best_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step={step}-val_loss={val_loss:.4f}",
        save_last='link',  # Will create last.ckpt symlink to best checkpoint
        monitor="val_loss",
        mode="min",
        save_top_k=1,  # Only keep the best checkpoint
        every_n_train_steps=val_freq,
    )
    callbacks.append(best_ckpt)

    # Also save periodic checkpoints (without affecting the "last" symlink)
    periodic_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{step}",
        save_last=False,  # Don't create/update symlink
        every_n_train_steps=ckpt_every_n_steps,
        save_top_k=-1,  # Keep all periodic checkpoints
    )
    callbacks.append(periodic_ckpt)

    return callbacks


from pathlib import Path
import inspect, state_sets_reproduce

CONFIG_DIR = (Path(__file__).resolve().parents[1] / "configs").as_posix()
print("Hydra CONFIG_DIR:", CONFIG_DIR)  # optional sanity print

@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Main training function."""
    # Convert config to YAML for logging
    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    print(cfg_yaml)

    # Setup output directory
    run_output_dir = join(cfg["output_dir"], cfg["name"])
    if os.path.exists(run_output_dir) and cfg["overwrite"]:
        print(f"Output dir {run_output_dir} already exists, overwriting")
        shutil.rmtree(run_output_dir)
    os.makedirs(run_output_dir, exist_ok=True)

    # Set up wandb directory if needed
    if cfg["use_wandb"]:
        os.makedirs(cfg["wandb"]["local_wandb_dir"], exist_ok=True)

    with open(join(run_output_dir, "config.yaml"), "w") as f:
        f.write(cfg_yaml)

    # Set random seeds
    if "train_seed" in cfg["training"]:
        pl.seed_everything(cfg["training"]["train_seed"])
        
    # if the provided pert_col is drugname_drugconc, hard code the value of control pert
    # this is because it's surprisingly hard to specify a list of tuples in the config as a string
    cfg["model"]["kwargs"]["control_pert"] = cfg["data"]["kwargs"]["control_pert"]
    if cfg["data"]["kwargs"]["pert_col"] == "drugname_drugconc":
        cfg["data"]["kwargs"]["control_pert"] = "[('DMSO_TF', 0.0, 'uM')]"

    # Initialize data module. this is backwards compatible with previous configs
    try:
        sentence_len = cfg["model"]["cell_set_len"]
    except KeyError:
        if cfg["model"]["name"].lower() in ["cpa", "scvi", "lowranklinear"] or cfg["model"]["name"].lower().startswith("scgpt"):
            if "cell_sentence_len" in cfg["model"]["kwargs"] and cfg["model"]["kwargs"]["cell_sentence_len"] > 1:
                sentence_len = cfg["model"]["kwargs"]["cell_sentence_len"]
                cfg["training"]["batch_size"] = 1
            else:
                sentence_len = 1
        else:
            sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["n_positions"]


    # Initialize data module using AdvancedPerturbationDataModule
    # This supports all cell-load features: multi-dataset, TOML config, 
    # zero-shot/few-shot splits, batch-aware control mapping, etc.
    
    # Check if TOML config is provided
    toml_config = cfg["data"]["kwargs"].get("toml_config_path", None)
    
    if toml_config:
        # Use TOML-based configuration
        logger.info(f"Using TOML configuration: {toml_config}")
        dm = AdvancedPerturbationDataModule(
            toml_config_path=toml_config,
            embed_key=cfg["data"]["kwargs"]["embed_key"],
            pert_col=cfg["data"]["kwargs"]["pert_col"],
            cell_type_key=cfg["data"]["kwargs"]["cell_type_key"],
            batch_col=cfg["data"]["kwargs"]["batch_col"],
            control_pert=cfg["data"]["kwargs"]["control_pert"],
            output_space=cfg["data"]["kwargs"]["output_space"],
            basal_mapping_strategy=cfg["data"]["kwargs"].get("basal_mapping_strategy", "batch"),
            n_basal_samples=cfg["data"]["kwargs"].get("n_basal_samples", 1),
            perturbation_features_file=cfg["data"]["kwargs"].get("perturbation_features_file", None),
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["data"]["kwargs"].get("num_workers", 4),
            random_seed=cfg["training"].get("train_seed", 42),
        )
    else:
        # Use simple configuration with data_path
        logger.info("Using simple data path configuration")
        dm = AdvancedPerturbationDataModule(
            data_path=cfg["data"]["kwargs"]["data_path"],
            embed_key=cfg["data"]["kwargs"]["embed_key"],
            pert_col=cfg["data"]["kwargs"]["pert_col"],
            cell_type_key=cfg["data"]["kwargs"]["cell_type_key"],
            batch_col=cfg["data"]["kwargs"]["batch_col"],
            control_pert=cfg["data"]["kwargs"]["control_pert"],
            output_space=cfg["data"]["kwargs"]["output_space"],
            basal_mapping_strategy=cfg["data"]["kwargs"].get("basal_mapping_strategy", "batch"),
            n_basal_samples=cfg["data"]["kwargs"].get("n_basal_samples", 1),
            perturbation_features_file=cfg["data"]["kwargs"].get("perturbation_features_file", None),
            hvg_indices_file=cfg["data"]["kwargs"].get("hvg_indices_file", None),
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["data"]["kwargs"].get("num_workers", 4),
            train_split=cfg["data"]["kwargs"].get("train_split", 0.8),
            val_split=cfg["data"]["kwargs"].get("val_split", 0.1),
            test_split=cfg["data"]["kwargs"].get("test_split", 0.1),
            random_seed=cfg["training"].get("train_seed", 42),
        )
    
    dm.setup()
    
    logger.info("Data module setup complete!")

    if cfg["model"]["name"].lower() in ["cpa", "scvi", "lowranklinear"] or cfg["model"]["name"].lower().startswith("scgpt"):
        cfg["model"]["kwargs"]["n_cell_types"] = len(dm.cell_type_onehot_map)
        cfg["model"]["kwargs"]["n_perts"] = len(dm.pert_onehot_map)
        cfg["model"]["kwargs"]["n_batches"] = len(dm.batch_onehot_map)

    # Create model (always needed!)
    model = get_lightning_module(
        cfg["model"]["name"],
        cfg["data"]["kwargs"],
        cfg["model"]["kwargs"],
        cfg["training"],
        dm.get_var_dims(),
    )

    # Set up logging (CSV + optional WandB)
    loggers = get_loggers(
        output_dir=cfg["output_dir"],
        name=cfg["name"],
        wandb_project=cfg["wandb"]["project"],
        wandb_entity=cfg["wandb"]["entity"],
        local_wandb_dir=cfg["wandb"]["local_wandb_dir"],
        use_wandb=cfg["use_wandb"],
        cfg=cfg,
    )

    # If using wandb, store the run path in a text file for eval
    for lg in loggers:
        if isinstance(lg, WandbLogger):
            wandb_info_path = os.path.join(run_output_dir, "wandb_path.txt")
            with open(wandb_info_path, "w") as f:
                f.write(lg.experiment.path)
            break

    # Set up callbacks
    ckpt_callbacks = get_checkpoint_callbacks(
        cfg["output_dir"],
        cfg["name"],
        cfg["training"]["val_freq"],
        cfg["training"].get("ckpt_every_n_steps", 4000),
    )
    # Add BatchSpeedMonitorCallback to log batches per second to wandb
    batch_speed_monitor = BatchSpeedMonitorCallback()
    callbacks = ckpt_callbacks + [batch_speed_monitor]

    logger.info('Loggers and callbacks set up.')
    
    if cfg["model"]["name"].lower().startswith("scgpt"):
        plugins = [
            MixedPrecision(
                precision="bf16-mixed",
                device="cuda",
            )
        ]
    else:
        plugins = []

    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    # Decide on trainer params
    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=1,
        max_steps=cfg["training"].get("max_steps", -1),  # for normal models
        max_epochs=cfg["training"].get("max_epochs", -1), 
        check_val_every_n_epoch=None,
        val_check_interval=cfg["training"]["val_freq"],
        logger=loggers,
        plugins=plugins,
        callbacks=callbacks,
        gradient_clip_val=cfg["training"].get("gradient_clip_val", None),
    )
    
    if cfg['model']['name'].lower() == 'cpa':
        trainer_kwargs['gradient_clip_val'] = 0


    # Build trainer
    trainer = pl.Trainer(**trainer_kwargs)

    # Load checkpoint if exists
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "last.ckpt")
    if not exists(checkpoint_path):
        checkpoint_path = None
    else:
        logging.info(f"!! Resuming training from {checkpoint_path} !!")

    logger.info('Starting trainer fit.')

    # Train
    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=checkpoint_path,
    )

    # at this point if checkpoint_path does not exist, manually create one
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "final.ckpt")
    if not exists(checkpoint_path):
        trainer.save_checkpoint(checkpoint_path)
        
    # save the data_module
    with open(join(run_output_dir, "data_module.pkl"), "wb") as f:
        pickle.dump(dm, f)

if __name__ == "__main__":
    train()