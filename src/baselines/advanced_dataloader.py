"""
Advanced Perturbation Data Loader with cell-load Features

A comprehensive PyTorch data loading library for single-cell perturbation data
that replicates and extends the functionality of the cell-load package.

Key Features:
- Multi-dataset support with TOML configuration
- Advanced splitting: zero-shot (cell types) and few-shot (perturbations)
- Batch-aware control cell mapping strategies
- Pre-computed gene embeddings support
- Comprehensive preprocessing utilities
- Production-ready quality control
- HVG (Highly Variable Gene) support for Parse datasets
"""

import logging
import numpy as np
import torch
import toml
import h5py
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Literal
from collections import defaultdict
import anndata as ad
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PerturbationDataset(Dataset):
    """
    Advanced dataset for single-cell perturbation data.
    
    Supports:
    - Multiple data files from a directory
    - HVG embeddings extraction
    - Batch-aware control cell mapping
    - Pre-computed perturbation embeddings
    - Quality control filtering
    """

    def __init__(
        self,
        data_paths: Union[str, Path, List[Union[str, Path]]],
        embed_key: str = "X_hvg",
        pert_col: str = "gene",
        cell_type_key: str = "cell_type",
        batch_col: str = "donor",
        control_pert: str = "non-targeting",
        output_space: str = "gene",
        basal_mapping_strategy: Literal["random", "batch", "cell_type", "batch_cell_type"] = "batch",
        n_basal_samples: int = 1,
        should_yield_control_cells: bool = True,
        perturbation_features_file: Optional[str] = None,
        barcode: bool = False,
        hvg_indices_file: str = "/home/dennis/state-benchmark/baselines/hvg_indices.npy",
        extract_hvg_if_missing: bool = True,
        random_seed: int = 42,
        **kwargs
    ):
        """
        Initialize the dataset.

        Args:
            data_paths: Path(s) to H5/AnnData file(s) or directory containing them
            embed_key: Key in obsm for cell embeddings (e.g., "X_hvg")
            pert_col: Column name for perturbations in obs
            cell_type_key: Column name for cell types in obs
            batch_col: Column name for batches/donors in obs
            control_pert: Value representing control cells
            output_space: Output space ("gene", "all", "embedding")
            basal_mapping_strategy: Strategy for mapping perturbed to control cells
                - "random": Random control cells
                - "batch": Controls from same batch
                - "cell_type": Controls from same cell type
                - "batch_cell_type": Controls from same batch and cell type
            n_basal_samples: Number of control cells per perturbed cell
            should_yield_control_cells: Whether to include control cells
            perturbation_features_file: Path to .pt/.h5 file with perturbation embeddings
            barcode: Whether to include cell barcodes
            hvg_indices_file: Path to HVG indices .npy file
            extract_hvg_if_missing: If True, extract HVG if not found
            random_seed: Random seed for reproducibility
        """
        super().__init__()

        self.embed_key = embed_key
        self.pert_col = pert_col
        self.cell_type_key = cell_type_key
        self.batch_col = batch_col
        self.control_pert = control_pert
        self.output_space = output_space
        self.basal_mapping_strategy = basal_mapping_strategy
        self.n_basal_samples = n_basal_samples
        self.should_yield_control_cells = should_yield_control_cells
        self.perturbation_features_file = perturbation_features_file
        self.barcode = barcode
        self.hvg_indices_file = hvg_indices_file
        self.extract_hvg_if_missing = extract_hvg_if_missing
        self.random_seed = random_seed

        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Load data
        print("Loading data path...")
        self.data_paths = self._resolve_data_paths(data_paths)
        print("Loading all data...")
        self._load_all_data()
        print("Setting up mappings...")
        self._setup_mappings()
        print("Loading perturbation features...")
        self._load_perturbation_features()
        print("Creating perturbation control pairs...")
        self._create_perturbation_control_pairs()

    def _resolve_data_paths(self, data_paths: Union[str, Path, List]) -> List[Path]:
        """Resolve data paths - if directory, find all h5/h5ad files."""
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        
        resolved = []
        for path in data_paths:
            path = Path(path)
            if path.is_dir():
                # Find all h5/h5ad files in directory
                h5_files = list(path.glob("*.h5")) + list(path.glob("*.h5ad"))
                resolved.extend(h5_files)
                logger.info(f"Found {len(h5_files)} data files in {path}")
            elif path.is_file():
                resolved.append(path)
            else:
                logger.warning(f"Path not found: {path}")
        
        if not resolved:
            raise ValueError(f"No data files found in: {data_paths}")
        
        return resolved

    def _extract_hvg_to_obsm(self, data_path: Path):
        """Extract HVG embeddings and save to obsm/X_hvg using backed mode efficiently."""
        logger.info(f"Extracting HVG embeddings for {data_path.name}...")
        
        # Step 1: Load in backed mode to extract HVG data (memory efficient!)
        adata_backed = ad.read_h5ad(data_path, backed="r")
        
        # Get HVG indices from the highly_variable column in var
        if "highly_variable" not in adata_backed.var.columns:
            raise ValueError(f"'highly_variable' column not found in adata.var for {data_path.name}")
        
        hvg_mask = adata_backed.var["highly_variable"].values
        hvg_indices = np.where(hvg_mask)[0]
        
        logger.info(f"Found {len(hvg_indices)} HVGs out of {adata_backed.shape[1]} total genes")
        
        # Extract HVG columns from expression matrix
        from scipy import sparse
        logger.info(f"Extracting {len(hvg_indices)} HVG columns from {adata_backed.shape}...")
        if sparse.issparse(adata_backed.X):
            X_hvg = adata_backed.X[:, hvg_indices].toarray().astype(np.float32)
        else:
            X_hvg = adata_backed.X[:, hvg_indices].astype(np.float32)
        
        logger.info(f"Created X_hvg array: {X_hvg.shape}")
        
        # Step 2: Close the backed file BEFORE writing
        adata_backed.file.close()
        del adata_backed
        
        # Step 3: Load again (not backed) to add HVG and write back
        logger.info("Reopening file to save HVG...")
        adata = ad.read_h5ad(data_path)
        adata.obsm["X_hvg"] = X_hvg
        
        logger.info(f"Writing updated file to {data_path.name}...")
        adata.write_h5ad(data_path)
        
        logger.info(f"✓ Successfully saved obsm/X_hvg to {data_path.name}")

    def _load_all_data(self):
        """Load all data files and concatenate."""
        all_embeddings = []
        all_perturbations = []
        all_cell_types = []
        all_batches = []
        all_barcodes = []
        all_gene_names = None
        all_dataset_labels = []
        
        for i, data_path in enumerate(self.data_paths):
            logger.info(f"Loading {data_path.name}...")
            
            # First, check if we need to extract HVG (peek without loading full data)
            need_hvg_extraction = False
            if self.embed_key == "X_hvg":
                # Quick check with h5py to see if X_hvg exists
                with h5py.File(str(data_path), 'r') as f:
                    if 'obsm' not in f or 'X_hvg' not in f['obsm']:
                        need_hvg_extraction = True
            
            # Extract HVG if needed (before loading data)
            if need_hvg_extraction:
                if self.extract_hvg_if_missing:
                    logger.warning(f"X_hvg not found in {data_path.name}, extracting...")
                    # Extract HVG embeddings using highly_variable column from adata.var
                    self._extract_hvg_to_obsm(data_path)
                else:
                    raise ValueError(f"X_hvg not found in {data_path.name} and extract_hvg_if_missing=False")
            
            # Now load AnnData (with HVG already present)
            adata = ad.read_h5ad(data_path)
            
            # Extract embeddings
            if self.embed_key in adata.obsm:
                embeddings = adata.obsm[self.embed_key]
            elif self.embed_key == "X":
                embeddings = adata.X
            else:
                raise ValueError(f"Embedding key '{self.embed_key}' not found")
            
            # Convert sparse to dense if needed
            if hasattr(embeddings, "toarray"):
                embeddings = embeddings.toarray()
            
            all_embeddings.append(embeddings)
            all_perturbations.append(adata.obs[self.pert_col].values)
            all_cell_types.append(adata.obs[self.cell_type_key].values)
            all_batches.append(adata.obs[self.batch_col].values)
            all_dataset_labels.extend([data_path.stem] * len(adata))
            
            # Barcodes
            if self.barcode and 'barcode' in adata.obs.columns:
                all_barcodes.append(adata.obs['barcode'].values)
            else:
                all_barcodes.append(np.array([f"cell_{j}" for j in range(len(adata))]))
            
            # Gene names (should be consistent across files)
            # If using HVG, store only HVG gene names
            if all_gene_names is None:
                if self.embed_key == "X_hvg" and "highly_variable" in adata.var.columns:
                    # Store only HVG gene names
                    hvg_mask = adata.var["highly_variable"].values
                    all_gene_names = adata.var_names[hvg_mask].tolist()
                    logger.info(f"Storing {len(all_gene_names)} HVG gene names")
                else:
                    # Store all gene names
                    all_gene_names = adata.var_names.tolist()
        
        # Concatenate all data
        self.embeddings = np.vstack(all_embeddings)
        self.perturbations = np.concatenate(all_perturbations)
        self.cell_types = np.concatenate(all_cell_types)
        self.batches = np.concatenate(all_batches)
        self.cell_barcodes = np.concatenate(all_barcodes) if all_barcodes else None
        self.dataset_labels = np.array(all_dataset_labels)
        self.gene_names = all_gene_names
        
        logger.info(f"Loaded total {len(self.embeddings)} cells with {len(self.gene_names)} genes")

    def _setup_mappings(self):
        """Set up encodings for categorical variables."""
        # Encoders
        self.pert_encoder = LabelEncoder()
        self.cell_type_encoder = LabelEncoder()
        self.batch_encoder = LabelEncoder()
        
        # Fit encoders
        self.pert_encoder.fit(self.perturbations)
        self.cell_type_encoder.fit(self.cell_types)
        self.batch_encoder.fit(self.batches)
        
        # Get encoded values
        self.pert_codes = self.pert_encoder.transform(self.perturbations)
        self.cell_type_codes = self.cell_type_encoder.transform(self.cell_types)
        self.batch_codes = self.batch_encoder.transform(self.batches)
        
        # Store names
        self.pert_names = self.pert_encoder.classes_.tolist()
        self.cell_type_names = self.cell_type_encoder.classes_.tolist()
        self.batch_names = self.batch_encoder.classes_.tolist()
        
        # Control perturbation code
        try:
            self.control_pert_code = self.pert_encoder.transform([self.control_pert])[0]
        except ValueError:
            logger.warning(f"Control perturbation '{self.control_pert}' not found in data")
            self.control_pert_code = -1
        
        logger.info(f"Found {len(self.pert_names)} perturbations, "
                   f"{len(self.cell_type_names)} cell types, "
                   f"{len(self.batch_names)} batches")

    def _load_perturbation_features(self):
        """Load pre-computed perturbation embeddings if provided."""
        if self.perturbation_features_file is None:
            # Use one-hot encoding
            self.pert_features = None
            logger.info("Using one-hot encoding for perturbations")
            return
        
        pert_file = Path(self.perturbation_features_file)
        
        if pert_file.suffix == '.pt':
            # PyTorch format
            pert_dict = torch.load(pert_file)
            self.pert_features = np.zeros((len(self.pert_names), 
                                           list(pert_dict.values())[0].shape[0]))
            for i, pert in enumerate(self.pert_names):
                if pert in pert_dict:
                    self.pert_features[i] = pert_dict[pert].cpu().numpy()
                else:
                    logger.warning(f"Perturbation '{pert}' not in embeddings file")
        
        elif pert_file.suffix == '.h5':
            # HDF5 format
            with h5py.File(pert_file, 'r') as f:
                # Assume structure: f[pert_name] = embedding
                first_key = list(f.keys())[0]
                emb_dim = f[first_key].shape[0]
                self.pert_features = np.zeros((len(self.pert_names), emb_dim))
                for i, pert in enumerate(self.pert_names):
                    if pert in f:
                        self.pert_features[i] = f[pert][:]
                    else:
                        logger.warning(f"Perturbation '{pert}' not in embeddings file")
        else:
            raise ValueError(f"Unsupported perturbation features format: {pert_file.suffix}")
        
        logger.info(f"Loaded perturbation features: {self.pert_features.shape}")

    def _create_perturbation_control_pairs(self):
        """Create perturbation-control cell pairs based on mapping strategy."""
        # Find control and perturbed cells
        control_mask = self.pert_codes == self.control_pert_code
        perturbed_mask = ~control_mask
        
        self.control_indices = np.where(control_mask)[0]
        self.perturbed_indices = np.where(perturbed_mask)[0]
        
        if len(self.control_indices) == 0:
            raise ValueError(f"No control cells found with perturbation '{self.control_pert}'")
        
        # Create control pairs based on strategy
        n_perturbed = len(self.perturbed_indices)
        self.control_pairs = np.zeros((n_perturbed, self.n_basal_samples), dtype=np.int64)
        
        if self.basal_mapping_strategy == "random":
            # Random control cells
            for i in range(self.n_basal_samples):
                self.control_pairs[:, i] = np.random.choice(
                    self.control_indices,
                    size=n_perturbed,
                    replace=True
                )
        
        elif self.basal_mapping_strategy == "batch":
            # Controls from same batch
            for idx, pert_idx in enumerate(self.perturbed_indices):
                pert_batch = self.batch_codes[pert_idx]
                # Find controls in same batch
                valid_controls = self.control_indices[
                    self.batch_codes[self.control_indices] == pert_batch
                ]
                if len(valid_controls) == 0:
                    # Fallback to random
                    valid_controls = self.control_indices
                self.control_pairs[idx] = np.random.choice(
                    valid_controls,
                    size=self.n_basal_samples,
                    replace=True
                )
        
        elif self.basal_mapping_strategy == "cell_type":
            # Controls from same cell type
            for idx, pert_idx in enumerate(self.perturbed_indices):
                pert_cell_type = self.cell_type_codes[pert_idx]
                valid_controls = self.control_indices[
                    self.cell_type_codes[self.control_indices] == pert_cell_type
                ]
                if len(valid_controls) == 0:
                    valid_controls = self.control_indices
                self.control_pairs[idx] = np.random.choice(
                    valid_controls,
                    size=self.n_basal_samples,
                    replace=True
                )
        
        elif self.basal_mapping_strategy == "batch_cell_type":
            # Controls from same batch AND cell type
            for idx, pert_idx in enumerate(self.perturbed_indices):
                pert_batch = self.batch_codes[pert_idx]
                pert_cell_type = self.cell_type_codes[pert_idx]
                valid_controls = self.control_indices[
                    (self.batch_codes[self.control_indices] == pert_batch) &
                    (self.cell_type_codes[self.control_indices] == pert_cell_type)
                ]
                if len(valid_controls) == 0:
                    # Fallback: same batch only
                    valid_controls = self.control_indices[
                        self.batch_codes[self.control_indices] == pert_batch
                    ]
                if len(valid_controls) == 0:
                    # Final fallback: random
                    valid_controls = self.control_indices
                self.control_pairs[idx] = np.random.choice(
                    valid_controls,
                    size=self.n_basal_samples,
                    replace=True
                )
        
        logger.info(f"Created {n_perturbed} perturbation-control pairs "
                   f"(strategy: {self.basal_mapping_strategy}, n_basal: {self.n_basal_samples})")

    def __len__(self):
        """Return number of samples."""
        return len(self.perturbed_indices)

    def __getitem__(self, idx):
        """Get a perturbation-control pair."""
        pert_idx = self.perturbed_indices[idx]
        ctrl_indices = self.control_pairs[idx]
        
        # Get embeddings
        pert_emb = torch.FloatTensor(self.embeddings[pert_idx])
        ctrl_embs = [torch.FloatTensor(self.embeddings[ci]) for ci in ctrl_indices]
        
        # Average control embeddings if multiple
        if len(ctrl_embs) > 1:
            ctrl_emb = torch.stack(ctrl_embs).mean(dim=0)
        else:
            ctrl_emb = ctrl_embs[0]
        
        # Get perturbation features
        pert_code = self.pert_codes[pert_idx]
        if self.pert_features is not None:
            pert_emb_feat = torch.FloatTensor(self.pert_features[pert_code])
        else:
            # One-hot encoding
            pert_emb_feat = torch.zeros(len(self.pert_names))
            pert_emb_feat[pert_code] = 1.0
        
        # Get cell type one-hot
        cell_type_code = self.cell_type_codes[pert_idx]
        cell_type_onehot = torch.zeros(len(self.cell_type_names))
        cell_type_onehot[cell_type_code] = 1.0
        
        # Get batch one-hot
        batch_code = self.batch_codes[pert_idx]
        batch_onehot = torch.zeros(len(self.batch_names))
        batch_onehot[batch_code] = 1.0
        
        # Prepare output
        batch = {
            "pert_cell_emb": pert_emb,
            "ctrl_cell_emb": ctrl_emb,
            "pert_emb": pert_emb_feat,
            "cell_type_onehot": cell_type_onehot,
            "batch": batch_onehot,
            "pert_name": self.perturbations[pert_idx],
            "cell_type": self.cell_types[pert_idx],
            "batch_name": self.batches[pert_idx],
            "dataset": self.dataset_labels[pert_idx],
        }
        
        # Add barcodes if requested
        if self.barcode and self.cell_barcodes is not None:
            batch["pert_cell_barcode"] = self.cell_barcodes[pert_idx]
            batch["ctrl_cell_barcode"] = self.cell_barcodes[ctrl_indices[0]]
        
        return batch


class AdvancedPerturbationDataModule:
    """
    Advanced data module with TOML configuration and sophisticated splitting.
    
    Supports:
    - TOML configuration files
    - Multi-dataset loading
    - Zero-shot splits (entire cell types held out)
    - Few-shot splits (specific perturbations held out)
    - All cell-load features
    """

    def __init__(
        self,
        toml_config_path: Optional[str] = None,
        data_path: Optional[Union[str, Path]] = None,
        embed_key: str = "X_hvg",
        pert_col: str = "gene",
        cell_type_key: str = "cell_type",
        batch_col: str = "donor",
        control_pert: str = "non-targeting",
        output_space: str = "gene",
        basal_mapping_strategy: str = "batch",
        n_basal_samples: int = 1,
        should_yield_control_cells: bool = True,
        perturbation_features_file: Optional[str] = None,
        barcode: bool = False,
        hvg_indices_file: Optional[str] = None,
        batch_size: int = 128,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42,
        **dataset_kwargs
    ):
        """
        Initialize the data module.

        Args:
            toml_config_path: Path to TOML configuration file (optional)
            data_path: Path to data file/directory (if not using TOML)
            ... (other parameters same as PerturbationDataset)
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            train_split: Train split fraction
            val_split: Validation split fraction
            test_split: Test split fraction
            random_seed: Random seed
        """
        self.toml_config_path = toml_config_path
        self.data_path = data_path
        self.embed_key = embed_key
        self.pert_col = pert_col
        self.cell_type_key = cell_type_key
        self.batch_col = batch_col
        self.control_pert = control_pert
        self.output_space = output_space
        self.basal_mapping_strategy = basal_mapping_strategy
        self.n_basal_samples = n_basal_samples
        self.should_yield_control_cells = should_yield_control_cells
        self.perturbation_features_file = perturbation_features_file
        self.barcode = barcode
        self.hvg_indices_file = hvg_indices_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.dataset_kwargs = dataset_kwargs
        
        # Parse TOML config if provided
        if toml_config_path:
            self._parse_toml_config()
        
        # Initialize datasets
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # For compatibility
        self.pert_onehot_map = None
        self.cell_type_onehot_map = None
        self.batch_onehot_map = None

    def _parse_toml_config(self):
        """Parse TOML configuration file."""
        config = toml.load(self.toml_config_path)
        
        # Parse datasets
        if "datasets" in config:
            self.datasets_config = config["datasets"]
            logger.info(f"Loaded {len(self.datasets_config)} datasets from TOML")
        
        # Parse training splits
        if "training" in config:
            self.training_config = config["training"]
        
        # Parse zero-shot splits
        if "zeroshot" in config:
            self.zeroshot_config = config["zeroshot"]
            logger.info(f"Zero-shot config: {self.zeroshot_config}")
        else:
            self.zeroshot_config = {}
        
        # Parse few-shot splits
        if "fewshot" in config:
            self.fewshot_config = config["fewshot"]
            logger.info(f"Few-shot config: {self.fewshot_config}")
        else:
            self.fewshot_config = {}

    def setup(self):
        """Set up train/val/test datasets with sophisticated splitting."""
        # Determine data paths
        if self.toml_config_path and hasattr(self, 'datasets_config'):
            # Use TOML config
            all_paths = list(self.datasets_config.values())
        elif self.data_path:
            # Use single path
            all_paths = [self.data_path]
        else:
            raise ValueError("Must provide either toml_config_path or data_path")
        
        # Create full dataset
        self.dataset = PerturbationDataset(
            data_paths=all_paths,
            embed_key=self.embed_key,
            pert_col=self.pert_col,
            cell_type_key=self.cell_type_key,
            batch_col=self.batch_col,
            control_pert=self.control_pert,
            output_space=self.output_space,
            basal_mapping_strategy=self.basal_mapping_strategy,
            n_basal_samples=self.n_basal_samples,
            should_yield_control_cells=self.should_yield_control_cells,
            perturbation_features_file=self.perturbation_features_file,
            barcode=self.barcode,
            hvg_indices_file=self.hvg_indices_file,
            random_seed=self.random_seed,
            **self.dataset_kwargs
        )
        
        # Set up compatibility attributes
        self.pert_onehot_map = {p: i for i, p in enumerate(self.dataset.pert_names)}
        self.cell_type_onehot_map = {ct: i for i, ct in enumerate(self.dataset.cell_type_names)}
        self.batch_onehot_map = {b: i for i, b in enumerate(self.dataset.batch_names)}
        self.pert_names = self.dataset.pert_names
        self.gene_names = self.dataset.gene_names
        
        # Create splits
        if self.toml_config_path and (self.zeroshot_config or self.fewshot_config):
            # Advanced splitting
            self._create_advanced_splits()
        else:
            # Simple random splitting
            self._create_simple_splits()
        
        logger.info(f"Created splits: {len(self.train_dataset)} train, "
                   f"{len(self.val_dataset)} val, {len(self.test_dataset)} test")

    def _create_simple_splits(self):
        """Create simple random train/val/test splits."""
        from torch.utils.data import random_split
        
        n_samples = len(self.dataset)
        n_train = int(n_samples * self.train_split)
        n_val = int(n_samples * self.val_split)
        n_test = n_samples - n_train - n_val
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.random_seed)
        )

    def _create_advanced_splits(self):
        """Create advanced splits based on TOML config (zero-shot/few-shot)."""
        train_indices = []
        val_indices = []
        test_indices = []
        
        # Process zero-shot splits (entire cell types)
        zeroshot_val_cell_types = set()
        zeroshot_test_cell_types = set()
        
        for key, split in self.zeroshot_config.items():
            # key format: "dataset.cell_type"
            parts = key.split(".")
            if len(parts) >= 2:
                cell_type = parts[-1]
                if split == "val":
                    zeroshot_val_cell_types.add(cell_type)
                elif split == "test":
                    zeroshot_test_cell_types.add(cell_type)
        
        # Process few-shot splits (specific perturbations)
        fewshot_val_perts = defaultdict(set)
        fewshot_test_perts = defaultdict(set)
        
        for key, splits in self.fewshot_config.items():
            # key format: "dataset.cell_type"
            if "val" in splits:
                for pert in splits["val"]:
                    fewshot_val_perts[key].add(pert)
            if "test" in splits:
                for pert in splits["test"]:
                    fewshot_test_perts[key].add(pert)
        
        # Assign indices based on splits
        for idx in range(len(self.dataset)):
            original_idx = self.dataset.perturbed_indices[idx]
            cell_type = self.dataset.cell_types[original_idx]
            pert = self.dataset.perturbations[original_idx]
            dataset_name = self.dataset.dataset_labels[original_idx]
            
            # Check zero-shot
            if cell_type in zeroshot_test_cell_types:
                test_indices.append(idx)
                continue
            elif cell_type in zeroshot_val_cell_types:
                val_indices.append(idx)
                continue
            
            # Check few-shot
            is_fewshot = False
            for key in fewshot_val_perts:
                if pert in fewshot_val_perts[key]:
                    val_indices.append(idx)
                    is_fewshot = True
                    break
            
            if not is_fewshot:
                for key in fewshot_test_perts:
                    if pert in fewshot_test_perts[key]:
                        test_indices.append(idx)
                        is_fewshot = True
                        break
            
            # Default to training
            if not is_fewshot:
                train_indices.append(idx)
        
        # If no advanced splits were applied, fall back to random
        if not train_indices and not val_indices and not test_indices:
            logger.warning("No advanced splits applied, falling back to random split")
            self._create_simple_splits()
            return
        
        # Ensure we have all splits
        if not val_indices:
            # Take some from training for validation
            n_val = int(len(train_indices) * 0.1)
            val_indices = train_indices[:n_val]
            train_indices = train_indices[n_val:]
        
        if not test_indices:
            # Take some from training for testing
            n_test = int(len(train_indices) * 0.1)
            test_indices = train_indices[:n_test]
            train_indices = train_indices[n_test:]
        
        # Create subset datasets
        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        self.test_dataset = Subset(self.dataset, test_indices)
        
        logger.info(f"Advanced splits - Train: {len(train_indices)}, "
                   f"Val: {len(val_indices)}, Test: {len(test_indices)}")

    def train_dataloader(self):
        """Create training DataLoader."""
        if self.train_dataset is None:
            raise ValueError("Must call setup() first")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        """Create validation DataLoader."""
        if self.val_dataset is None:
            raise ValueError("Must call setup() first")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        """Create test DataLoader."""
        if self.test_dataset is None:
            raise ValueError("Must call setup() first")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_var_dims(self):
        """Get variable dimensions for model initialization."""
        if self.dataset is None:
            raise ValueError("Must call setup() first")

        input_dim = self.dataset.embeddings.shape[1]
        gene_dim = len(self.dataset.gene_names)
        pert_dim = len(self.dataset.pert_names)
        batch_dim = len(self.dataset.batch_names)

        return {
            "input_dim": input_dim,
            "gene_dim": gene_dim,
            "hvg_dim": input_dim,  # Same as input_dim for HVG
            "output_dim": input_dim,
            "pert_dim": pert_dim,
            "gene_names": self.dataset.gene_names,
            "batch_dim": batch_dim,
            "pert_names": self.dataset.pert_names,
            "cell_type_names": self.dataset.cell_type_names,
            "batch_names": self.dataset.batch_names,
        }


# Preprocessing and Quality Control Utilities

def filter_on_target_knockdown(
    adata: ad.AnnData,
    perturbation_column: str = "gene",
    control_label: str = "non-targeting",
    residual_expression: float = 0.30,
    cell_residual_expression: float = 0.50,
    min_cells: int = 30,
    layer: Optional[str] = None,
    var_gene_name: str = "gene_name"
) -> ad.AnnData:
    """
    Filter perturbation data based on on-target knockdown effectiveness.
    
    Args:
        adata: AnnData object
        perturbation_column: Column in obs containing perturbation info
        control_label: Label for control cells
        residual_expression: Perturbation-level threshold
        cell_residual_expression: Cell-level threshold
        min_cells: Minimum cells per perturbation
        layer: Layer to use (None = adata.X)
        var_gene_name: Column in var with gene names
    
    Returns:
        Filtered AnnData object
    """
    expression = adata.layers[layer] if layer else adata.X
    
    if hasattr(expression, "toarray"):
        expression = expression.toarray()
    
    perts = adata.obs[perturbation_column].values
    unique_perts = np.unique(perts)
    
    control_mask = perts == control_label
    control_expression = expression[control_mask]
    
    if control_expression.shape[0] == 0:
        logger.warning(f"No control cells with label '{control_label}'")
        return adata
    
    control_means = np.array(control_expression.mean(axis=0)).flatten()
    
    valid_cells = []
    for pert in unique_perts:
        if pert == control_label:
            pert_mask = perts == pert
            valid_cells.extend(np.where(pert_mask)[0])
            continue
        
        pert_mask = perts == pert
        pert_indices = np.where(pert_mask)[0]
        
        if len(pert_indices) == 0:
            continue
        
        # Find gene index for this perturbation
        if var_gene_name in adata.var.columns:
            gene_idx = np.where(adata.var[var_gene_name] == pert)[0]
            if len(gene_idx) > 0:
                gene_idx = gene_idx[0]
                pert_expression = expression[pert_indices, gene_idx]
                control_mean = control_means[gene_idx]
                
                # Filter cells by knockdown
                knockdown_ratio = pert_expression / (control_mean + 1e-8)
                cell_mask = knockdown_ratio <= cell_residual_expression
                valid_pert_indices = pert_indices[cell_mask]
                
                if len(valid_pert_indices) >= min_cells:
                    valid_cells.extend(valid_pert_indices)
                else:
                    logger.info(f"Removing '{pert}' - {len(valid_pert_indices)} cells")
            else:
                # Gene not found, keep all cells
                valid_cells.extend(pert_indices)
        else:
            # No gene name column, keep all
            valid_cells.extend(pert_indices)
    
    valid_cells = np.array(valid_cells)
    filtered_adata = adata[valid_cells].copy()
    
    logger.info(f"Filtered {len(adata)} → {len(filtered_adata)} cells")
    
    return filtered_adata


# Example usage
if __name__ == "__main__":
    # Example 1: Simple usage without TOML
    dm = AdvancedPerturbationDataModule(
        data_path="/home/dennis/Parse_10M_PBMC_cytokines.h5ad",
        embed_key="X_hvg",
        pert_col="cytokine",
        cell_type_key="cell_type",
        batch_col="donor",
        control_pert="PBS",
        basal_mapping_strategy="batch",
        batch_size=128,
        num_workers=4,
        hvg_indices_file="hvg_indices.npy"
    )
    dm.setup()
    
    # Example 2: TOML-based configuration
    # dm = AdvancedPerturbationDataModule(
    #     toml_config_path="config.toml",
    #     embed_key="X_hvg",
    #     batch_size=128
    # )
    # dm.setup()

