"""
Training and validation module for the Topological Materials Diffusion project.

This module contains the training pipeline, validation tools, and related utilities.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
import logging
import os
from tqdm import tqdm
import wandb
from pymatgen.core.structure import Structure
import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score

from model import CrystalGraphDiffusionModel, DiffusionProcess
from data import CrystalGraphDataset, CrystalGraphCollator, CrystalGraphConverter
from utils import load_structure_from_file, save_structure_to_file, calculate_sustainability_metrics

from model import CrystalGraphDiffusionModel, DiffusionProcess
from data import CrystalGraphDataset, CrystalGraphCollator, CrystalGraphConverter
from utils import load_structure_from_file, save_structure_to_file, calculate_sustainability_metrics

logger = logging.getLogger(__name__)

class DiffusionTrainer:
    """
    Trainer class for the crystal graph diffusion model.
    
    This class handles the training loop, validation, checkpointing,
    and logging for the diffusion model.
    """
    
    def __init__(
        self,
        model: CrystalGraphDiffusionModel,
        diffusion: DiffusionProcess,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
        use_wandb: bool = False,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The diffusion model to train
            diffusion: The diffusion process
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_interval: Interval for logging training progress
            use_wandb: Whether to use Weights & Biases for logging
            loss_weights: Dictionary of loss component weights
        """
        self.model = model
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.loss_weights = loss_weights or {
            "diffusion": 1.0,
            "topological": 0.5,
            "stability": 0.5,
            "sustainability": 0.5,
            "validity": 0.5
        }
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(device)
        
        # Initialize best validation loss
        self.best_val_loss = float('inf')
        
        # Initialize training metrics
        self.train_metrics_history = []
        self.val_metrics_history = []
        
        # Initialize WandB if requested
        if use_wandb:
            try:
                wandb.init(project="topo_diffusion", config={
                    "model": model.__class__.__name__,
                    "diffusion": {
                        "num_timesteps": diffusion.num_timesteps,
                        "beta_schedule": diffusion.beta_schedule,
                        "beta_start": diffusion.beta_start,
                        "beta_end": diffusion.beta_end
                    },
                    "optimizer": optimizer.__class__.__name__,
                    "loss_weights": self.loss_weights
                })
                wandb.watch(model)
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.use_wandb = False
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {
            "diffusion_loss": 0.0,
            "topological_loss": 0.0,
            "stability_loss": 0.0,
            "sustainability_loss": 0.0,
            "validity_loss": 0.0,
            "total_loss": 0.0
        }
        
        # Training loop
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Get node features, edge indices, and edge features
            x = batch["x"]
            edge_index = batch["edge_index"]
            edge_attr = batch["edge_attr"]
            batch_idx = batch["batch"]
            
            # Sample random timesteps
            t = torch.randint(0, self.diffusion.num_timesteps, (batch["num_graphs"],), 
                             device=self.device).long()
            
            # Sample noise
            noise = torch.randn_like(x)
            
            # Add noise to input according to diffusion schedule
            x_noisy = self.diffusion.q_sample(x, t, noise=noise)
            
            # Forward pass to predict noise
            predicted_noise = self.model(
                x=x_noisy,
                edge_index=edge_index,
                edge_attr=edge_attr,
                t=t,
                batch=batch_idx
            )
            
            # Prepare conditioning targets if available
            topological_props = None
            if "is_topological" in batch and "z2_invariant" in batch:
                # Combine Z2 invariants and topological classification
                z2 = batch["z2_invariant"]
                is_topo = batch["is_topological"].unsqueeze(1)
                topological_props = torch.cat([z2, is_topo], dim=1)
            
            stability_metrics = None
            if "formation_energy_per_atom" in batch and "energy_above_hull" in batch:
                # Combine formation energy and energy above hull
                formation = batch["formation_energy_per_atom"].unsqueeze(1)
                hull = batch.get("energy_above_hull", torch.zeros_like(formation))
                if hull.dim() == 1:
                    hull = hull.unsqueeze(1)
                stability_metrics = torch.cat([formation, hull], dim=1)
            
            sustainability_scores = None
            if "sustainability_score" in batch:
                sustainability_scores = batch["sustainability_score"]
            
            # Compute loss
            loss_dict = self.model.loss_function(
                pred=predicted_noise,
                target=noise,
                topological_props=topological_props,
                stability_metrics=stability_metrics,
                sustainability_scores=sustainability_scores,
                loss_weights=self.loss_weights
            )
            
            # Get total loss
            loss = loss_dict["total_loss"]
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Log metrics
            if batch_idx % self.log_interval == 0:
                logger.info(f"Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] "
                           f"Loss: {loss.item():.6f}")
                
                if self.use_wandb:
                    wandb.log({f"train/{k}": v.item() for k, v in loss_dict.items()})
            
            # Update running metrics
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] += v.item()
        
        # Compute average metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_components = {k: v / len(self.train_loader) for k, v in loss_components.items()}
        
        # Store metrics
        metrics = {"loss": avg_loss, **avg_components}
        self.train_metrics_history.append({"epoch": epoch, **metrics})
        
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model on the validation set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        loss_components = {
            "diffusion_loss": 0.0,
            "topological_loss": 0.0,
            "stability_loss": 0.0,
            "sustainability_loss": 0.0,
            "validity_loss": 0.0,
            "total_loss": 0.0
        }
        
        # Validation loop
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get node features, edge indices, and edge features
                x = batch["x"]
                edge_index = batch["edge_index"]
                edge_attr = batch["edge_attr"]
                batch_idx = batch["batch"]
                
                # Sample random timesteps
                t = torch.randint(0, self.diffusion.num_timesteps, (batch["num_graphs"],), 
                                 device=self.device).long()
                
                # Sample noise
                noise = torch.randn_like(x)
                
                # Add noise to input according to diffusion schedule
                x_noisy = self.diffusion.q_sample(x, t, noise=noise)
                
                # Forward pass to predict noise
                predicted_noise = self.model(
                    x=x_noisy,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    t=t,
                    batch=batch_idx
                )
                
                # Prepare conditioning targets if available
                topological_props = None
                if "is_topological" in batch and "z2_invariant" in batch:
                    # Combine Z2 invariants and topological classification
                    z2 = batch["z2_invariant"]
                    is_topo = batch["is_topological"].unsqueeze(1)
                    topological_props = torch.cat([z2, is_topo], dim=1)
                
                stability_metrics = None
                if "formation_energy_per_atom" in batch and "energy_above_hull" in batch:
                    # Combine formation energy and energy above hull
                    formation = batch["formation_energy_per_atom"].unsqueeze(1)
                    hull = batch.get("energy_above_hull", torch.zeros_like(formation))
                    if hull.dim() == 1:
                        hull = hull.unsqueeze(1)
                    stability_metrics = torch.cat([formation, hull], dim=1)
                
                sustainability_scores = None
                if "sustainability_score" in batch:
                    sustainability_scores = batch["sustainability_score"]
                
                # Compute loss
                loss_dict = self.model.loss_function(
                    pred=predicted_noise,
                    target=noise,
                    topological_props=topological_props,
                    stability_metrics=stability_metrics,
                    sustainability_scores=sustainability_scores,
                    loss_weights=self.loss_weights
                )
                
                # Update running metrics
                total_loss += loss_dict["total_loss"].item()
                for k, v in loss_dict.items():
                    loss_components[k] += v.item()
        
        # Compute average metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_components = {k: v / len(self.val_loader) for k, v in loss_components.items()}
        
        logger.info(f"Validation Epoch: {epoch}, Loss: {avg_loss:.6f}")
        
        if self.use_wandb:
            wandb.log({f"val/{k}": v for k, v in avg_components.items()})
        
        # Store metrics
        metrics = {"val_loss": avg_loss, **{f"val_{k}": v for k, v in avg_components.items()}}
        self.val_metrics_history.append({"epoch": epoch, **metrics})
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> str:
        """
        Save a checkpoint of the model.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to save
            is_best: Whether this is the best model so far
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "train_metrics_history": self.train_metrics_history,
            "val_metrics_history": self.val_metrics_history
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            The epoch number of the loaded checkpoint
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load metrics history if available
        if "train_metrics_history" in checkpoint:
            self.train_metrics_history = checkpoint["train_metrics_history"]
        
        if "val_metrics_history" in checkpoint:
            self.val_metrics_history = checkpoint["val_metrics_history"]
        
        return checkpoint["epoch"]
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10) -> Dict[str, float]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            Dictionary of final metrics
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("val_loss", train_metrics["loss"]))
                else:
                    self.scheduler.step()
            
            # Check for improvement
            current_val_loss = val_metrics.get("val_loss", float("inf"))
            is_best = current_val_loss < best_val_loss
            
            if is_best:
                best_val_loss = current_val_loss
                patience_counter = 0
                logger.info(f"New best validation loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter} epochs")
            
            # Save checkpoint
            self.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best)
            
            # Plot training curves
            if epoch % 5 == 0 or epoch == num_epochs:
                self.plot_training_curves()
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Load best model
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            self.load_checkpoint(best_model_path)
            logger.info("Loaded best model for final evaluation")
        
        # Final metrics
        final_metrics = {}
        if self.train_metrics_history:
            final_metrics.update(self.train_metrics_history[-1])
        if self.val_metrics_history:
            final_metrics.update(self.val_metrics_history[-1])
        
        logger.info("Training completed")
        
        return final_metrics
    
    def plot_training_curves(self) -> str:
        """
        Plot training and validation curves.
        
        Returns:
            Path to the saved plot
        """
        if not self.train_metrics_history:
            return ""
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # Convert metrics history to DataFrame
        train_df = pd.DataFrame(self.train_metrics_history)
        val_df = pd.DataFrame(self.val_metrics_history) if self.val_metrics_history else None
        
        # Plot total loss
        ax = axes[0]
        ax.plot(train_df["epoch"], train_df["total_loss"], label="Train")
        if val_df is not None:
            ax.plot(val_df["epoch"], val_df["val_total_loss"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_title("Total Loss vs. Epoch")
        ax.legend()
        ax.grid(True)
        
        # Plot diffusion loss
        ax = axes[1]
        ax.plot(train_df["epoch"], train_df["diffusion_loss"], label="Train")
        if val_df is not None:
            ax.plot(val_df["epoch"], val_df["val_diffusion_loss"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Diffusion Loss")
        ax.set_title("Diffusion Loss vs. Epoch")
        ax.legend()
        ax.grid(True)
        
        # Plot topological loss
        ax = axes[2]
        ax.plot(train_df["epoch"], train_df["topological_loss"], label="Train")
        if val_df is not None:
            ax.plot(val_df["epoch"], val_df["val_topological_loss"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Topological Loss")
        ax.set_title("Topological Loss vs. Epoch")
        ax.legend()
        ax.grid(True)
        
        # Plot stability loss
        ax = axes[3]
        ax.plot(train_df["epoch"], train_df["stability_loss"], label="Train")
        if val_df is not None:
            ax.plot(val_df["epoch"], val_df["val_stability_loss"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Stability Loss")
        ax.set_title("Stability Loss vs. Epoch")
        ax.legend()
        ax.grid(True)
        
        # Plot sustainability loss
        ax = axes[4]
        ax.plot(train_df["epoch"], train_df["sustainability_loss"], label="Train")
        if val_df is not None:
            ax.plot(val_df["epoch"], val_df["val_sustainability_loss"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Sustainability Loss")
        ax.set_title("Sustainability Loss vs. Epoch")
        ax.legend()
        ax.grid(True)
        
        # Plot validity loss
        ax = axes[5]
        ax.plot(train_df["epoch"], train_df["validity_loss"], label="Train")
        if val_df is not None:
            ax.plot(val_df["epoch"], val_df["val_validity_loss"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validity Loss")
        ax.set_title("Validity Loss vs. Epoch")
        ax.legend()
        ax.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.join(self.checkpoint_dir, "plots"), exist_ok=True)
        plot_path = os.path.join(self.checkpoint_dir, "plots", "training_curves.png")
        plt.savefig(plot_path)
        plt.close(fig)
        
        # Log to WandB if enabled
        if self.use_wandb:
            wandb.log({"training_curves": wandb.Image(plot_path)})
        
        return plot_path
    
    def generate_samples(
        self,
        num_samples: int,
        batch_size: int = 4,
        num_nodes: int = 16,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        output_dir: str = "generated_samples"
    ) -> List[Structure]:
        """
        Generate crystal structure samples using the trained model.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            num_nodes: Number of nodes in each graph
            condition: Optional conditioning parameters
            output_dir: Directory to save generated samples
            
        Returns:
            List of generated crystal structures
        """
        logger.info(f"Generating {num_samples} samples")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Generate samples in batches
        generated_structures = []
        
        for i in range(0, num_samples, batch_size):
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, num_samples - i)
            
            # Generate samples
            with torch.no_grad():
                # Prepare model_kwargs with graph attributes
                model_kwargs = {
                    'edge_index': torch.randint(0, 2, (2, num_nodes * current_batch_size)),
                    'edge_attr': torch.randn(num_nodes * current_batch_size, 128),
                    'batch': torch.arange(current_batch_size, device=self.device)
                }
                
                # Compute shape for diffusion sampling
                shape = (batch_size * num_nodes, self.model.node_feature_dim)
                
                samples = self.diffusion.sample(
                    self.model,
                    shape=shape,
                    device=self.device,
                    edge_feature_dim=self.model.edge_feature_dim,
                    condition=condition
                )
                
                # Convert samples to crystal structures
                for j in range(current_batch_size):
                    # Extract sample
                    sample = samples[j]
                    
                    # Convert to crystal structure (placeholder)
                    # In a real implementation, would convert the graph representation
                    # back to a crystal structure
                    structure = Structure.from_spacegroup(
                        "Fm-3m", 
                        lattice=5.0 * np.eye(3),
                        species=["Si"] * 4,
                        coords=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
                    )
                    
                    # Save structure
                    structure_path = os.path.join(output_dir, f"sample_{i+j:04d}.cif")
                    structure.to(filename=structure_path)
                    
                    # Add to list
                    generated_structures.append(structure)
                    
                    logger.info(f"Generated sample {i+j+1}/{num_samples}")
        
        logger.info(f"Generated {len(generated_structures)} samples")
        
        return generated_structures


class MaterialValidator:
    """
    Class for validating generated crystal structures.
    
    This class handles the validation of generated crystal structures,
    including physical validity checks, property prediction, and ranking.
    """
    
    def __init__(
        self,
        dft_interface=None,
        ml_models=None,
        validity_checks=None,
        output_dir: str = "validation_results"
    ):
        """
        Initialize the validator.
        
        Args:
            dft_interface: Interface to DFT calculations
            ml_models: Dictionary of ML models for property prediction
            validity_checks: List of validity check functions
            output_dir: Directory to save validation results
        """
        self.dft_interface = dft_interface
        self.ml_models = ml_models or {}
        self.validity_checks = validity_checks or []
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize property predictors
        self._init_property_predictors()
    
    def _init_property_predictors(self):
        """Initialize property prediction models."""
        # In a real implementation, would load pre-trained models
        # for property prediction
        
        # For now, just log the initialization
        logger.info("Initializing property predictors")
        
        # Check if ML models are provided
        if not self.ml_models:
            logger.warning("No ML models provided for property prediction")
    
    def check_physical_validity(self, structure: Structure) -> Tuple[bool, Dict]:
        """
        Check if a crystal structure is physically valid.
        
        Args:
            structure: Pymatgen Structure object
            
        Returns:
            Tuple of (is_valid, details_dict)
        """
        logger.info(f"Checking physical validity of structure with {len(structure)} atoms")
        
        # Initialize validity results
        validity_results = {
            "has_short_bonds": False,
            "has_unrealistic_angles": False,
            "has_isolated_atoms": False,
            "has_charge_imbalance": False
        }
        
        # Check for short bonds
        min_distance = structure.distance_matrix.min()
        if min_distance < 0.5:  # Angstroms
            validity_results["has_short_bonds"] = True
            logger.warning(f"Structure has short bonds (min distance: {min_distance:.3f} Å)")
        
        # Check for unrealistic bond angles
        # In a real implementation, would compute bond angles and check for unrealistic values
        
        # Check for isolated atoms
        # In a real implementation, would check for atoms with no neighbors
        
        # Check for charge imbalance
        # In a real implementation, would check for charge balance
        
        # Determine overall validity
        is_valid = all(not v for v in validity_results.values())
        
        return is_valid, validity_results
    
    def predict_properties(self, structure: Structure) -> Dict:
        """
        Predict properties of a crystal structure using ML models.
        
        Args:
            structure: Pymatgen Structure object
            
        Returns:
            Dictionary of predicted properties
        """
        logger.info(f"Predicting properties for structure with {len(structure)} atoms")
        
        # Initialize predicted properties
        predicted_properties = {}
        
        # Use ML models to predict properties if available
        if self.ml_models:
            # In a real implementation, would use the ML models to predict properties
            
            # For now, use placeholder values
            predicted_properties = {
                "formation_energy": -0.5,  # eV/atom
                "band_gap": 0.2,  # eV
                "is_topological": True,
                "z2_invariant": [1, 0, 0, 0],
                "bulk_modulus": 100.0  # GPa
            }
        else:
            # Use simple heuristics if no ML models are available
            
            # Estimate formation energy based on electronegativity differences
            electronegativities = [element.X for element in structure.composition.elements]
            if len(electronegativities) > 1:
                en_diff = max(electronegativities) - min(electronegativities)
                formation_energy = -0.2 - 0.3 * en_diff  # Simple heuristic
            else:
                formation_energy = 0.0
            
            # Estimate band gap based on element properties
            avg_en = sum(electronegativities) / len(electronegativities)
            band_gap = 0.1 * avg_en  # Simple heuristic
            
            # Placeholder for topological properties
            is_topological = False
            z2_invariant = [0, 0, 0, 0]
            
            # Estimate bulk modulus based on bond strengths
            bulk_modulus = 50.0 + 10.0 * avg_en  # Simple heuristic
            
            predicted_properties = {
                "formation_energy": formation_energy,
                "band_gap": band_gap,
                "is_topological": is_topological,
                "z2_invariant": z2_invariant,
                "bulk_modulus": bulk_modulus
            }
        
        return predicted_properties
    
    def calculate_sustainability_score(self, structure: Structure) -> Dict:
        """
        Calculate sustainability metrics for a crystal structure.
        
        Args:
            structure: Pymatgen Structure object
            
        Returns:
            Dictionary of sustainability metrics
        """
        logger.info(f"Calculating sustainability score for structure with {len(structure)} atoms")
        
        # Get element symbols
        elements = [str(element) for element in structure.composition.elements]
        
        # Calculate sustainability metrics
        sustainability_metrics = calculate_sustainability_metrics(elements)
        
        return sustainability_metrics
    
    def run_dft_validation(self, structure: Structure) -> Dict:
        """
        Run DFT calculations to validate the structure and its properties.
        
        Args:
            structure: Pymatgen Structure object
            
        Returns:
            Dictionary of DFT-calculated properties
        """
        logger.info(f"Running DFT validation for structure with {len(structure)} atoms")
        
        # Check if DFT interface is available
        if self.dft_interface is None:
            logger.warning("DFT interface not provided, skipping DFT validation")
            return {}
        
        # In a real implementation, would use the DFT interface to:
        # - Optimize the structure
        # - Calculate formation energy
        # - Calculate band structure
        # - Calculate topological invariants
        
        # For now, use placeholder values
        dft_results = {
            "formation_energy_dft": -0.45,  # eV/atom
            "band_gap_dft": 0.18,  # eV
            "is_topological_dft": True,
            "z2_invariant_dft": [1, 0, 0, 0]
        }
        
        return dft_results
    
    def validate_structure(self, structure: Structure, run_dft: bool = False) -> Dict:
        """
        Perform comprehensive validation of a crystal structure.
        
        Args:
            structure: Pymatgen Structure object
            run_dft: Whether to run DFT validation
            
        Returns:
            Dictionary of validation results
        """
        # Check physical validity
        is_valid, validity_details = self.check_physical_validity(structure)
        
        if not is_valid:
            logger.warning(f"Structure failed physical validity checks: {validity_details}")
            return {"is_valid": False, "validity_details": validity_details}
        
        # Predict properties
        predicted_properties = self.predict_properties(structure)
        
        # Calculate sustainability score
        sustainability_metrics = self.calculate_sustainability_score(structure)
        
        # Run DFT validation if requested
        dft_results = {}
        if run_dft:
            dft_results = self.run_dft_validation(structure)
        
        # Combine all results
        validation_results = {
            "is_valid": is_valid,
            "validity_details": validity_details,
            "predicted_properties": predicted_properties,
            "sustainability_metrics": sustainability_metrics,
            "dft_results": dft_results,
            "structure": structure.as_dict()
        }
        
        return validation_results
    
    def batch_validate(self, structures: List[Structure], run_dft: bool = False) -> List[Dict]:
        """
        Validate a batch of structures.
        
        Args:
            structures: List of Pymatgen Structure objects
            run_dft: Whether to run DFT validation
            
        Returns:
            List of validation result dictionaries
        """
        logger.info(f"Validating {len(structures)} structures")
        
        # Validate each structure
        validation_results = []
        for i, structure in enumerate(tqdm(structures, desc="Validating structures")):
            result = self.validate_structure(structure, run_dft=run_dft)
            validation_results.append(result)
            
            # Log progress
            if (i + 1) % 10 == 0 or i + 1 == len(structures):
                logger.info(f"Validated {i+1}/{len(structures)} structures")
        
        return validation_results
    
    def rank_structures(self, validation_results: List[Dict], criteria: Dict = None) -> List[Dict]:
        """
        Rank structures based on validation results and specified criteria.
        
        Args:
            validation_results: List of validation result dictionaries
            criteria: Dictionary of criteria weights
            
        Returns:
            List of ranked validation results
        """
        logger.info(f"Ranking {len(validation_results)} structures")
        
        # Default criteria weights
        default_criteria = {
            "topological_score": 0.4,
            "stability_score": 0.3,
            "sustainability_score": 0.3
        }
        
        # Use provided criteria or defaults
        criteria = criteria or default_criteria
        
        # Filter out invalid structures
        valid_results = [result for result in validation_results if result.get("is_valid", False)]
        
        if not valid_results:
            logger.warning("No valid structures to rank")
            return validation_results
        
        # Calculate scores for each structure
        for result in valid_results:
            # Initialize scores
            scores = {}
            
            # Topological score
            props = result.get("predicted_properties", {})
            is_topo = props.get("is_topological", False)
            z2_sum = sum(props.get("z2_invariant", [0, 0, 0, 0]))
            
            # Higher score for topological materials with non-trivial Z2 invariants
            scores["topological_score"] = float(is_topo) * (0.5 + 0.5 * min(z2_sum, 1))
            
            # Stability score
            formation_energy = props.get("formation_energy", 0.0)
            energy_above_hull = props.get("energy_above_hull", 1.0)
            
            # Lower formation energy and energy above hull are better
            # Normalize to [0, 1] range
            stability_score = 0.0
            if formation_energy is not None:
                # Normalize formation energy to [0, 1]
                # Assume range of [-5, 5] eV/atom
                norm_formation = (formation_energy + 5) / 10
                norm_formation = max(0, min(1, 1 - norm_formation))
                
                # Normalize energy above hull to [0, 1]
                # Assume range of [0, 1] eV/atom
                norm_hull = max(0, min(1, 1 - energy_above_hull))
                
                # Combine with higher weight on energy above hull
                stability_score = 0.4 * norm_formation + 0.6 * norm_hull
            
            scores["stability_score"] = stability_score
            
            # Sustainability score
            sustainability = result.get("sustainability_metrics", {}).get("overall_score", 0.5)
            scores["sustainability_score"] = sustainability
            
            # Calculate combined score
            combined_score = sum(criteria[k] * scores[k] for k in criteria)
            
            # Add scores to result
            result["scores"] = scores
            result["combined_score"] = combined_score
        
        # Sort by combined score (descending)
        ranked_results = sorted(valid_results, key=lambda x: x.get("combined_score", 0), reverse=True)
        
        # Add invalid structures at the end
        invalid_results = [result for result in validation_results if not result.get("is_valid", False)]
        ranked_results.extend(invalid_results)
        
        return ranked_results
    
    def save_validation_results(self, results: List[Dict], output_path: str = None) -> str:
        """
        Save validation results to a file.
        
        Args:
            results: List of validation result dictionaries
            output_path: Path to save the results
            
        Returns:
            Path to the saved results
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"validation_results_{timestamp}.json")
        
        # Create a serializable version of the results
        serializable_results = []
        for result in results:
            # Create a copy to avoid modifying the original
            serializable_result = result.copy()
            
            # Remove structure dictionary (can be large)
            if "structure" in serializable_result:
                del serializable_result["structure"]
            
            # Ensure all values are serializable
            for k, v in serializable_result.items():
                if isinstance(v, np.ndarray):
                    serializable_result[k] = v.tolist()
                elif isinstance(v, np.integer):
                    serializable_result[k] = int(v)
                elif isinstance(v, np.floating):
                    serializable_result[k] = float(v)
            
            serializable_results.append(serializable_result)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump({"results": serializable_results}, f, indent=2)
        
        logger.info(f"Saved validation results to {output_path}")
        
        return output_path
    
    def generate_validation_report(self, results: List[Dict], output_path: str = None) -> str:
        """
        Generate a validation report.
        
        Args:
            results: List of validation result dictionaries
            output_path: Path to save the report
            
        Returns:
            Path to the saved report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"validation_report_{timestamp}.html")
        
        # Count valid structures
        valid_count = sum(1 for result in results if result.get("is_valid", False))
        
        # Extract property statistics
        property_stats = {}
        for prop in ["formation_energy", "band_gap", "is_topological"]:
            values = [result.get("predicted_properties", {}).get(prop) for result in results 
                     if result.get("is_valid", False)]
            values = [v for v in values if v is not None]
            
            if values:
                if prop == "is_topological":
                    # Count True values
                    true_count = sum(1 for v in values if v)
                    property_stats[prop] = {
                        "count": len(values),
                        "true_count": true_count,
                        "true_percentage": 100 * true_count / len(values)
                    }
                else:
                    # Calculate statistics
                    property_stats[prop] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "median": sorted(values)[len(values) // 2]
                    }
        
        # Extract sustainability statistics
        sustainability_values = [result.get("sustainability_metrics", {}).get("overall_score") 
                               for result in results if result.get("is_valid", False)]
        sustainability_values = [v for v in sustainability_values if v is not None]
        
        if sustainability_values:
            property_stats["sustainability_score"] = {
                "count": len(sustainability_values),
                "min": min(sustainability_values),
                "max": max(sustainability_values),
                "mean": sum(sustainability_values) / len(sustainability_values),
                "median": sorted(sustainability_values)[len(sustainability_values) // 2]
            }
        
        # Generate HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .valid {{ color: green; }}
                .invalid {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Validation Report</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Summary</h2>
            <p>Total structures: {len(results)}</p>
            <p>Valid structures: <span class="valid">{valid_count}</span> ({100 * valid_count / len(results):.1f}%)</p>
            <p>Invalid structures: <span class="invalid">{len(results) - valid_count}</span> ({100 * (len(results) - valid_count) / len(results):.1f}%)</p>
            
            <h2>Property Statistics</h2>
            <table>
                <tr>
                    <th>Property</th>
                    <th>Count</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Mean</th>
                    <th>Median</th>
                </tr>
        """
        
        # Add property statistics to the report
        for prop, stats in property_stats.items():
            if prop == "is_topological":
                html += f"""
                <tr>
                    <td>{prop}</td>
                    <td>{stats["count"]}</td>
                    <td>-</td>
                    <td>-</td>
                    <td>{stats["true_count"]} ({stats["true_percentage"]:.1f}%)</td>
                    <td>-</td>
                </tr>
                """
            else:
                html += f"""
                <tr>
                    <td>{prop}</td>
                    <td>{stats["count"]}</td>
                    <td>{stats["min"]:.3f}</td>
                    <td>{stats["max"]:.3f}</td>
                    <td>{stats["mean"]:.3f}</td>
                    <td>{stats["median"]:.3f}</td>
                </tr>
                """
        
        html += """
            </table>
            
            <h2>Top 10 Structures</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Formula</th>
                    <th>Topological</th>
                    <th>Formation Energy</th>
                    <th>Band Gap</th>
                    <th>Sustainability</th>
                    <th>Combined Score</th>
                </tr>
        """
        
        # Add top 10 structures to the report
        for i, result in enumerate(results[:10]):
            if not result.get("is_valid", False):
                continue
            
            props = result.get("predicted_properties", {})
            scores = result.get("scores", {})
            
            formula = "Unknown"
            if "structure" in result:
                try:
                    from pymatgen.core.composition import Composition
                    formula = Composition.from_dict(result["structure"]["composition"]).reduced_formula
                except:
                    pass
            
            html += f"""
            <tr>
                <td>{i+1}</td>
                <td>{formula}</td>
                <td>{"Yes" if props.get("is_topological", False) else "No"}</td>
                <td>{props.get("formation_energy", "-"):.3f}</td>
                <td>{props.get("band_gap", "-"):.3f}</td>
                <td>{result.get("sustainability_metrics", {}).get("overall_score", "-"):.3f}</td>
                <td>{result.get("combined_score", "-"):.3f}</td>
            </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        # Save the report
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Generated validation report at {output_path}")
        
        return output_path

