"""
Training script for the Topological Materials Diffusion model.

This script trains the diffusion model on real JARVIS data.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm
import argparse
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("train")

# Check NumPy version and warn if incompatible
numpy_version = np.__version__
if numpy_version.startswith('2.'):
    logger.warning(f"NumPy version {numpy_version} detected. This may cause compatibility issues.")
    logger.warning("Consider downgrading to numpy<2 with: pip install 'numpy<2'")
    user_input = input("Continue anyway? (y/n): ")
    if user_input.lower() != 'y':
        logger.info("Exiting. Please downgrade NumPy and try again.")
        sys.exit(0)

# Import project modules
try:
    from data import CrystalGraphDataset, CrystalGraphConverter, CrystalGraphCollator
    from model import CrystalGraphDiffusionModel
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("This may be due to NumPy version incompatibility or missing dependencies.")
    logger.error("Try: pip install 'numpy<2' torch torch-geometric")
    sys.exit(1)

def train(args):
    """
    Train the diffusion model.
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create graph converter
    graph_converter = CrystalGraphConverter(
        cutoff_radius=args.cutoff_radius,
        max_neighbors=args.max_neighbors
    )
    
    # Get feature dimensions
    node_feature_dim, edge_feature_dim = graph_converter.get_feature_dimensions()
    
    # Create datasets
    logger.info(f"Loading training data from {args.train_data}")
    train_dataset = CrystalGraphDataset(
        data_path=args.train_data,
        graph_converter=graph_converter,
        target_properties=args.target_properties
    )
    
    logger.info(f"Loading validation data from {args.val_data}")
    val_dataset = CrystalGraphDataset(
        data_path=args.val_data,
        graph_converter=graph_converter,
        target_properties=args.target_properties
    )
    
    # Validate datasets are not empty
    if len(train_dataset) == 0:
        logger.error(f"Training dataset is empty. Please check {args.train_data} and ensure data processing completed successfully.")
        logger.error("Try running the data processing pipeline again: python src/download_jarvis.py")
        sys.exit(1)
    
    if len(val_dataset) == 0:
        logger.error(f"Validation dataset is empty. Please check {args.val_data} and ensure data processing completed successfully.")
        logger.error("Try running the data processing pipeline again: python src/download_jarvis.py")
        sys.exit(1)
    
    logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    # Create data loaders
    collator = CrystalGraphCollator(target_properties=args.target_properties)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator
    )
    
    # Create model
    logger.info("Creating model")
    model = CrystalGraphDiffusionModel(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        condition_dim=len(args.target_properties)
    ).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
        #verbose=True
    )
    
    # Create loss function
    mse_loss = nn.MSELoss()
    
    # Training loop
    logger.info("Starting training")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_noise_loss = 0.0
        train_property_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} (Train)"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Extract features
            x = batch['x']
            edge_index = batch['edge_index']
            edge_attr = batch['edge_attr']
            batch_idx = batch['batch']
            
            # Extract target properties
            targets = torch.stack([batch[prop] for prop in args.target_properties], dim=1)
            
            # Sample random timesteps
            t = torch.randint(0, args.diffusion_steps, (x.size(0),), device=device)
            
            # Add noise to node features
            noise = torch.randn_like(x)
            noise_coeff = get_noise_schedule(t, args.diffusion_steps, device)
            signal_coeff = 1 - noise_coeff
            
            # Unsqueeze coefficients for proper broadcasting
            noise_coeff = noise_coeff.unsqueeze(1)  # Shape: [num_nodes, 1]
            signal_coeff = signal_coeff.unsqueeze(1)  # Shape: [num_nodes, 1]
            
            # Apply noise
            noisy_x = signal_coeff * x + noise_coeff * noise
            
            # Forward pass
            pred_noise, pred_properties = model(
                noisy_x, edge_index, edge_attr, t, batch_idx, targets
            )
            
            # Compute losses
            noise_loss = mse_loss(pred_noise, noise)
            property_loss = mse_loss(pred_properties, targets)
            
            # Total loss
            loss = noise_loss + args.property_weight * property_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update parameters
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_noise_loss += noise_loss.item()
            train_property_loss += property_loss.item()
        
        # Compute average losses
        train_loss /= len(train_loader)
        train_noise_loss /= len(train_loader)
        train_property_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_noise_loss = 0.0
        val_property_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} (Val)"):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Extract features
                x = batch['x']
                edge_index = batch['edge_index']
                edge_attr = batch['edge_attr']
                batch_idx = batch['batch']
                
                # Extract target properties
                targets = torch.stack([batch[prop] for prop in args.target_properties], dim=1)
                
                # Sample random timesteps
                t = torch.randint(0, args.diffusion_steps, (x.size(0),), device=device)
                
                # Add noise to node features
                noise = torch.randn_like(x)
                noise_coeff = get_noise_schedule(t, args.diffusion_steps, device)
                signal_coeff = 1 - noise_coeff
                
                # Unsqueeze coefficients for proper broadcasting
                noise_coeff = noise_coeff.unsqueeze(1)  # Shape: [num_nodes, 1]
                signal_coeff = signal_coeff.unsqueeze(1)  # Shape: [num_nodes, 1]
                
                # Apply noise
                noisy_x = signal_coeff * x + noise_coeff * noise
                
                # Forward pass
                pred_noise, pred_properties = model(
                    noisy_x, edge_index, edge_attr, t, batch_idx, targets
                )
                
                # Compute losses
                noise_loss = mse_loss(pred_noise, noise)
                property_loss = mse_loss(pred_properties, targets)
                
                # Total loss
                loss = noise_loss + args.property_weight * property_loss
                
                # Update metrics
                val_loss += loss.item()
                val_noise_loss += noise_loss.item()
                val_property_loss += property_loss.item()
        
        # Compute average losses
        val_loss /= len(val_loader)
        val_noise_loss /= len(val_loader)
        val_property_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}:")
        logger.info(f"  Train Loss: {train_loss:.4f} (Noise: {train_noise_loss:.4f}, Property: {train_property_loss:.4f})")
        logger.info(f"  Val Loss: {val_loss:.4f} (Noise: {val_noise_loss:.4f}, Property: {val_property_loss:.4f})")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
            logger.info(f"  Saved best model to {checkpoint_path}")
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(args.output_dir, "latest_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': vars(args)
        }, checkpoint_path)
    
    logger.info("Training completed")


def get_noise_schedule(t, num_steps, device):
    """
    Get noise schedule for diffusion process.
    
    Args:
        t: Timesteps
        num_steps: Total number of diffusion steps
        device: Device
        
    Returns:
        Noise coefficients
    """
    # Linear schedule
    return t.float() / num_steps


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train diffusion model")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, default="data/processed/train_dataset.json",
                        help="Path to training data")
    parser.add_argument("--val_data", type=str, default="data/processed/val_dataset.json",
                        help="Path to validation data")
    parser.add_argument("--target_properties", type=str, nargs="+",
                        default=["formation_energy_per_atom", "band_gap", "is_topological", "sustainability_score"],
                        help="Target properties to predict")
    
    # Model arguments
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of message passing layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Graph arguments
    parser.add_argument("--cutoff_radius", type=float, default=5.0,
                        help="Cutoff radius for neighbors")
    parser.add_argument("--max_neighbors", type=int, default=12,
                        help="Maximum number of neighbors per atom")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm")
    parser.add_argument("--diffusion_steps", type=int, default=1000,
                        help="Number of diffusion steps")
    parser.add_argument("--property_weight", type=float, default=0.1,
                        help="Weight for property prediction loss")
    
    # Other arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Output directory")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
