"""
Main scripts for the Topological Materials Diffusion project.

This module contains the main scripts for data processing, model training,
material generation, and validation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import yaml
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import JARVISDataDownloader, CrystalGraphConverter, CrystalGraphDataset, CrystalGraphCollator
from model import CrystalGraphDiffusionModel, DiffusionProcess
from training import DiffusionTrainer, MaterialValidator
from utils import (
    setup_logging, 
    load_structure_from_file, 
    save_structure_to_file, 
    load_config,
    calculate_sustainability_metrics,
    visualize_structure,
    analyze_structure,
    create_sustainability_radar_chart
)

logger = logging.getLogger(__name__)

# Initialize logging
setup_logging("DEBUG")

def download_data(args):
    """
    Download and process JARVIS datasets.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting JARVIS data download and processing")
    
    # Create downloader
    downloader = JARVISDataDownloader(data_dir=args.output_dir)
    
    # Download data
    logger.info(f"Downloading JARVIS-DFT data (limit={args.limit})")
    dft_path = downloader.download_dft_data(limit=args.limit)
    
    logger.info(f"Downloading JARVIS-TOPO data (limit={args.limit})")
    topo_path = downloader.download_topo_data(limit=args.limit)
    
    logger.info("Downloading JARVIS-ML models")
    ml_models = downloader.download_ml_models(
        model_types=["formation_energy", "band_gap", "is_topological", "bulk_modulus"]
    )
    
    # Merge datasets
    logger.info("Merging datasets")
    unified_path = downloader.merge_datasets(dft_path, topo_path)
    
    # Process data for graph representation
    logger.info("Processing data for graph representation")
    processed_dir = os.path.join(os.path.dirname(args.output_dir), "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create graph converter
    converter = CrystalGraphConverter(
        cutoff_radius=5.0,
        max_neighbors=12
    )
    
    # Create dataset
    dataset = CrystalGraphDataset(
        data_path=unified_path,
        graph_converter=converter
    )
    
    # Save processed dataset info
    dataset_info = {
        "num_structures": len(dataset),
        "node_feature_dim": converter.get_feature_dimensions()[0],
        "edge_feature_dim": converter.get_feature_dimensions()[1],
        "target_properties": dataset.target_properties,
        "data_path": unified_path,
        "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(processed_dir, "dataset_info.json"), 'w') as f:
        import json
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"Data processing complete. Dataset info saved to {processed_dir}/dataset_info.json")
    logger.info(f"Processed dataset contains {len(dataset)} structures")
    
    # Create a sample batch to verify everything works
    collator = CrystalGraphCollator(target_properties=dataset.target_properties)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collator,
        num_workers=0
    )
    
    # Get a sample batch
    sample_batch = next(iter(dataloader))
    
    # Log sample batch info
    logger.info(f"Sample batch contains {sample_batch['num_graphs']} graphs")
    logger.info(f"Node features shape: {sample_batch['x'].shape}")
    logger.info(f"Edge index shape: {sample_batch['edge_index'].shape}")
    logger.info(f"Edge features shape: {sample_batch['edge_attr'].shape}")
    
    # Create a visualization of the dataset statistics
    logger.info("Creating dataset statistics visualization")
    
    # Extract property statistics
    property_stats = {}
    for prop in dataset.target_properties:
        values = []
        for i in range(min(len(dataset), 100)):  # Sample up to 100 structures
            item = dataset[i]
            if prop in item["targets"]:
                values.append(item["targets"][prop])
        
        if values:
            property_stats[prop] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2]
            }
    
    # Create visualization directory
    vis_dir = os.path.join(processed_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create property distribution plots
    fig, axes = plt.subplots(len(property_stats), 1, figsize=(10, 4 * len(property_stats)))
    
    if len(property_stats) == 1:
        axes = [axes]
    
    for i, (prop, stats) in enumerate(property_stats.items()):
        ax = axes[i]
        values = []
        for j in range(min(len(dataset), 100)):  # Sample up to 100 structures
            item = dataset[j]
            if prop in item["targets"]:
                values.append(item["targets"][prop])
        
        ax.hist(values, bins=20, alpha=0.7)
        ax.set_xlabel(prop)
        ax.set_ylabel("Count")
        ax.set_title(f"{prop} Distribution")
        
        # Add statistics as text
        stats_text = (
            f"Count: {stats['count']}\n"
            f"Min: {stats['min']:.3f}\n"
            f"Max: {stats['max']:.3f}\n"
            f"Mean: {stats['mean']:.3f}\n"
            f"Median: {stats['median']:.3f}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "property_distributions.png"), dpi=300)
    plt.close(fig)
    
    logger.info(f"Dataset statistics visualization saved to {vis_dir}/property_distributions.png")
    logger.info("Data download and processing completed successfully")
    
    return unified_path

def train_model(args):
    """
    Train the diffusion model.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting model training")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load dataset info
    dataset_info_path = os.path.join(args.data_dir, "dataset_info.json")
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r') as f:
            import json
            dataset_info = json.load(f)
        logger.info(f"Loaded dataset info from {dataset_info_path}")
    else:
        # Try to infer from the raw data
        raw_data_dir = os.path.join(os.path.dirname(args.data_dir), "raw")
        unified_path = os.path.join(raw_data_dir, "unified_dataset.json")
        
        if not os.path.exists(unified_path):
            logger.error(f"Dataset info not found at {dataset_info_path} and no raw data found at {unified_path}")
            logger.error("Please run the download_data command first")
            return None
        
        # Create graph converter
        converter = CrystalGraphConverter(
            cutoff_radius=5.0,
            max_neighbors=12
        )
        
        # Create temporary dataset to get info
        temp_dataset = CrystalGraphDataset(
            data_path=unified_path,
            graph_converter=converter
        )
        
        # Create dataset info
        dataset_info = {
            "num_structures": len(temp_dataset),
            "node_feature_dim": converter.get_feature_dimensions()[0],
            "edge_feature_dim": converter.get_feature_dimensions()[1],
            "target_properties": temp_dataset.target_properties,
            "data_path": unified_path,
            "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Created dataset info from raw data at {unified_path}")
    
    # Create graph converter
    converter = CrystalGraphConverter(
        cutoff_radius=5.0,
        max_neighbors=12
    )
    
    # Create dataset
    dataset = CrystalGraphDataset(
        data_path=dataset_info["data_path"],
        graph_converter=converter
    )
    
    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    logger.info(f"Split dataset into {train_size} training and {val_size} validation samples")
    
    # Create data loaders
    collator = CrystalGraphCollator(target_properties=dataset.target_properties)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=config["training"].get("num_workers", 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=config["training"].get("num_workers", 0)
    )
    
    # Create model
    model = CrystalGraphDiffusionModel(
        node_feature_dim=dataset_info["node_feature_dim"],
        edge_feature_dim=dataset_info["edge_feature_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
        condition_dim=config["model"]["condition_dim"]
    )
    
    # Create diffusion process
    diffusion = DiffusionProcess(
        num_timesteps=config["diffusion"]["num_timesteps"],
        beta_schedule=config["diffusion"]["beta_schedule"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"]
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=config["training"].get("log_interval", 10),
        use_wandb=args.use_wandb,
        loss_weights=config["loss_weights"]
    )
    
    # Train model
    logger.info(f"Starting training for {config['training']['num_epochs']} epochs")
    
    final_metrics = trainer.train(
        num_epochs=config["training"]["num_epochs"],
        early_stopping_patience=config["training"].get("early_stopping_patience", 10)
    )
    
    # Log final metrics
    logger.info("Training completed")
    logger.info(f"Final metrics: {final_metrics}")
    
    # Plot training curves
    plot_path = trainer.plot_training_curves()
    logger.info(f"Training curves saved to {plot_path}")
    
    # Generate a few samples as a test
    logger.info("Generating test samples")
    
    test_samples = trainer.generate_samples(
        num_samples=5,
        batch_size=5,
        num_nodes=16,
        output_dir=os.path.join(args.checkpoint_dir, "test_samples")
    )
    
    logger.info(f"Generated {len(test_samples)} test samples")
    
    # Return path to best model
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    logger.info(f"Best model saved to {best_model_path}")
    
    return best_model_path

def generate_materials(args):
    """
    Generate new materials using the trained model.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting material generation")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model checkpoint
    logger.info(f"Loading model from {args.model_checkpoint}")
    
    checkpoint = torch.load(args.model_checkpoint, map_location=args.device)
    
    # Extract model configuration from checkpoint
    model_state_dict = checkpoint["model_state_dict"]
    
    # Try to infer model parameters from state dict
    # This is a heuristic approach and might need adjustment
    node_feature_dim = None
    edge_feature_dim = None
    hidden_dim = None
    num_layers = 0
    num_heads = None
    condition_dim = None

    for key, value in model_state_dict.items():
        if key == "node_embedding.0.weight":
            node_feature_dim = value.shape[1]
            hidden_dim = value.shape[0]  # [hidden_dim, node_feature_dim]
        elif key == "edge_embedding.0.weight":
            edge_feature_dim = value.shape[1]
            if hidden_dim is None:
                hidden_dim = value.shape[0]  # [hidden_dim, edge_feature_dim]
        elif "message_passing_layers" in key and ".weight" in key:
            layer_idx = int(key.split(".")[1])
            num_layers = max(num_layers, layer_idx + 1)
        elif key.startswith("attention_layers.") and key.endswith(".edge_proj.weight"):
            num_heads = value.shape[0]  # [num_heads, hidden_dim]
        elif key == "condition_embedding.0.weight":
            condition_dim = value.shape[1]  # [hidden_dim, condition_dim]
    
    if None in (node_feature_dim, edge_feature_dim, hidden_dim, num_heads, condition_dim):
        logger.error("Could not infer all model parameters from checkpoint")
        logger.error("Please provide a configuration file with model parameters")
        return None
    
    # Debug: Print inferred parameters
    logger.info(f"Inferred model parameters: node_feature_dim={node_feature_dim}, "
               f"edge_feature_dim={edge_feature_dim}, hidden_dim={hidden_dim}, "
               f"num_layers={num_layers}, num_heads={num_heads}, condition_dim={condition_dim}")
    
    # Create model
    model = CrystalGraphDiffusionModel(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
        condition_dim=condition_dim
    )
    
    # Load model weights
    model.load_state_dict(model_state_dict, strict=False)
    model.to(args.device)
    model.eval()
    
    # Create diffusion process
    # Use default parameters if not available in checkpoint
    diffusion_params = checkpoint.get("diffusion_params", {
        "num_timesteps": 1000,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02
    })
    
    diffusion = DiffusionProcess(
        num_timesteps=diffusion_params.get("num_timesteps", 1000),
        beta_schedule=diffusion_params.get("beta_schedule", "linear"),
        beta_start=diffusion_params.get("beta_start", 0.0001),
        beta_end=diffusion_params.get("beta_end", 0.02)
    )
    
    # Create trainer for generation
    trainer = DiffusionTrainer(
        model=model,
        diffusion=diffusion,
        train_loader=None,
        device=args.device,
        checkpoint_dir=os.path.dirname(args.model_checkpoint)
    )
    
    # Generate materials
    logger.info(f"Generating {args.num_samples} materials")
    
    # Set conditioning if provided
    condition = None
    if args.condition:
        try:
            condition_dict = {}
            for cond_str in args.condition.split(","):
                key, value = cond_str.split("=")
                condition_dict[key.strip()] = float(value.strip())
            
            logger.info(f"Using conditioning: {condition_dict}")
            
            # Convert to tensor
            condition = {
                k: torch.tensor([v], device=args.device)
                for k, v in condition_dict.items()
            }
        except Exception as e:
            logger.error(f"Error parsing condition string: {e}")
            logger.error("Condition should be in the format 'key1=value1,key2=value2'")
            logger.error("Proceeding without conditioning")
    
    # Generate samples
    generated_structures = trainer.generate_samples(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        condition=condition,
        output_dir=args.output_dir
    )
    
    logger.info(f"Generated {len(generated_structures)} materials")
    
    # Create visualizations
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, structure in enumerate(generated_structures[:min(10, len(generated_structures))]):
        # Visualize structure
        vis_path = visualize_structure(
            structure,
            output_path=os.path.join(vis_dir, f"structure_{i:03d}.png")
        )
        logger.info(f"Created visualization for structure {i}: {vis_path}")
    
    # Create sustainability radar chart
    if len(generated_structures) > 1:
        radar_path = create_sustainability_radar_chart(
            generated_structures,
            output_path=os.path.join(vis_dir, "sustainability_radar.png")
        )
        logger.info(f"Created sustainability radar chart: {radar_path}")
    
    logger.info("Material generation completed successfully")
    
    return args.output_dir

def validate_materials(args):
    """
    Validate generated materials.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting material validation")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load structures
    logger.info(f"Loading structures from {args.input_dir}")
    
    structures = []
    structure_files = []
    
    # Find structure files
    for ext in [".cif", ".poscar", ".vasp", ".json", ".xsf"]:
        structure_files.extend(list(Path(args.input_dir).glob(f"*{ext}")))
    
    logger.info(f"Found {len(structure_files)} structure files")
    
    # Load structures
    for file_path in tqdm(structure_files, desc="Loading structures"):
        try:
            structure = load_structure_from_file(str(file_path))
            structures.append(structure)
        except Exception as e:
            logger.error(f"Error loading structure from {file_path}: {e}")
    
    logger.info(f"Loaded {len(structures)} structures")
    
    # Create validator
    validator = MaterialValidator(
        output_dir=args.output_dir
    )
    
    # Validate structures
    logger.info(f"Validating {len(structures)} structures")
    
    validation_results = validator.batch_validate(
        structures,
        run_dft=args.run_dft
    )
    
    logger.info(f"Validated {len(validation_results)} structures")
    
    # Rank structures
    ranked_results = validator.rank_structures(validation_results)
    
    # Save validation results
    results_path = validator.save_validation_results(
        ranked_results,
        output_path=os.path.join(args.output_dir, "validation_results.json")
    )
    
    logger.info(f"Saved validation results to {results_path}")
    
    # Generate validation report
    report_path = validator.generate_validation_report(
        ranked_results,
        output_path=os.path.join(args.output_dir, "validation_report.html")
    )
    
    logger.info(f"Generated validation report at {report_path}")
    
    # Create visualizations for top structures
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get top 5 valid structures
    top_structures = []
    for result in ranked_results:
        if result.get("is_valid", False) and len(top_structures) < 5:
            if "structure" in result:
                from pymatgen.core.structure import Structure
                structure = Structure.from_dict(result["structure"])
                top_structures.append(structure)
    
    # Visualize top structures
    for i, structure in enumerate(top_structures):
        # Visualize structure
        vis_path = visualize_structure(
            structure,
            output_path=os.path.join(vis_dir, f"top_structure_{i+1}.png")
        )
        logger.info(f"Created visualization for top structure {i+1}: {vis_path}")
    
    # Create sustainability radar chart for top structures
    if len(top_structures) > 1:
        radar_path = create_sustainability_radar_chart(
            top_structures,
            output_path=os.path.join(vis_dir, "top_sustainability_radar.png")
        )
        logger.info(f"Created sustainability radar chart for top structures: {radar_path}")
    
    logger.info("Material validation completed successfully")
    
    return os.path.join(args.output_dir, "validation_report.html")

def parse_download_args(subparsers):
    """Add download data command parser"""
    parser = subparsers.add_parser("download", help="Download and process JARVIS datasets")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                        help="Directory to store downloaded data")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of structures to download")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.set_defaults(func=download_data)

def parse_train_args(subparsers):
    """Add train model command parser"""
    parser = subparsers.add_parser("train", help="Train the diffusion model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing processed data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on (cuda or cpu)")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.set_defaults(func=train_model)

def parse_generate_args(subparsers):
    """Add generate materials command parser"""
    parser = subparsers.add_parser("generate", help="Generate new materials")
    parser.add_argument("--model-checkpoint", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of materials to generate")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument("--num-nodes", type=int, default=16,
                        help="Number of nodes in each generated graph")
    parser.add_argument("--condition", type=str, default=None,
                        help="Conditioning parameters (format: 'key1=value1,key2=value2')")
    parser.add_argument("--output-dir", type=str, default="generated_materials",
                        help="Directory to save generated materials")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to generate on (cuda or cpu)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.set_defaults(func=generate_materials)

def parse_validate_args(subparsers):
    """Add validate materials command parser"""
    parser = subparsers.add_parser("validate", help="Validate generated materials")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing materials to validate")
    parser.add_argument("--output-dir", type=str, default="validation_results",
                        help="Directory to save validation results")
    parser.add_argument("--run-dft", action="store_true",
                        help="Whether to run DFT validation")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.set_defaults(func=validate_materials)

def main():
    """Main entry point for the command-line interface"""
    parser = argparse.ArgumentParser(description="Topological Materials Diffusion CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Add command parsers
    parse_download_args(subparsers)
    parse_train_args(subparsers)
    parse_generate_args(subparsers)
    parse_validate_args(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

