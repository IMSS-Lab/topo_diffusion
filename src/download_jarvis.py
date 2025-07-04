"""
Comprehensive JARVIS data downloader and processor.

This script handles downloading and processing data from the JARVIS databases,
including DFT, topological, and superconductor datasets.
"""

import os
import json
import gzip
import shutil
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("jarvis_downloader")

class JARVISDownloader:
    """
    Class for downloading and processing data from JARVIS databases.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the downloader with the target directory.
        
        Args:
            data_dir: Base directory to store downloaded and processed data
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create directories
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # JARVIS dataset URLs
        self.dataset_urls = {
            "dft_3d": "https://jarvis.nist.gov/static/jarvis_dft_3d.json.gz",
            "supercon_3d": "https://jarvis.nist.gov/static/jarvis_supercon_3d.json.gz"
        }
    
    def download_dataset(self, dataset_name: str) -> str:
        """
        Download a dataset from JARVIS.
        
        Args:
            dataset_name: Name of the dataset to download
            
        Returns:
            Path to the downloaded dataset
        """
        if dataset_name not in self.dataset_urls:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        url = self.dataset_urls[dataset_name]
        gz_path = os.path.join(self.raw_dir, f"{dataset_name}.json.gz")
        json_path = os.path.join(self.raw_dir, f"{dataset_name}.json")
        
        logger.info(f"Downloading {dataset_name} from {url}")
        
        try:
            # Download the gzipped file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(gz_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {dataset_name}") as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Extract the gzipped file
            logger.info(f"Extracting {gz_path}")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(json_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            logger.info(f"Downloaded and extracted {dataset_name} to {json_path}")
            return json_path
            
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {e}")
            
            # Try using jarvis-tools as fallback
            try:
                logger.info(f"Attempting to download {dataset_name} using JARVIS-Tools API")
                from jarvis.db.figshare import data as jarvis_data
                
                # Download data
                data = jarvis_data(dataset_name)
                
                # Save to file
                with open(json_path, 'w') as f:
                    json.dump(data, f)
                
                logger.info(f"Downloaded {dataset_name} to {json_path} using JARVIS-Tools API")
                return json_path
                
            except Exception as e2:
                logger.error(f"Error downloading {dataset_name} using JARVIS-Tools API: {e2}")
                raise RuntimeError(f"Failed to download {dataset_name} using both methods")
    
    def download_all_datasets(self) -> Dict[str, str]:
        """
        Download all datasets from JARVIS.
        
        Returns:
            Dictionary mapping dataset names to file paths
        """
        dataset_paths = {}
        
        for dataset_name in self.dataset_urls:
            try:
                path = self.download_dataset(dataset_name)
                dataset_paths[dataset_name] = path
            except Exception as e:
                logger.error(f"Error downloading {dataset_name}: {e}")
        
        return dataset_paths
    
    def extract_topological_data(self, dft_path: str) -> str:
        """
        Extract topological data from the DFT dataset.
        
        Args:
            dft_path: Path to DFT data file
            
        Returns:
            Path to extracted topological data file
        """
        logger.info(f"Extracting topological data from {dft_path}")
        
        # Create output path
        topo_path = os.path.join(self.processed_dir, "topological_materials.json")
        
        try:
            # Load data using pandas instead of polars for better compatibility
            # Read in chunks to handle large files
            chunk_size = 1000
            topo_data = []
            
            # Process the file in chunks
            with open(dft_path, 'r') as f:
                # Load the entire JSON array
                data = json.load(f)
                
                # Process in chunks to avoid memory issues
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i+chunk_size]
                    
                    # Extract materials with topological properties
                    for entry in chunk:
                        # Check for topological indicators
                        is_topological = entry.get("is_topological", False)
                        
                        # Safely convert spillage to float
                        spillage = None
                        try:
                            spillage_value = entry.get("spillage", None)
                            if spillage_value is not None:
                                spillage = float(spillage_value)
                        except (ValueError, TypeError):
                            # If conversion fails, set to None
                            spillage = None
                        
                        z2_invariant = entry.get("z2_invariant", [0, 0, 0, 0])
                        
                        # Determine if material is topological
                        is_topo = False
                        if isinstance(is_topological, bool) and is_topological:
                            is_topo = True
                        elif spillage is not None and spillage > 0.5:
                            is_topo = True
                        elif any(z != 0 for z in z2_invariant):
                            is_topo = True
                        
                        if is_topo:
                            # Extract basic properties
                            topo_entry = {
                                "jid": entry.get("jid", ""),
                                "formula": entry.get("formula", ""),
                                "spillage": spillage,
                                "z2_invariant": z2_invariant,
                                "is_topological": True,
                                "band_gap": entry.get("band_gap", None)
                            }
                            
                            # Add to topological dataset
                            topo_data.append(topo_entry)
            
            # Save to file
            with open(topo_path, 'w') as f:
                json.dump(topo_data, f)
            
            logger.info(f"Extracted {len(topo_data)} topological materials to {topo_path}")
            
        except Exception as e:
            logger.error(f"Error extracting topological data: {e}")
            raise RuntimeError(f"Failed to extract topological data from {dft_path}")
        
        return topo_path
    
    def process_dft_data(self, dft_path: str) -> str:
        """
        Process DFT data for use in the diffusion model.
        
        Args:
            dft_path: Path to DFT data file
            
        Returns:
            Path to processed data file
        """
        logger.info(f"Processing DFT data from {dft_path}")
        
        # Create output path
        processed_path = os.path.join(self.processed_dir, "processed_dft.json")
        
        try:
            # Load data using pandas instead of polars for better compatibility
            # Process in chunks to handle large files
            chunk_size = 1000
            processed_data = []
            
            # Read the file
            with open(dft_path, 'r') as f:
                # Load the entire JSON array
                data = json.load(f)
                
                # Process in chunks to avoid memory issues
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i+chunk_size]
                    
                    # Process each entry
                    for entry in chunk:
                        # Check for required properties
                        if not all(entry.get(prop) is not None for prop in ["jid", "formula", "atoms"]):
                            continue
                        
                        # Safely convert spillage to float
                        try:
                            spillage_value = entry.get("spillage", None)
                            if spillage_value is not None:
                                entry["spillage"] = float(spillage_value)
                        except (ValueError, TypeError):
                            # If conversion fails, set to None
                            entry["spillage"] = None
                        
                        # Add topological classification if not present
                        if "is_topological" not in entry:
                            # Use spillage as indicator of topological character
                            spillage = entry.get("spillage", None)
                            if spillage is not None and isinstance(spillage, (int, float)) and spillage > 0.5:
                                entry["is_topological"] = True
                            else:
                                # Default to False if no spillage data or not high enough
                                entry["is_topological"] = False
                        
                        # Add Z2 invariant if not present
                        if "z2_invariant" not in entry:
                            entry["z2_invariant"] = [0, 0, 0, 0]
                        
                        # Calculate sustainability score
                        entry["sustainability_score"] = self._calculate_sustainability_score(entry["formula"])
                        
                        # Add to processed data
                        processed_data.append(entry)
            
            # Save to file
            with open(processed_path, 'w') as f:
                json.dump(processed_data, f)
            
            logger.info(f"Processed DFT data saved to {processed_path} with {len(processed_data)} entries")
            
        except Exception as e:
            logger.error(f"Error processing DFT data: {e}")
            raise RuntimeError(f"Failed to process DFT data from {dft_path}")
        
        return processed_path
    
    def merge_datasets(self, dataset_paths: Dict[str, str]) -> str:
        """
        Merge multiple datasets into a unified dataset.
        
        Args:
            dataset_paths: Dictionary mapping dataset names to file paths
            
        Returns:
            Path to merged dataset file
        """
        logger.info(f"Merging datasets: {list(dataset_paths.keys())}")
        
        # Create output path
        merged_path = os.path.join(self.processed_dir, "unified_dataset.json")
        
        try:
            # Load DFT data as base
            if "dft_3d" not in dataset_paths:
                raise ValueError("DFT dataset is required for merging")
            
            # Load DFT data
            with open(dataset_paths["dft_3d"], 'r') as f:
                dft_data = json.load(f)
            
            # Create JID index for efficient lookups
            dft_jids = {entry["jid"]: i for i, entry in enumerate(dft_data) if "jid" in entry}
            
            # Process and merge each additional dataset
            for name, path in dataset_paths.items():
                if name == "dft_3d":
                    continue
                
                logger.info(f"Merging {name} dataset")
                
                # Load dataset
                with open(path, 'r') as f:
                    additional_data = json.load(f)
                
                # Merge data
                for entry in additional_data:
                    if "jid" in entry and entry["jid"] in dft_jids:
                        # Get index of matching DFT entry
                        idx = dft_jids[entry["jid"]]
                        
                        # Add properties from additional dataset
                        for key, value in entry.items():
                            if key != "jid" and key not in dft_data[idx]:
                                dft_data[idx][key] = value
            
            # Calculate sustainability score for all entries
            for entry in dft_data:
                if "formula" in entry and "sustainability_score" not in entry:
                    entry["sustainability_score"] = self._calculate_sustainability_score(entry["formula"])
            
            # Save to file
            with open(merged_path, 'w') as f:
                json.dump(dft_data, f)
            
            logger.info(f"Merged dataset saved to {merged_path} with {len(dft_data)} entries")
            
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            raise RuntimeError(f"Failed to merge datasets")
        
        return merged_path
    
    def create_train_val_split(self, input_path: str, train_ratio: float = 0.8) -> Tuple[str, str]:
        """
        Create train/validation split of a dataset.
        
        Args:
            input_path: Path to the input JSON file
            train_ratio: Ratio of data to use for training
            
        Returns:
            Tuple of (train_path, val_path)
        """
        logger.info(f"Creating train/val split for {input_path} with ratio {train_ratio}")
        
        # Create output paths
        train_path = os.path.join(self.processed_dir, "train_dataset.json")
        val_path = os.path.join(self.processed_dir, "val_dataset.json")
        
        try:
            # Load data
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Shuffle the data
            import random
            random.seed(42)
            random.shuffle(data)
            
            # Calculate split index
            split_idx = int(len(data) * train_ratio)
            
            # Split the data
            train_data = data[:split_idx]
            val_data = data[split_idx:]
            
            # Save to files
            with open(train_path, 'w') as f:
                json.dump(train_data, f)
            
            with open(val_path, 'w') as f:
                json.dump(val_data, f)
            
            logger.info(f"Created train dataset with {len(train_data)} entries at {train_path}")
            logger.info(f"Created validation dataset with {len(val_data)} entries at {val_path}")
            
        except Exception as e:
            logger.error(f"Error creating train/val split: {e}")
            raise RuntimeError(f"Failed to create train/val split for {input_path}")
        
        return train_path, val_path
    
    def _calculate_sustainability_score(self, formula: str) -> float:
        """
        Calculate sustainability score for a material formula.
        
        Args:
            formula: Chemical formula of the material
            
        Returns:
            Sustainability score between 0 and 1
        """
        # Element properties for sustainability calculation
        element_abundance = {
            "H": 1400, "He": 0.008, "Li": 20, "Be": 2.8, "B": 10, "C": 200, "N": 20, "O": 461000,
            "F": 585, "Ne": 0.005, "Na": 23600, "Mg": 23300, "Al": 82300, "Si": 282000, "P": 1050,
            "S": 350, "Cl": 145, "Ar": 3.5, "K": 20900, "Ca": 41500, "Sc": 22, "Ti": 5650,
            "V": 120, "Cr": 102, "Mn": 950, "Fe": 56300, "Co": 25, "Ni": 84, "Cu": 60, "Zn": 70,
            "Ga": 19, "Ge": 1.5, "As": 1.8, "Se": 0.05, "Br": 2.4, "Kr": 0.0001, "Rb": 90,
            "Sr": 370, "Y": 33, "Zr": 165, "Nb": 20, "Mo": 1.2, "Tc": 0, "Ru": 0.001, "Rh": 0.001,
            "Pd": 0.015, "Ag": 0.075, "Cd": 0.15, "In": 0.25, "Sn": 2.3, "Sb": 0.2, "Te": 0.001,
            "I": 0.45, "Xe": 0.00003, "Cs": 3, "Ba": 425, "La": 39, "Ce": 66.5, "Pr": 9.2,
            "Nd": 41.5, "Pm": 0, "Sm": 7.05, "Eu": 2, "Gd": 6.2, "Tb": 1.2, "Dy": 5.2, "Ho": 1.3,
            "Er": 3.5, "Tm": 0.52, "Yb": 3.2, "Lu": 0.8, "Hf": 3, "Ta": 2, "W": 1.25, "Re": 0.0007,
            "Os": 0.0015, "Ir": 0.001, "Pt": 0.005, "Au": 0.004, "Hg": 0.085, "Tl": 0.85, "Pb": 14,
            "Bi": 0.009, "Po": 0, "At": 0, "Rn": 0, "Fr": 0, "Ra": 0, "Ac": 0, "Th": 9.6,
            "Pa": 0, "U": 2.7, "Np": 0, "Pu": 0
        }
        
        element_toxicity = {
            "H": 1, "He": 0, "Li": 3, "Be": 7, "B": 2, "C": 1, "N": 1, "O": 1,
            "F": 4, "Ne": 0, "Na": 2, "Mg": 1, "Al": 2, "Si": 1, "P": 3, "S": 2,
            "Cl": 3, "Ar": 0, "K": 2, "Ca": 1, "Sc": 2, "Ti": 1, "V": 4, "Cr": 6,
            "Mn": 3, "Fe": 2, "Co": 5, "Ni": 4, "Cu": 3, "Zn": 3, "Ga": 3, "Ge": 2,
            "As": 8, "Se": 5, "Br": 4, "Kr": 0, "Rb": 3, "Sr": 2, "Y": 2, "Zr": 3,
            "Nb": 2, "Mo": 3, "Tc": 5, "Ru": 4, "Rh": 4, "Pd": 3, "Ag": 3, "Cd": 7,
            "In": 4, "Sn": 3, "Sb": 5, "Te": 4, "I": 3, "Xe": 0, "Cs": 3, "Ba": 4,
            "La": 3, "Ce": 3, "Pr": 3, "Nd": 3, "Pm": 4, "Sm": 3, "Eu": 3, "Gd": 3,
            "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3, "Hf": 3,
            "Ta": 3, "W": 4, "Re": 4, "Os": 5, "Ir": 4, "Pt": 3, "Au": 2, "Hg": 9,
            "Tl": 8, "Pb": 8, "Bi": 5, "Po": 10, "At": 10, "Rn": 7, "Fr": 7, "Ra": 9,
            "Ac": 7, "Th": 7, "Pa": 8, "U": 8, "Np": 8, "Pu": 9
        }
        
        # Parse formula to get elements
        import re
        elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
        
        if not elements:
            return 0.5  # Default score
        
        # Calculate abundance score (higher is better)
        abundance_scores = []
        toxicity_scores = []
        
        for element, count in elements:
            # Convert count to integer (default to 1 if empty)
            count = int(count) if count else 1
            
            # Get abundance and calculate score
            abundance = element_abundance.get(element, 0)
            # Log scale transformation to handle wide range of values
            if abundance > 0:
                abundance_score = min(1.0, max(0.0, (np.log10(abundance) + 3) / 8))
            else:
                abundance_score = 0.0
            
            # Get toxicity and calculate score (higher is better = less toxic)
            toxicity = element_toxicity.get(element, 5)
            toxicity_score = 1.0 - (toxicity / 10.0)
            
            # Add to lists with weight by count
            abundance_scores.extend([abundance_score] * count)
            toxicity_scores.extend([toxicity_score] * count)
        
        # Calculate average scores
        avg_abundance = sum(abundance_scores) / len(abundance_scores)
        avg_toxicity = sum(toxicity_scores) / len(toxicity_scores)
        
        # Overall sustainability score (weighted average)
        overall_score = 0.6 * avg_abundance + 0.4 * avg_toxicity
        
        return overall_score

def main():
    """Main function to download and process JARVIS datasets."""
    # Create downloader
    downloader = JARVISDownloader(data_dir="data")
    
    # Download all datasets
    logger.info("Downloading JARVIS datasets")
    dataset_paths = downloader.download_all_datasets()
    
    # Process DFT data
    if "dft_3d" in dataset_paths:
        logger.info("Processing DFT data")
        processed_dft = downloader.process_dft_data(dataset_paths["dft_3d"])
        
        # Extract topological data
        logger.info("Extracting topological data")
        topo_data = downloader.extract_topological_data(dataset_paths["dft_3d"])
        
        # Merge datasets
        logger.info("Merging datasets")
        merged_data = downloader.merge_datasets(dataset_paths)
        
        # Create train/val split
        logger.info("Creating train/val split")
        train_path, val_path = downloader.create_train_val_split(merged_data)
        
        logger.info("All processing completed successfully")
        logger.info(f"Processed data available at: {downloader.processed_dir}")
    else:
        logger.error("DFT dataset not downloaded successfully")

if __name__ == "__main__":
    main()
