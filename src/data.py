"""
Data processing module for the Topological Materials Diffusion project.

This module handles data acquisition, processing, and loading from JARVIS datasets,
as well as crystal graph representation.
"""

import os
import json
import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
import torch
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jarvis_data
# Removed import: from jarvis.core.graphs import Graph
import requests
from tqdm import tqdm
import gzip
import shutil

logger = logging.getLogger(__name__)

class JARVISDataDownloader:
    """
    Class for downloading and processing data from JARVIS databases.
    
    This class handles the acquisition of data from:
    - JARVIS-DFT: Crystal structures and DFT-calculated properties
    - JARVIS-TOPO: Topological classifications integrated within DFT data
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the downloader with the target directory.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_dft_data(self, limit: Optional[int] = None) -> str:
        """
        Download crystal structures and properties from JARVIS-DFT.
        
        Args:
            limit: Optional limit on number of structures to download
            
        Returns:
            Path to downloaded data file
        """
        logger.info(f"Downloading JARVIS-DFT data (limit={limit})")
        
        # Create output path
        dft_path = os.path.join(self.data_dir, "jarvis_dft_3d.json")
        gz_path = os.path.join(self.data_dir, "jarvis_dft_3d.json.gz")
        
        try:
            # Direct download from JARVIS website
            url = "https://jarvis.nist.gov/static/jarvis_dft_3d.json.gz"
            logger.info(f"Downloading from {url}")
            
            # Download the gzipped file
            response = requests.get(url, stream=True)
            with open(gz_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the gzipped file
            logger.info(f"Extracting {gz_path}")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(dft_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Load the data to process and limit if needed
            with open(dft_path, 'r') as f:
                dft_data = json.load(f)
            
            # Limit the number of structures if specified
            if limit is not None:
                dft_data = dft_data[:limit]
                # Save the limited dataset
                with open(dft_path, 'w') as f:
                    json.dump(dft_data, f)
            
            logger.info(f"Downloaded {len(dft_data)} structures to {dft_path}")
            
        except Exception as e:
            logger.error(f"Error downloading DFT data: {e}")
            
            # Try using JARVIS-Tools API as fallback
            try:
                logger.info("Attempting to download using JARVIS-Tools API")
                dft_data = jarvis_data("dft_3d")
                
                # Limit the number of structures if specified
                if limit is not None:
                    dft_data = dft_data[:limit]
                
                # Save to file
                with open(dft_path, 'w') as f:
                    json.dump(dft_data, f)
                
                logger.info(f"Downloaded {len(dft_data)} structures to {dft_path} using JARVIS-Tools API")
            except Exception as e2:
                logger.error(f"Error downloading DFT data using JARVIS-Tools API: {e2}")
                raise RuntimeError("Failed to download JARVIS-DFT data using both methods")
        
        return dft_path
    
    def extract_topo_data(self, dft_path: str) -> str:
        """
        Extract topological data from the DFT dataset.
        
        Args:
            dft_path: Path to DFT data file
            
        Returns:
            Path to extracted topological data file
        """
        logger.info(f"Extracting topological data from {dft_path}")
        
        # Create output path
        topo_path = os.path.join(self.data_dir, "jarvis_topo.json")
        
        try:
            # Load DFT data
            with open(dft_path, 'r') as f:
                dft_data = json.load(f)
            
            # Extract materials with topological properties
            topo_data = []
            for entry in tqdm(dft_data, desc="Extracting topological data"):
                # Check for topological indicators
                spillage = entry.get("spillage", None)
                
                if spillage is not None and spillage > 0.5:  # Threshold for topological character
                    # Extract basic properties
                    topo_entry = {
                        "jid": entry.get("jid", ""),
                        "formula": entry.get("formula", ""),
                        "spillage": spillage,
                        "band_gap": entry.get("band_gap", None)
                    }
                    
                    # Extract Z2 invariants if available
                    if "z2_invariant" in entry:
                        topo_entry["z2_invariant"] = entry["z2_invariant"]
                    else:
                        # Default Z2 invariant (0;000) for non-topological
                        topo_entry["z2_invariant"] = [0, 0, 0, 0]
                    
                    # Determine if material is topological
                    topo_entry["is_topological"] = any(z != 0 for z in topo_entry["z2_invariant"]) or spillage > 1.0
                    
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
    
    def merge_datasets(self, dft_path: str, topo_path: str) -> str:
        """
        Merge DFT and topological data into a unified dataset.
        
        Args:
            dft_path: Path to DFT data file
            topo_path: Path to topological data file
            
        Returns:
            Path to merged dataset file
        """
        logger.info(f"Merging datasets: {dft_path} and {topo_path}")
        
        # Create output path
        merged_path = os.path.join(self.data_dir, "unified_dataset.json")
        
        try:
            # Load datasets
            with open(dft_path, 'r') as f:
                dft_data = json.load(f)
            
            with open(topo_path, 'r') as f:
                topo_data = json.load(f)
            
            # Create mapping from JID to topological properties
            topo_map = {entry["jid"]: entry for entry in topo_data}
            
            # Merge datasets
            merged_data = []
            for dft_entry in tqdm(dft_data, desc="Merging datasets"):
                jid = dft_entry.get("jid", "")
                
                # Get topological properties if available
                topo_entry = topo_map.get(jid, {})
                
                # Merge properties
                merged_entry = dft_entry.copy()
                
                # Add topological properties
                merged_entry["z2_invariant"] = topo_entry.get("z2_invariant", [0, 0, 0, 0])
                merged_entry["is_topological"] = topo_entry.get("is_topological", False)
                merged_entry["spillage"] = topo_entry.get("spillage", None)
                
                # Add to merged dataset
                merged_data.append(merged_entry)
            
            # Save to file
            with open(merged_path, 'w') as f:
                json.dump(merged_data, f)
            
            logger.info(f"Merged dataset saved to {merged_path} with {len(merged_data)} entries")
            
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            raise RuntimeError(f"Failed to merge datasets {dft_path} and {topo_path}")
        
        return merged_path
    
    def _jarvis_atoms_to_pymatgen(self, atoms_dict: Dict) -> Structure:
        """
        Convert JARVIS Atoms to pymatgen Structure.
        
        Args:
            atoms_dict: Dictionary representation of JARVIS Atoms
            
        Returns:
            Pymatgen Structure object
        """
        # Convert JARVIS Atoms to pymatgen Structure
        atoms = Atoms.from_dict(atoms_dict)
        
        lattice = atoms.lattice.matrix
        species = atoms.elements
        coords = atoms.frac_coords
        
        return Structure(lattice, species, coords)


class CrystalGraphDataset(torch.utils.data.Dataset):
    """
    Dataset for crystal structures represented as graphs.
    
    This dataset loads crystal structures from a JSON file and converts them
    to graph representations for use with graph neural networks.
    """
    
    def __init__(
        self,
        data_path: str,
        graph_converter: 'CrystalGraphConverter',
        target_properties: List[str] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the JSON file containing crystal structures
            graph_converter: Converter for crystal structures to graphs
            target_properties: List of property names to include as targets
            transform: Optional transform to apply to the data
        """
        self.data_path = data_path
        self.graph_converter = graph_converter
        self.target_properties = target_properties or ["formation_energy_per_atom", "band_gap", "is_topological"]
        self.transform = transform
        
        # Property name mapping from JARVIS to expected names
        self.property_mapping = {
            "formation_energy_per_atom": ["formation_energy_per_atom", "formation_energy_peratom"],
            "band_gap": ["band_gap", "optb88vdw_bandgap"],
            "is_topological": ["is_topological", "spillage"],  # Use spillage as fallback for is_topological
            "sustainability_score": ["sustainability_score"]
        }
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Process and filter data
        self.filtered_data = []
        for entry in self.data:
            # Convert atoms to structure if needed
            if "structure" not in entry and "atoms" in entry:
                try:
                    # Convert JARVIS atoms to pymatgen structure
                    atoms_dict = entry["atoms"]
                    structure = self._jarvis_atoms_to_pymatgen(atoms_dict)
                    entry["structure"] = structure.as_dict()
                except Exception as e:
                    logger.warning(f"Failed to convert atoms to structure: {e}")
                    continue
            
            # Map property names
            self._map_property_names(entry)
            
            # Special handling for is_topological based on spillage
            if "is_topological" not in entry and "spillage" in entry and entry["spillage"] is not None:
                # If spillage > 0.5, consider it topological (this is a heuristic)
                try:
                    spillage_value = float(entry["spillage"])
                    entry["is_topological"] = spillage_value > 0.5
                except (ValueError, TypeError):
                    # Default to False if spillage can't be converted to float
                    entry["is_topological"] = False
            
            # Check if all required properties are present
            if all(entry.get(prop) is not None for prop in self.target_properties):
                self.filtered_data.append(entry)
        
        logger.info(f"Loaded {len(self.filtered_data)} structures with all required properties")
    
    def _jarvis_atoms_to_pymatgen(self, atoms_dict):
        """
        Convert JARVIS Atoms to pymatgen Structure.
        
        Args:
            atoms_dict: Dictionary representation of JARVIS Atoms
            
        Returns:
            Pymatgen Structure object
        """
        # Convert JARVIS Atoms to pymatgen Structure
        atoms = Atoms.from_dict(atoms_dict)
        
        lattice = atoms.lattice.matrix
        species = atoms.elements
        coords = atoms.frac_coords
        
        return Structure(lattice, species, coords)
        
    def _map_property_names(self, entry):
        """
        Map property names from JARVIS format to expected format.
        
        Args:
            entry: Dictionary containing entry data
        """
        for target_prop, source_props in self.property_mapping.items():
            # Skip if target property already exists
            if entry.get(target_prop) is not None:
                continue
                
            # Try to find value from alternative property names
            for source_prop in source_props:
                if source_prop != target_prop and entry.get(source_prop) is not None:
                    entry[target_prop] = entry[source_prop]
                    break
    
    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return len(self.filtered_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing the graph representation and target properties
        """
        entry = self.filtered_data[idx]
        
        # Load structure
        structure_dict = entry["structure"]
        structure = Structure.from_dict(structure_dict)
        
        # Convert to graph
        graph_data = self.graph_converter.convert_structure(structure)
        
        # Extract target properties
        targets = {}
        for prop in self.target_properties:
            value = entry.get(prop)
            if isinstance(value, bool):
                value = float(value)
            targets[prop] = value
        
        # Create item
        item = {
            "graph": graph_data,
            "targets": targets,
            "jid": entry.get("jid", ""),
            "formula": entry.get("formula", "")
        }
        
        # Apply transform if provided
        if self.transform is not None:
            item = self.transform(item)
        
        return item


class CrystalGraphConverter:
    """
    Class for converting crystal structures to graph representations.
    
    This class handles the conversion of crystal structures to graph representations
    suitable for use with graph neural networks and diffusion models.
    """
    
    def __init__(
        self, 
        cutoff_radius: float = 5.0,
        max_neighbors: int = 12,
        node_features: List[str] = None,
        edge_features: List[str] = None
    ):
        """
        Initialize the converter with parameters for graph construction.
        
        Args:
            cutoff_radius: Maximum distance for considering atoms as neighbors
            max_neighbors: Maximum number of neighbors per atom
            node_features: List of node features to include
            edge_features: List of edge features to include
        """
        self.cutoff_radius = cutoff_radius
        self.max_neighbors = max_neighbors
        self.node_features = node_features or ["atomic_number", "electronegativity", "radius", "row", "group", "block"]
        self.edge_features = edge_features or ["distance", "vector_x"]  # Reduced to only 2 features
        
        # Initialize neighbor finder
        self.neighbor_finder = CrystalNN(
            weighted_cn=False,
            cation_anion=False,
            distance_cutoffs=(0.5, cutoff_radius)
        )
        
        # Element property cache
        self._element_properties = {}
    
    def convert_structure(self, structure: Structure) -> Dict:
        """
        Convert a pymatgen Structure to a graph representation.
        
        Args:
            structure: Pymatgen Structure object
            
        Returns:
            Dictionary containing graph representation with node and edge features
        """
        num_atoms = len(structure)
        
        # Compute node features
        node_features = np.zeros((num_atoms, len(self.node_features)))
        for i, site in enumerate(structure):
            node_features[i] = self._compute_node_features(site.specie)
        
        # Compute edges and edge features
        edge_index = []
        edge_features = []
        
        for i, site in enumerate(structure):
            # Get neighbors within cutoff radius
            neighbors = self.neighbor_finder.get_nn_info(structure, i)
            
            # Limit to max_neighbors
            if len(neighbors) > self.max_neighbors:
                neighbors = neighbors[:self.max_neighbors]
            
            for neighbor in neighbors:
                j = neighbor["site_index"]
                distance = neighbor["weight"]
                
                # Skip self-loops
                if i == j:
                    continue
                
                # Add edge
                edge_index.append([i, j])
                
                # Compute edge features
                edge_feat = self._compute_edge_features(site, structure[j], distance)
                edge_features.append(edge_feat)
        
        # Convert to numpy arrays
        if edge_index:
            edge_index = np.array(edge_index).T  # Shape: [2, num_edges]
            edge_features = np.array(edge_features)  # Shape: [num_edges, edge_feature_dim]
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, len(self.edge_features)), dtype=np.float32)
        
        # Create graph data
        graph_data = {
            "num_nodes": num_atoms,
            "node_features": node_features.astype(np.float32),
            "edge_index": edge_index.astype(np.int64),
            "edge_features": edge_features.astype(np.float32),
            "cell": structure.lattice.matrix.copy().astype(np.float32),
            "periodic": True
        }
        
        return graph_data
    
    def batch_convert(self, structures: List[Structure]) -> List[Dict]:
        """
        Convert a list of structures to graph representations.
        
        Args:
            structures: List of pymatgen Structure objects
            
        Returns:
            List of graph representation dictionaries
        """
        return [self.convert_structure(structure) for structure in structures]
    
    def get_feature_dimensions(self) -> Tuple[int, int]:
        """
        Get the dimensions of node and edge features.
        
        Returns:
            Tuple of (node_feature_dim, edge_feature_dim)
        """
        return len(self.node_features), len(self.edge_features)
    
    def _compute_node_features(self, element: Element) -> np.ndarray:
        """
        Compute features for an atom.
        
        Args:
            element: Pymatgen Element object
            
        Returns:
            Array of node features
        """
        # Cache element properties
        if element.symbol not in self._element_properties:
            props = {}
            props["atomic_number"] = element.Z
            props["electronegativity"] = element.X if element.X is not None else 0.0
            props["radius"] = element.atomic_radius if element.atomic_radius is not None else 1.0
            props["row"] = element.row
            props["group"] = element.group
            
            # Convert block to one-hot
            block_map = {"s": 0, "p": 1, "d": 2, "f": 3}
            props["block"] = block_map.get(element.block, 0)
            
            self._element_properties[element.symbol] = props
        
        # Get cached properties
        props = self._element_properties[element.symbol]
        
        # Create feature vector
        features = []
        for feat_name in self.node_features:
            if feat_name in props:
                features.append(props[feat_name])
            else:
                features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_edge_features(self, site1, site2, distance: float) -> np.ndarray:
        """
        Compute features for an edge between two atoms.
        
        Args:
            site1: First site
            site2: Second site
            distance: Distance between sites
            
        Returns:
            Array of edge features
        """
        # Compute displacement vector
        frac_diff = site2.frac_coords - site1.frac_coords
        
        # Wrap to [-0.5, 0.5)
        frac_diff = frac_diff - np.round(frac_diff)
        
        # Convert to Cartesian coordinates
        cart_diff = np.dot(frac_diff, site1.lattice.matrix)
        
        # Normalize
        cart_diff_norm = np.linalg.norm(cart_diff)
        if cart_diff_norm > 1e-8:
            cart_diff = cart_diff / cart_diff_norm
        
        # Create feature vector
        features = []
        for feat_name in self.edge_features:
            if feat_name == "distance":
                features.append(distance)
            elif feat_name == "vector_x":
                # Only use the x-component of the vector
                features.append(cart_diff[0])
            elif feat_name == "vector":
                # For backward compatibility, but should not be used with current config
                features.extend(cart_diff)
            else:
                features.append(0.0)
        
        return np.array(features, dtype=np.float32)


class CrystalGraphCollator:
    """
    Collator for batching crystal graphs.
    
    This class handles the batching of crystal graphs for use with PyTorch DataLoader.
    """
    
    def __init__(self, target_properties: List[str] = None):
        """
        Initialize the collator.
        
        Args:
            target_properties: List of property names to include as targets
        """
        self.target_properties = target_properties or ["formation_energy_per_atom", "band_gap", "is_topological"]
    
    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate a batch of crystal graphs.
        
        Args:
            batch: List of items from the dataset
            
        Returns:
            Batched data
        """
        # Extract graphs and targets
        graphs = [item["graph"] for item in batch]
        targets = {prop: [] for prop in self.target_properties}
        
        for item in batch:
            for prop in self.target_properties:
                targets[prop].append(item["targets"].get(prop, 0.0))
        
        # Batch node features
        batch_size = len(graphs)
        num_nodes = sum(graph["num_nodes"] for graph in graphs)
        
        # Create batched node features
        node_features = np.zeros((num_nodes, graphs[0]["node_features"].shape[1]), dtype=np.float32)
        
        # Create batched edge index and edge features
        edge_indices = []
        edge_features = []
        
        # Create batch assignment
        batch_assignment = np.zeros(num_nodes, dtype=np.int64)
        
        # Offset for node indices
        node_offset = 0
        
        # Process each graph
        for i, graph in enumerate(graphs):
            # Get number of nodes and edges
            n_nodes = graph["num_nodes"]
            n_edges = graph["edge_index"].shape[1]
            
            # Add node features
            node_features[node_offset:node_offset + n_nodes] = graph["node_features"]
            
            # Add edge index with offset
            if n_edges > 0:
                edge_index = graph["edge_index"].copy()
                edge_index += node_offset
                edge_indices.append(edge_index)
                
                # Add edge features
                edge_features.append(graph["edge_features"])
            
            # Add batch assignment
            batch_assignment[node_offset:node_offset + n_nodes] = i
            
            # Update offset
            node_offset += n_nodes
        
        # Concatenate edge indices and features
        if edge_indices:
            edge_index = np.concatenate(edge_indices, axis=1)
            edge_features = np.concatenate(edge_features, axis=0)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, graphs[0]["edge_features"].shape[1]), dtype=np.float32)
        
        # Convert targets to tensors with robust type conversion
        for prop in self.target_properties:
            # Ensure all values are properly converted to float
            converted_values = []
            for val in targets[prop]:
                try:
                    # Handle multi-dimensional values (lists, arrays)
                    if isinstance(val, (list, tuple, np.ndarray)):
                        # Extract first element if possible, otherwise use 0.0
                        if len(val) > 0:
                            # Recursively handle the first element
                            first_val = val[0]
                            if isinstance(first_val, str):
                                if first_val.lower() == 'true':
                                    converted_values.append(1.0)
                                elif first_val.lower() == 'false':
                                    converted_values.append(0.0)
                                elif first_val.lower() in ('na', 'nan', 'none', 'null', ''):
                                    converted_values.append(0.0)
                                else:
                                    try:
                                        converted_values.append(float(first_val))
                                    except (ValueError, TypeError):
                                        converted_values.append(0.0)
                            elif isinstance(first_val, bool):
                                converted_values.append(1.0 if first_val else 0.0)
                            elif first_val is None:
                                converted_values.append(0.0)
                            else:
                                try:
                                    converted_values.append(float(first_val))
                                except (ValueError, TypeError):
                                    converted_values.append(0.0)
                        else:
                            converted_values.append(0.0)
                    # Handle string representations of numbers
                    elif isinstance(val, str):
                        if val.lower() == 'true':
                            converted_values.append(1.0)
                        elif val.lower() == 'false':
                            converted_values.append(0.0)
                        elif val.lower() in ('na', 'nan', 'none', 'null', ''):
                            # Handle non-numeric strings
                            converted_values.append(0.0)
                        else:
                            try:
                                converted_values.append(float(val))
                            except (ValueError, TypeError):
                                converted_values.append(0.0)
                    # Handle boolean values
                    elif isinstance(val, bool):
                        converted_values.append(1.0 if val else 0.0)
                    # Handle None values
                    elif val is None:
                        converted_values.append(0.0)
                    # Handle numeric values
                    else:
                        try:
                            converted_values.append(float(val))
                        except (ValueError, TypeError):
                            converted_values.append(0.0)
                except Exception as e:
                    # Catch any other unexpected errors and default to 0.0
                    logger.warning(f"Unexpected error converting value '{val}' for property '{prop}': {e}. Using 0.0 instead.")
                    converted_values.append(0.0)
            
            targets[prop] = torch.tensor(converted_values, dtype=torch.float32)
        
        # Create batched data
        batched_data = {
            "x": torch.tensor(node_features, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "edge_attr": torch.tensor(edge_features, dtype=torch.float32),
            "batch": torch.tensor(batch_assignment, dtype=torch.long),
            "num_graphs": batch_size
        }
        
        # Add targets
        for prop, value in targets.items():
            batched_data[prop] = value
        
        return batched_data


class JARVISDataProcessor:
    """
    Class for processing JARVIS data using Polars for better performance.
    
    This class handles the processing of JARVIS datasets, including filtering,
    feature engineering, and preparation for machine learning.
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize the processor with the target directory.
        
        Args:
            data_dir: Directory to store processed data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def process_dataset(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Process a JARVIS dataset using Polars.
        
        Args:
            input_path: Path to the input JSON file
            output_path: Optional path to save the processed dataset
            
        Returns:
            Path to the processed dataset
        """
        logger.info(f"Processing dataset: {input_path}")
        
        # Create default output path if not provided
        if output_path is None:
            output_path = os.path.join(self.data_dir, "processed_dataset.json")
        
        try:
            # Load data
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Convert to Polars DataFrame
            df = pl.from_dicts(data)
            
            # Filter out entries without required properties
            required_props = ["formation_energy_per_atom", "band_gap", "structure"]
            
            for prop in required_props:
                df = df.filter(pl.col(prop).is_not_null())
            
            # Add topological classification if not present
            if "is_topological" not in df.columns:
                # Use spillage as indicator of topological character
                if "spillage" in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col("spillage") > 0.5)
                        .then(True)
                        .otherwise(False)
                        .alias("is_topological")
                    )
                else:
                    # Default to False if no spillage data
                    df = df.with_columns(pl.lit(False).alias("is_topological"))
            
            # Add Z2 invariant if not present
            if "z2_invariant" not in df.columns:
                df = df.with_columns(pl.lit([[0, 0, 0, 0]]).alias("z2_invariant"))
            
            # Calculate sustainability score
            df = self._add_sustainability_score(df)
            
            # Convert back to list of dictionaries
            processed_data = df.to_dicts()
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(processed_data, f)
            
            logger.info(f"Processed dataset saved to {output_path} with {len(processed_data)} entries")
            
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise RuntimeError(f"Failed to process dataset {input_path}")
        
        return output_path
    
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
        train_path = os.path.join(self.data_dir, "train_dataset.json")
        val_path = os.path.join(self.data_dir, "val_dataset.json")
        
        try:
            # Load data
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Convert to Polars DataFrame
            df = pl.from_dicts(data)
            
            # Shuffle the data
            df = df.sample(fraction=1.0, seed=42)
            
            # Calculate split index
            split_idx = int(len(df) * train_ratio)
            
            # Split the data
            train_df = df.slice(0, split_idx)
            val_df = df.slice(split_idx)
            
            # Convert back to list of dictionaries
            train_data = train_df.to_dicts()
            val_data = val_df.to_dicts()
            
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
    
    def create_topological_subset(self, input_path: str) -> str:
        """
        Create a subset of topological materials.
        
        Args:
            input_path: Path to the input JSON file
            
        Returns:
            Path to the topological subset
        """
        logger.info(f"Creating topological subset for {input_path}")
        
        # Create output path
        output_path = os.path.join(self.data_dir, "topological_subset.json")
        
        try:
            # Load data
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Convert to Polars DataFrame
            df = pl.from_dicts(data)
            
            # Filter for topological materials
            topo_df = df.filter(pl.col("is_topological") == True)
            
            # Convert back to list of dictionaries
            topo_data = topo_df.to_dicts()
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(topo_data, f)
            
            logger.info(f"Created topological subset with {len(topo_data)} entries at {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating topological subset: {e}")
            raise RuntimeError(f"Failed to create topological subset for {input_path}")
        
        return output_path
    
    def _add_sustainability_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add sustainability score to the dataset.
        
        Args:
            df: Polars DataFrame
            
        Returns:
            DataFrame with sustainability score
        """
        # Element properties for sustainability calculation
        element_abundance = {
            "H": 1400, "He": 0.008, "Li": 20, "Be": 2.8, "B": 10, "C": 200, "N": 20, "O": 461000,
            "F": 585, "Ne": 0.005, "Na": 23600, "Mg": 23300, "Al": 82300, "Si": 282000, "P": 1050,
            "S": 350, "Cl": 145, "Ar": 3.5, "K": 20900, "Ca": 41500, "Sc": 22, "Ti": 5650,
            "V": 120, "Cr": 102, "Mn": 950, "Fe": 56300, "Co": 25, "Ni": 84, "Cu": 60, "Zn": 70,
            "Ga": 19, "Ge": 1.5, "As": 1.8, "Se": 0.05, "Br": 2.4, "Kr": 0.0001
        }
        
        element_toxicity = {
            "H": 0, "He": 0, "Li": 2, "Be": 8, "B": 3, "C": 0, "N": 0, "O": 0,
            "F": 3, "Ne": 0, "Na": 1, "Mg": 0, "Al": 2, "Si": 1, "P": 3, "S": 1,
            "Cl": 2, "Ar": 0, "K": 1, "Ca": 0, "Sc": 2, "Ti": 1, "V": 4, "Cr": 5,
            "Mn": 3, "Fe": 1, "Co": 4, "Ni": 4, "Cu": 2, "Zn": 2, "Ga": 3, "Ge": 2,
            "As": 7, "Se": 4, "Br": 3, "Kr": 0
        }
        
        # Function to calculate sustainability score
        def calculate_sustainability(formula: str) -> float:
            # Parse formula to get element counts
            element_counts = {}
            current_element = ""
            current_count = ""
            
            for char in formula:
                if char.isupper():
                    if current_element:
                        count = int(current_count) if current_count else 1
                        element_counts[current_element] = element_counts.get(current_element, 0) + count
                    current_element = char
                    current_count = ""
                elif char.islower():
                    current_element += char
                elif char.isdigit():
                    current_count += char
            
            # Add the last element
            if current_element:
                count = int(current_count) if current_count else 1
                element_counts[current_element] = element_counts.get(current_element, 0) + count
            
            # Calculate abundance score (higher is better)
            total_atoms = sum(element_counts.values())
            abundance_score = 0
            for element, count in element_counts.items():
                if element in element_abundance:
                    # Log scale for abundance
                    abundance = element_abundance.get(element, 0.001)
                    abundance_score += (np.log10(abundance + 1) * count / total_atoms)
                
            # Normalize to [0, 1]
            abundance_score = min(1.0, abundance_score / 5.0)
            
            # Calculate toxicity score (lower is better)
            toxicity_score = 0
            for element, count in element_counts.items():
                if element in element_toxicity:
                    toxicity_score += (element_toxicity.get(element, 0) * count / total_atoms)
            
            # Normalize to [0, 1] and invert (higher is better)
            toxicity_score = 1.0 - min(1.0, toxicity_score / 8.0)
            
            # Combine scores (equal weight)
            sustainability_score = 0.5 * abundance_score + 0.5 * toxicity_score
            
            return sustainability_score
        
        # Apply sustainability calculation to each formula
        df = df.with_columns(
            pl.col("formula").map_elements(calculate_sustainability).alias("sustainability_score")
        )
        
        return df
