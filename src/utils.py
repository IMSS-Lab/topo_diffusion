"""
Utility functions for the Topological Materials Diffusion project.

This module contains various utility functions used across the project,
including crystal structure manipulation, sustainability metrics calculation,
and other helper functions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json
import os
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import requests
from io import StringIO
import re

logger = logging.getLogger(__name__)

# Element property data
ELEMENT_DATA = {
    # Abundance in Earth's crust (ppm by weight)
    "abundance": {
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
    },
    
    # Toxicity score (0-10, higher is more toxic)
    "toxicity": {
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
    },
    
    # Processing energy (relative scale, higher means more energy intensive)
    "processing_energy": {
        "H": 5, "He": 8, "Li": 7, "Be": 8, "B": 7, "C": 3, "N": 6, "O": 2,
        "F": 7, "Ne": 8, "Na": 4, "Mg": 5, "Al": 6, "Si": 5, "P": 6, "S": 4,
        "Cl": 5, "Ar": 8, "K": 3, "Ca": 4, "Sc": 7, "Ti": 8, "V": 8, "Cr": 7,
        "Mn": 6, "Fe": 5, "Co": 7, "Ni": 7, "Cu": 6, "Zn": 5, "Ga": 7, "Ge": 7,
        "As": 6, "Se": 7, "Br": 6, "Kr": 8, "Rb": 5, "Sr": 6, "Y": 7, "Zr": 8,
        "Nb": 8, "Mo": 8, "Tc": 9, "Ru": 8, "Rh": 9, "Pd": 8, "Ag": 7, "Cd": 6,
        "In": 7, "Sn": 6, "Sb": 7, "Te": 7, "I": 6, "Xe": 8, "Cs": 5, "Ba": 6,
        "La": 7, "Ce": 7, "Pr": 8, "Nd": 8, "Pm": 9, "Sm": 8, "Eu": 8, "Gd": 8,
        "Tb": 8, "Dy": 8, "Ho": 8, "Er": 8, "Tm": 8, "Yb": 8, "Lu": 8, "Hf": 8,
        "Ta": 9, "W": 8, "Re": 9, "Os": 9, "Ir": 9, "Pt": 8, "Au": 7, "Hg": 6,
        "Tl": 7, "Pb": 5, "Bi": 6, "Po": 9, "At": 9, "Rn": 9, "Fr": 9, "Ra": 9,
        "Ac": 9, "Th": 8, "Pa": 9, "U": 8, "Np": 9, "Pu": 9
    },
    
    # Recyclability score (0-10, higher is more recyclable)
    "recyclability": {
        "H": 8, "He": 7, "Li": 6, "Be": 5, "B": 6, "C": 8, "N": 7, "O": 9,
        "F": 6, "Ne": 7, "Na": 7, "Mg": 8, "Al": 9, "Si": 8, "P": 7, "S": 7,
        "Cl": 6, "Ar": 7, "K": 7, "Ca": 8, "Sc": 6, "Ti": 7, "V": 6, "Cr": 7,
        "Mn": 7, "Fe": 9, "Co": 7, "Ni": 8, "Cu": 9, "Zn": 8, "Ga": 6, "Ge": 6,
        "As": 5, "Se": 5, "Br": 6, "Kr": 7, "Rb": 6, "Sr": 6, "Y": 6, "Zr": 7,
        "Nb": 7, "Mo": 7, "Tc": 4, "Ru": 7, "Rh": 8, "Pd": 9, "Ag": 9, "Cd": 7,
        "In": 6, "Sn": 8, "Sb": 7, "Te": 6, "I": 6, "Xe": 7, "Cs": 6, "Ba": 6,
        "La": 6, "Ce": 6, "Pr": 6, "Nd": 6, "Pm": 4, "Sm": 6, "Eu": 6, "Gd": 6,
        "Tb": 6, "Dy": 6, "Ho": 6, "Er": 6, "Tm": 6, "Yb": 6, "Lu": 6, "Hf": 7,
        "Ta": 7, "W": 7, "Re": 7, "Os": 7, "Ir": 8, "Pt": 9, "Au": 9, "Hg": 7,
        "Tl": 6, "Pb": 8, "Bi": 7, "Po": 3, "At": 3, "Rn": 4, "Fr": 4, "Ra": 4,
        "Ac": 5, "Th": 5, "Pa": 4, "U": 6, "Np": 4, "Pu": 4
    }
}

def load_structure_from_file(file_path: str) -> Structure:
    """
    Load a crystal structure from a file.
    
    Args:
        file_path: Path to the structure file
        
    Returns:
        Pymatgen Structure object
    """
    logger.info(f"Loading structure from {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Structure file not found: {file_path}")
    
    # Determine file format from extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.cif':
            # Load CIF file
            structure = Structure.from_file(file_path)
        elif file_ext == '.poscar' or file_ext == '.vasp':
            # Load VASP POSCAR file
            from pymatgen.io.vasp import Poscar
            poscar = Poscar.from_file(file_path)
            structure = poscar.structure
        elif file_ext == '.json':
            # Load JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            structure = Structure.from_dict(data)
        elif file_ext == '.xsf':
            # Load XSF file
            from pymatgen.io.xcrysden import XSF
            xsf = XSF.from_file(file_path)
            structure = xsf.structure
        else:
            # Try generic loader
            structure = Structure.from_file(file_path)
    except Exception as e:
        logger.error(f"Error loading structure from {file_path}: {e}")
        # Create a simple cubic structure as fallback
        from pymatgen.core.lattice import Lattice
        lattice = Lattice.cubic(5.0)
        structure = Structure(lattice, ["Si", "Si", "Si", "Si"], 
                             [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        logger.warning(f"Created fallback structure: {structure.formula}")
    
    logger.info(f"Loaded structure: {structure.formula} with {len(structure)} atoms")
    return structure

def save_structure_to_file(structure: Structure, file_path: str) -> None:
    """
    Save a crystal structure to a file.
    
    Args:
        structure: Pymatgen Structure object
        file_path: Path to save the structure file
    """
    logger.info(f"Saving structure to {file_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Determine file format from extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.cif':
            # Save as CIF
            structure.to(filename=file_path)
        elif file_ext == '.poscar' or file_ext == '.vasp':
            # Save as VASP POSCAR
            from pymatgen.io.vasp import Poscar
            poscar = Poscar(structure)
            poscar.write_file(file_path)
        elif file_ext == '.json':
            # Save as JSON
            with open(file_path, 'w') as f:
                json.dump(structure.as_dict(), f, indent=2)
        elif file_ext == '.xsf':
            # Save as XSF
            from pymatgen.io.xcrysden import XSF
            xsf = XSF(structure)
            xsf.write_file(file_path)
        else:
            # Default to CIF
            structure.to(filename=file_path)
    except Exception as e:
        logger.error(f"Error saving structure to {file_path}: {e}")
        raise

def calculate_element_statistics(structures: List[Structure]) -> Dict:
    """
    Calculate element statistics across a list of structures.
    
    Args:
        structures: List of Pymatgen Structure objects
        
    Returns:
        Dictionary of element statistics
    """
    logger.info(f"Calculating element statistics for {len(structures)} structures")
    
    # Initialize counters
    element_counts = {}
    total_atoms = 0
    
    # Count elements across all structures
    for structure in structures:
        for element in structure.composition.elements:
            symbol = element.symbol
            count = structure.composition[element]
            
            if symbol not in element_counts:
                element_counts[symbol] = 0
            
            element_counts[symbol] += count
            total_atoms += count
    
    # Calculate average composition
    average_composition = {symbol: count / total_atoms for symbol, count in element_counts.items()}
    
    # Identify rare elements (less than 0.1% abundance in Earth's crust)
    rare_elements = []
    for symbol in element_counts:
        abundance = ELEMENT_DATA["abundance"].get(symbol, 0)
        if abundance < 1.0:  # ppm threshold for rare elements
            rare_elements.append(symbol)
    
    # Calculate element diversity
    element_diversity = len(element_counts)
    
    # Calculate average atomic number
    atomic_numbers = [Element(symbol).Z for symbol in element_counts]
    avg_atomic_number = sum(atomic_numbers) / len(atomic_numbers) if atomic_numbers else 0
    
    # Return statistics
    element_stats = {
        "element_counts": element_counts,
        "average_composition": average_composition,
        "rare_elements": rare_elements,
        "element_diversity": element_diversity,
        "average_atomic_number": avg_atomic_number
    }
    
    return element_stats

def calculate_sustainability_metrics(elements: List[str]) -> Dict:
    """
    Calculate sustainability metrics for a list of elements.
    
    Args:
        elements: List of element symbols
        
    Returns:
        Dictionary of sustainability metrics
    """
    logger.info(f"Calculating sustainability metrics for {elements}")
    
    # Initialize scores
    abundance_scores = {}
    toxicity_scores = {}
    processing_scores = {}
    recyclability_scores = {}
    
    # Calculate scores for each element
    for element in elements:
        # Abundance score (higher is better)
        abundance = ELEMENT_DATA["abundance"].get(element, 0)
        # Log scale transformation to handle wide range of values
        if abundance > 0:
            abundance_score = min(1.0, max(0.0, (np.log10(abundance) + 3) / 8))
        else:
            abundance_score = 0.0
        abundance_scores[element] = abundance_score
        
        # Toxicity score (higher is better = less toxic)
        toxicity = ELEMENT_DATA["toxicity"].get(element, 5)
        toxicity_score = 1.0 - (toxicity / 10.0)
        toxicity_scores[element] = toxicity_score
        
        # Processing energy score (higher is better = less energy)
        processing = ELEMENT_DATA["processing_energy"].get(element, 5)
        processing_score = 1.0 - (processing / 10.0)
        processing_scores[element] = processing_score
        
        # Recyclability score (higher is better)
        recyclability = ELEMENT_DATA["recyclability"].get(element, 5)
        recyclability_score = recyclability / 10.0
        recyclability_scores[element] = recyclability_score
    
    # Calculate overall scores
    if elements:
        avg_abundance = sum(abundance_scores.values()) / len(elements)
        avg_toxicity = sum(toxicity_scores.values()) / len(elements)
        avg_processing = sum(processing_scores.values()) / len(elements)
        avg_recyclability = sum(recyclability_scores.values()) / len(elements)
        
        # Overall sustainability score (weighted average)
        overall_score = (
            0.3 * avg_abundance +
            0.3 * avg_toxicity +
            0.2 * avg_processing +
            0.2 * avg_recyclability
        )
    else:
        avg_abundance = 0.0
        avg_toxicity = 0.0
        avg_processing = 0.0
        avg_recyclability = 0.0
        overall_score = 0.0
    
    # Return metrics
    sustainability_metrics = {
        "abundance_scores": abundance_scores,
        "toxicity_scores": toxicity_scores,
        "processing_scores": processing_scores,
        "recyclability_scores": recyclability_scores,
        "average_abundance": avg_abundance,
        "average_toxicity": avg_toxicity,
        "average_processing": avg_processing,
        "average_recyclability": avg_recyclability,
        "overall_score": overall_score
    }
    
    return sustainability_metrics

def get_periodic_table_properties() -> Dict:
    """
    Get properties of elements from the periodic table.
    
    Returns:
        Dictionary mapping element symbols to properties
    """
    logger.info("Getting periodic table properties")
    
    properties = {}
    
    # Iterate through all elements
    for z in range(1, 103):  # Z=1 (H) to Z=102 (No)
        try:
            element = Element.from_Z(z)
            symbol = element.symbol
            
            # Collect basic properties
            props = {
                "atomic_number": element.Z,
                "name": element.name,
                "mass": element.atomic_mass,
                "group": element.group,
                "row": element.row,
                "block": element.block,
                "radius": element.atomic_radius,
                "electronegativity": element.X,
                "electron_affinity": element.electron_affinity,
                "ionization_energy": element.ionization_energy
            }
            
            # Add sustainability-related properties
            props["abundance"] = ELEMENT_DATA["abundance"].get(symbol, 0)
            props["toxicity"] = ELEMENT_DATA["toxicity"].get(symbol, 5)
            props["processing_energy"] = ELEMENT_DATA["processing_energy"].get(symbol, 5)
            props["recyclability"] = ELEMENT_DATA["recyclability"].get(symbol, 5)
            
            # Store properties
            properties[symbol] = props
            
        except Exception as e:
            logger.warning(f"Error getting properties for element Z={z}: {e}")
    
    return properties

def analyze_structure(structure: Structure) -> Dict:
    """
    Analyze a crystal structure and extract key properties.
    
    Args:
        structure: Pymatgen Structure object
        
    Returns:
        Dictionary of structural properties
    """
    logger.info(f"Analyzing structure: {structure.formula}")
    
    # Basic properties
    properties = {
        "formula": structure.formula,
        "reduced_formula": structure.composition.reduced_formula,
        "num_atoms": len(structure),
        "volume": structure.volume,
        "density": structure.density,
        "elements": [str(element) for element in structure.composition.elements],
        "element_counts": {str(element): structure.composition[element] for element in structure.composition.elements}
    }
    
    # Lattice parameters
    lattice = structure.lattice
    properties["lattice"] = {
        "a": lattice.a,
        "b": lattice.b,
        "c": lattice.c,
        "alpha": lattice.alpha,
        "beta": lattice.beta,
        "gamma": lattice.gamma,
        "volume": lattice.volume
    }
    
    # Symmetry analysis
    try:
        spg_analyzer = SpacegroupAnalyzer(structure)
        properties["symmetry"] = {
            "spacegroup_symbol": spg_analyzer.get_space_group_symbol(),
            "spacegroup_number": spg_analyzer.get_space_group_number(),
            "crystal_system": spg_analyzer.get_crystal_system(),
            "point_group": spg_analyzer.get_point_group_symbol()
        }
    except Exception as e:
        logger.warning(f"Error in symmetry analysis: {e}")
        properties["symmetry"] = {
            "spacegroup_symbol": "Unknown",
            "spacegroup_number": 0,
            "crystal_system": "Unknown",
            "point_group": "Unknown"
        }
    
    # Connectivity analysis
    try:
        voro = VoronoiConnectivity(structure)
        connectivity = voro.connectivity_array
        
        # Calculate average coordination number
        coord_numbers = [sum(connectivity[i]) for i in range(len(structure))]
        properties["connectivity"] = {
            "average_coordination": sum(coord_numbers) / len(coord_numbers) if coord_numbers else 0,
            "min_coordination": min(coord_numbers) if coord_numbers else 0,
            "max_coordination": max(coord_numbers) if coord_numbers else 0
        }
    except Exception as e:
        logger.warning(f"Error in connectivity analysis: {e}")
        properties["connectivity"] = {
            "average_coordination": 0,
            "min_coordination": 0,
            "max_coordination": 0
        }
    
    # Calculate sustainability metrics
    properties["sustainability"] = calculate_sustainability_metrics(properties["elements"])
    
    return properties

def visualize_structure(structure: Structure, output_path: str = None, show: bool = False) -> str:
    """
    Visualize a crystal structure and save the visualization.
    
    Args:
        structure: Pymatgen Structure object
        output_path: Path to save the visualization
        show: Whether to show the visualization
        
    Returns:
        Path to the saved visualization
    """
    logger.info(f"Visualizing structure: {structure.formula}")
    
    # Create default output path if not provided
    if output_path is None:
        os.makedirs("visualizations", exist_ok=True)
        output_path = f"visualizations/{structure.composition.reduced_formula}.png"
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get element colors
    element_colors = {
        "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5",
        "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "Ne": "#B3E3F5",
        "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000",
        "S": "#FFFF30", "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00",
        "Sc": "#E6E6E6", "Ti": "#BFC2C7", "V": "#A6A6AB", "Cr": "#8A99C7", "Mn": "#9C7AC7",
        "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0",
        "Ga": "#C28F8F", "Ge": "#668F8F", "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929",
        "Kr": "#5CB8D1", "Rb": "#702EB0", "Sr": "#00FF00", "Y": "#94FFFF", "Zr": "#94E0E0",
        "Nb": "#73C2C9", "Mo": "#54B5B5", "Tc": "#3B9E9E", "Ru": "#248F8F", "Rh": "#0A7D8C",
        "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F", "In": "#A67573", "Sn": "#668080",
        "Sb": "#9E63B5", "Te": "#D47A00", "I": "#940094", "Xe": "#429EB0", "Cs": "#57178F",
        "Ba": "#00C900", "La": "#70D4FF", "Ce": "#FFFFC7", "Pr": "#D9FFC7", "Nd": "#C7FFC7",
        "Pm": "#A3FFC7", "Sm": "#8FFFC7", "Eu": "#61FFC7", "Gd": "#45FFC7", "Tb": "#30FFC7",
        "Dy": "#1FFFC7", "Ho": "#00FF9C", "Er": "#00E675", "Tm": "#00D452", "Yb": "#00BF38",
        "Lu": "#00AB24", "Hf": "#4DC2FF", "Ta": "#4DA6FF", "W": "#2194D6", "Re": "#267DAB",
        "Os": "#266696", "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123", "Hg": "#B8B8D0",
        "Tl": "#A6544D", "Pb": "#575961", "Bi": "#9E4FB5", "Po": "#AB5C00", "At": "#754F45",
        "Rn": "#428296", "Fr": "#420066", "Ra": "#007D00", "Ac": "#70ABFA", "Th": "#00BAFF",
        "Pa": "#00A1FF", "U": "#008FFF", "Np": "#0080FF", "Pu": "#006BFF"
    }
    
    # Plot unit cell
    lattice = structure.lattice
    origin = np.array([0, 0, 0])
    a = lattice.matrix[0]
    b = lattice.matrix[1]
    c = lattice.matrix[2]
    
    # Plot unit cell edges
    for start, end in [
        (origin, a),
        (origin, b),
        (origin, c),
        (a, a+b),
        (a, a+c),
        (b, b+a),
        (b, b+c),
        (c, c+a),
        (c, c+b),
        (a+b, a+b+c),
        (a+c, a+c+b),
        (b+c, b+c+a)
    ]:
        ax.plot3D(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            'k-', alpha=0.5
        )
    
    # Plot atoms
    for site in structure:
        # Get element color
        element = site.specie.symbol
        color = element_colors.get(element, "#CCCCCC")
        
        # Get atomic radius (scaled)
        radius = site.specie.atomic_radius
        if radius is None:
            radius = 1.0
        radius = radius / 10  # Scale down for visualization
        
        # Get position
        pos = site.coords
        
        # Plot atom
        ax.scatter(
            pos[0], pos[1], pos[2],
            color=color,
            s=radius * 100,
            edgecolor='black',
            alpha=0.8
        )
        
        # Add element label
        ax.text(pos[0], pos[1], pos[2], element, fontsize=8)
    
    # Set axis labels
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    
    # Set title
    ax.set_title(f"Structure: {structure.composition.reduced_formula}")
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    logger.info(f"Saved visualization to {output_path}")
    
    return output_path

def setup_logging(log_level="INFO", log_file=None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    
    # Set matplotlib logging level to WARNING to reduce verbosity
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized at level {log_level}")

def load_config(config_path: str) -> Dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary of configuration parameters
    """
    logger.info(f"Loading configuration from {config_path}")
    
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML file
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise
    
    logger.info(f"Loaded configuration with {len(config)} top-level keys")
    
    return config

def save_config(config: Dict, config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Dictionary of configuration parameters
        config_path: Path to save the configuration file
    """
    logger.info(f"Saving configuration to {config_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    # Save YAML file
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
        raise
    
    logger.info(f"Saved configuration with {len(config)} top-level keys")

def fetch_element_data(element: str) -> Dict:
    """
    Fetch data for an element from online sources.
    
    Args:
        element: Element symbol
        
    Returns:
        Dictionary of element data
    """
    logger.info(f"Fetching data for element: {element}")
    
    # Initialize data dictionary
    element_data = {}
    
    try:
        # Get basic data from pymatgen
        el = Element(element)
        element_data = {
            "symbol": el.symbol,
            "name": el.name,
            "atomic_number": el.Z,
            "atomic_mass": el.atomic_mass,
            "group": el.group,
            "period": el.row,
            "block": el.block,
            "category": el.category,
            "atomic_radius": el.atomic_radius,
            "electronegativity": el.X,
            "electron_affinity": el.electron_affinity,
            "ionization_energy": el.ionization_energy
        }
        
        # Add sustainability data
        element_data["abundance"] = ELEMENT_DATA["abundance"].get(element, 0)
        element_data["toxicity"] = ELEMENT_DATA["toxicity"].get(element, 5)
        element_data["processing_energy"] = ELEMENT_DATA["processing_energy"].get(element, 5)
        element_data["recyclability"] = ELEMENT_DATA["recyclability"].get(element, 5)
        
    except Exception as e:
        logger.error(f"Error fetching data for element {element}: {e}")
    
    return element_data

def create_composition_heatmap(compositions: List[Dict], output_path: str = None, show: bool = False) -> str:
    """
    Create a heatmap of element compositions across multiple structures.
    
    Args:
        compositions: List of composition dictionaries
        output_path: Path to save the heatmap
        show: Whether to show the heatmap
        
    Returns:
        Path to the saved heatmap
    """
    logger.info(f"Creating composition heatmap for {len(compositions)} structures")
    
    # Create default output path if not provided
    if output_path is None:
        os.makedirs("visualizations", exist_ok=True)
        output_path = "visualizations/composition_heatmap.png"
    
    # Extract all elements
    all_elements = set()
    for comp in compositions:
        all_elements.update(comp.keys())
    
    # Sort elements by atomic number
    all_elements = sorted(all_elements, key=lambda x: Element(x).Z)
    
    # Create data matrix
    data = np.zeros((len(compositions), len(all_elements)))
    
    # Fill data matrix
    for i, comp in enumerate(compositions):
        for j, element in enumerate(all_elements):
            data[i, j] = comp.get(element, 0)
    
    # Normalize by row (structure)
    row_sums = data.sum(axis=1, keepdims=True)
    data_norm = data / row_sums
    
    # Create figure
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    # Create heatmap
    cmap = LinearSegmentedColormap.from_list(
        "composition_cmap", ["#FFFFFF", "#FFC107", "#FF5722", "#E91E63", "#9C27B0"]
    )
    im = ax.imshow(data_norm, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fraction of Composition')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(all_elements)))
    ax.set_xticklabels(all_elements, rotation=90)
    ax.set_yticks(np.arange(len(compositions)))
    ax.set_yticklabels([f"Structure {i+1}" for i in range(len(compositions))])
    
    # Set title and labels
    ax.set_title("Element Composition Across Structures")
    
    # Add grid
    ax.set_xticks(np.arange(-.5, len(all_elements), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(compositions), 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.2)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    logger.info(f"Saved composition heatmap to {output_path}")
    
    return output_path

def create_property_correlation_plot(properties: List[Dict], x_prop: str, y_prop: str, 
                                    output_path: str = None, show: bool = False) -> str:
    """
    Create a correlation plot between two properties.
    
    Args:
        properties: List of property dictionaries
        x_prop: Name of the x-axis property
        y_prop: Name of the y-axis property
        output_path: Path to save the plot
        show: Whether to show the plot
        
    Returns:
        Path to the saved plot
    """
    logger.info(f"Creating correlation plot between {x_prop} and {y_prop}")
    
    # Create default output path if not provided
    if output_path is None:
        os.makedirs("visualizations", exist_ok=True)
        output_path = f"visualizations/correlation_{x_prop}_{y_prop}.png"
    
    # Extract property values
    x_values = []
    y_values = []
    labels = []
    
    for i, prop_dict in enumerate(properties):
        # Extract x property (support nested dictionaries with dot notation)
        x_val = prop_dict
        for key in x_prop.split('.'):
            if isinstance(x_val, dict) and key in x_val:
                x_val = x_val[key]
            else:
                x_val = None
                break
        
        # Extract y property (support nested dictionaries with dot notation)
        y_val = prop_dict
        for key in y_prop.split('.'):
            if isinstance(y_val, dict) and key in y_val:
                y_val = y_val[key]
            else:
                y_val = None
                break
        
        # Skip if either property is missing
        if x_val is None or y_val is None:
            continue
        
        # Add to lists
        x_values.append(float(x_val))
        y_values.append(float(y_val))
        labels.append(f"Structure {i+1}")
    
    # Create figure
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    
    # Create scatter plot
    scatter = ax.scatter(x_values, y_values, c=range(len(x_values)), 
                        cmap='viridis', alpha=0.8, s=100, edgecolor='black')
    
    # Add labels for each point
    for i, label in enumerate(labels):
        ax.annotate(label, (x_values[i], y_values[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Calculate and add trend line
    if len(x_values) > 1:
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        ax.plot(x_values, p(x_values), "r--", alpha=0.8)
        
        # Calculate correlation coefficient
        corr_coef = np.corrcoef(x_values, y_values)[0, 1]
        ax.text(0.05, 0.95, f"Correlation: {corr_coef:.3f}", transform=ax.transAxes,
               fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set title and labels
    ax.set_title(f"Correlation between {x_prop} and {y_prop}")
    ax.set_xlabel(x_prop)
    ax.set_ylabel(y_prop)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    logger.info(f"Saved correlation plot to {output_path}")
    
    return output_path

def create_sustainability_radar_chart(structures: List[Structure], output_path: str = None, show: bool = False) -> str:
    """
    Create a radar chart of sustainability metrics for multiple structures.
    
    Args:
        structures: List of Pymatgen Structure objects
        output_path: Path to save the chart
        show: Whether to show the chart
        
    Returns:
        Path to the saved chart
    """
    logger.info(f"Creating sustainability radar chart for {len(structures)} structures")
    
    # Create default output path if not provided
    if output_path is None:
        os.makedirs("visualizations", exist_ok=True)
        output_path = "visualizations/sustainability_radar.png"
    
    # Calculate sustainability metrics for each structure
    metrics = []
    for structure in structures:
        elements = [str(element) for element in structure.composition.elements]
        metrics.append(calculate_sustainability_metrics(elements))
    
    # Extract metrics for radar chart
    categories = ['Abundance', 'Toxicity', 'Processing', 'Recyclability']
    values = []
    
    for metric in metrics:
        values.append([
            metric['average_abundance'],
            metric['average_toxicity'],
            metric['average_processing'],
            metric['average_recyclability']
        ])
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    
    # Create radar chart
    ax = fig.add_subplot(111, polar=True)
    
    # Set the angles for each category
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Set the labels for each category
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Plot each structure
    for i, vals in enumerate(values):
        vals = vals + vals[:1]  # Close the loop
        ax.plot(angles, vals, linewidth=2, linestyle='solid', label=f"Structure {i+1}")
        ax.fill(angles, vals, alpha=0.1)
    
    # Set y-limits
    ax.set_ylim(0, 1)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Set title
    ax.set_title("Sustainability Metrics Comparison", size=15, y=1.1)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    logger.info(f"Saved sustainability radar chart to {output_path}")
    
    return output_path
