"""
Initialization file for the Topological Materials Diffusion project.

This file makes the src directory a Python package.
"""

from src.data import (
    JARVISDataDownloader, 
    CrystalGraphConverter, 
    CrystalGraphDataset, 
    CrystalGraphCollator
)

from src.model import (
    CrystalGraphDiffusionModel, 
    DiffusionProcess, 
    GraphAttention, 
    EquivariantGraphConv, 
    SinusoidalPositionEmbeddings
)

from src.training import (
    DiffusionTrainer, 
    MaterialValidator
)

from src.utils import (
    load_structure_from_file, 
    save_structure_to_file, 
    calculate_element_statistics,
    calculate_sustainability_metrics,
    get_periodic_table_properties,
    setup_logging,
    load_config,
    save_config,
    analyze_structure,
    visualize_structure,
    create_composition_heatmap,
    create_property_correlation_plot,
    create_sustainability_radar_chart
)

__all__ = [
    # Data module
    'JARVISDataDownloader',
    'CrystalGraphConverter',
    'CrystalGraphDataset',
    'CrystalGraphCollator',
    
    # Model module
    'CrystalGraphDiffusionModel',
    'DiffusionProcess',
    'GraphAttention',
    'EquivariantGraphConv',
    'SinusoidalPositionEmbeddings',
    
    # Training module
    'DiffusionTrainer',
    'MaterialValidator',
    
    # Utils module
    'load_structure_from_file',
    'save_structure_to_file',
    'calculate_element_statistics',
    'calculate_sustainability_metrics',
    'get_periodic_table_properties',
    'setup_logging',
    'load_config',
    'save_config',
    'analyze_structure',
    'visualize_structure',
    'create_composition_heatmap',
    'create_property_correlation_plot',
    'create_sustainability_radar_chart'
]
