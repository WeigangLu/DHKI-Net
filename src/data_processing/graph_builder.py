# src/data_processing/graph_builder.py

import logging
import torch
import numpy as np
from itertools import product
from .gridding import UniformGridMapper, GeoJsonMapper, BaseMapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_graph_adjacency(mapper: BaseMapper) -> torch.Tensor:
    """
    Builds the graph adjacency information (edge_index) for GNN layers.

    Args:
        mapper (BaseMapper): The initialized mapper object ('uniform' or 'geojson').

    Returns:
        torch.Tensor: A tensor of shape (2, num_edges) representing graph connectivity.
    """
    logging.info(f"Building graph adjacency for mapper type: {type(mapper).__name__}")
    
    edges = []
    
    if isinstance(mapper, UniformGridMapper):
        rows, cols = mapper.n_rows, mapper.n_cols
        total_zones = rows * cols
        
        # Iterate over each grid cell to find its neighbors
        for r in range(rows):
            for c in range(cols):
                current_zone_id = r * cols + c
                # Check all 8 neighbors (and the cell itself)
                for dr, dc in product([-1, 0, 1], [-1, 0, 1]):
                    if dr == 0 and dc == 0:
                        continue # A node is its own neighbor in GCNs, but we'll let PyG handle self-loops.
                    
                    nr, nc = r + dr, c + dc
                    
                    # Check if neighbor is within bounds
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor_zone_id = nr * cols + nc
                        edges.append([current_zone_id, neighbor_zone_id])
                        
    elif isinstance(mapper, GeoJsonMapper):
        gdf = mapper.zones_gdf
        # Create a mapping from zone_id property (e.g., 'ca_eng') to an integer index
        zone_to_idx = {zone_id: i for i, zone_id in enumerate(gdf['zone_id'])}
        
        # Find adjacent polygons
        for i, zone in gdf.iterrows():
            # touches() finds polygons that share a border
            neighbors = gdf[gdf.geometry.touches(zone.geometry)]
            if not neighbors.empty:
                current_zone_idx = zone_to_idx[zone['zone_id']]
                for _, neighbor in neighbors.iterrows():
                    neighbor_idx = zone_to_idx[neighbor['zone_id']]
                    edges.append([current_zone_idx, neighbor_idx])
                    
    else:
        raise TypeError("Unsupported mapper type for graph building.")
        
    # Convert to a torch tensor of shape (2, num_edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    logging.info(f"Built graph with {edge_index.shape[1]} edges.")
    return edge_index