import logging
import pickle
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import geopandas as gpd
from scipy.spatial.distance import cdist # For Haversine distance
from haversine import haversine, Unit # For Haversine distance

logger = logging.getLogger(__name__)

class DHKIDataset(Dataset):
    def __init__(self, config: Dict, dataset_type: str = 'train'):
        super().__init__()
        logger.info(f"Initializing DHKIDataset for {dataset_type}...")
        self.dataset_type = dataset_type
        self.config = config
        self.look_back = config['features']['historical_look_back_steps']
        self.num_contrastive_negatives = config['data']['num_contrastive_negatives']

        # --- 1. Load data from files ---
        data_path = Path(config['data']['processed_output_dir'])
        feature_suffix = f"_scaler-{config['features']['scaling_method']}_lagged-{config['features']['enable_lagged_features']}_rolling-{config['features']['enable_rolling_window_features']}_weather-{config['features']['enable_weather_features']}.pkl"
        interval = config['spatio_temporal']['time_interval_minutes']
        dynamic_path = data_path / f"interval_{interval}" / f"{self.dataset_type}_dynamic_features{feature_suffix}"
        static_path = data_path / f"interval_{interval}" / f"static_features{feature_suffix}"
        targets_path = data_path / f"interval_{interval}" / f"targets_features{feature_suffix}"

        try:
            self.df_dynamic = pd.read_pickle(dynamic_path)
            self.df_static = pd.read_pickle(static_path)
            self.df_targets = pd.read_pickle(targets_path)
            for col in ['has_orders', 'trip_count', 'average_income']:
                if col in self.df_targets.columns:
                    self.df_targets[col] = pd.to_numeric(self.df_targets[col], errors='coerce').fillna(0)
        except FileNotFoundError as e:
            logger.error(f"Could not find required data file: {e}. Aborting.")
            self.valid_sample_coordinates = [] # Set to empty to indicate failure
            return

        # --- 2. Define Attributes and Mappings ---
        # Get all zone IDs from the GeoJSON file to ensure all zones are included
        geojson_path = config['spatio_temporal']['gridding']['geojson']['file_path']
        zone_id_property = config['spatio_temporal']['gridding']['geojson']['zone_id_property']
        all_possible_zones_gdf = gpd.read_file(geojson_path)
        self.all_zone_ids = sorted(all_possible_zones_gdf[zone_id_property].astype(str).unique())
        
        self.num_zones = len(self.all_zone_ids)
        self.zone_to_idx = {zone_id: i for i, zone_id in enumerate(self.all_zone_ids)}
        self.target_cols = ['has_orders', 'trip_count', 'average_income', 'total_income']
        self.df_dynamic['zone_idx'] = self.df_dynamic['zone_id'].map(self.zone_to_idx)
        self.dynamic_feature_cols = [col for col in self.df_dynamic.columns if col not in ['time_slot', 'zone_id', 'zone_idx']]
        self.num_input_features = len(self.dynamic_feature_cols)
        self.num_static_features = len(self.df_static.columns) # Ensure this is defined
        self.config['data']['num_static_features'] = self.num_static_features # Add to config for model access
        self.static_features_tensor = torch.tensor(self.df_static.values, dtype=torch.float32)

        # --- Calculate and store zone centroids (lat/lon for Haversine) ---
        self.zone_lat_lon = self._get_zone_lat_lon(config)

        # --- Pre-compute Haversine distances and bucketize them ---
        distances_km = self._compute_haversine_distances()
        self.haversine_distance_matrix_static = torch.from_numpy(self._bucketize_distances(distances_km)).long()
        logger.info(f"DHKIDataset.__init__ - haversine_distance_matrix_static.shape: {self.haversine_distance_matrix_static.shape}")

        # --- Create Graph for GNN-based baselines ---
        self.edge_index = self._create_graph_from_distances(distances_km)
        self.adjacency_matrix = self._calculate_normalized_adjacency(self.edge_index)

        # --- 3. Create Dense Grid for efficient lookup ---
        logger.info("Creating dense spatio-temporal grid for efficient lookups...")
        all_time_slots = sorted(self.df_dynamic['time_slot'].unique())
        full_grid_index = pd.MultiIndex.from_product([all_time_slots, self.all_zone_ids], names=['time_slot', 'zone_id'])
        df_dynamic_dense = self.df_dynamic.set_index(['time_slot', 'zone_id']).reindex(full_grid_index, fill_value=0)
        df_targets_dense = self.df_targets.set_index(['time_slot', 'zone_id']).reindex(full_grid_index, fill_value=0)
        self.dynamic_features_np = df_dynamic_dense[self.dynamic_feature_cols].values.reshape(len(all_time_slots), self.num_zones, -1)
        self.targets_np = df_targets_dense[self.target_cols].values.astype(np.float32).reshape(len(all_time_slots), self.num_zones, -1)
        logger.info("Dense grid created.")

        # --- 4. Create list of valid sample coordinates (time_idx only) ---
        # Each sample will now represent a full time slice (all zones)
        logger.info("Identifying valid time slices for samples...")
        self.time_to_idx = {t: i for i, t in enumerate(all_time_slots)}
        # We only need the time index, as each sample will contain all zones for that time
        self.valid_time_indices = sorted(self.time_to_idx.values())

        # --- 5. Apply Time Window Filter (Optional) ---
        time_filter_config = self.config.get('data_filtering', {}).get('time_window_filter', {})
        if time_filter_config.get('enable', False):
            start_hour = time_filter_config['start_hour']
            end_hour = time_filter_config['end_hour']
            logging.info(f"Applying time window filter from {start_hour}:00 to {end_hour}:00...")
            
            # Create a boolean mask for the time slots to keep
            time_slots_to_keep = [t for t, i in self.time_to_idx.items() if start_hour <= pd.to_datetime(t).hour <= end_hour]
            indices_to_keep = [self.time_to_idx[t] for t in time_slots_to_keep]
            
            self.valid_time_indices = sorted(list(set(self.valid_time_indices) & set(indices_to_keep)))
            logging.info(f"Filtered down to {len(self.valid_time_indices)} valid time slices.")

        logger.info(f"Found {len(self.valid_time_indices)} valid time slices to generate samples from.")

    def _get_zone_lat_lon(self, config):
        """Helper to load and filter zone geometries to get lat/lon centroids."""
        logger.info("Getting zone lat/lon from GeoJSON for Haversine distance calculation...")
        zone_id_property = config['spatio_temporal']['gridding']['geojson']['zone_id_property']
        gdf = gpd.read_file(config['spatio_temporal']['gridding']['geojson']['file_path'])
        
        # Robustly filter the GeoDataFrame
        gdf[zone_id_property] = gdf[zone_id_property].astype(str)
        
        # Filter gdf to only include zones present in the current dataset split
        filtered_gdf = gdf[gdf[zone_id_property].isin(self.all_zone_ids)].set_index(zone_id_property)

        if filtered_gdf.empty:
            logger.warning(f"No matching zones found between GeoJSON and target data. Zone IDs from data: {self.all_zone_ids[:5]}...")
            return {}

        # Ensure the CRS is geographic for correct centroid calculation in degrees
        if filtered_gdf.crs.is_geographic:
            pass # Already in a geographic CRS (like WGS84)
        else:
            filtered_gdf = filtered_gdf.to_crs(epsg=4326) # Reproject to WGS84 if needed
            
        gdf['centroid'] = gdf.geometry.centroid
        return {zone_id: (point.y, point.x) for zone_id, point in gdf['centroid'].items()}

    def _compute_haversine_distances(self):
        """Computes pairwise Haversine distances if zone coordinates are available."""
        logger.info("Computing pairwise Haversine distances between zones...")
        if not self.zone_lat_lon:
            logger.warning("Zone lat/lon data is empty, cannot compute Haversine distances.")
            # Return a 2D array with shape (0, 0) to avoid downstream errors
            return np.zeros((0, 0))

        # Create a sorted list of zone IDs and corresponding coordinates
        sorted_zone_ids = sorted(self.zone_lat_lon.keys())
        coords = [self.zone_lat_lon[zid] for zid in sorted_zone_ids]
        
        # Use cdist for efficient pairwise distance calculation
        distances = cdist(coords, coords, metric=haversine)
        return distances

    def _bucketize_distances(self, distances_matrix: np.ndarray) -> np.ndarray:
        logger.info("Bucketizing Haversine distances...")
        num_distance_buckets = self.config['model']['num_distance_buckets'] # Assuming this will be in config
        
        # Flatten the distance matrix, remove self-distances (0), and find max
        non_zero_distances = distances_matrix[distances_matrix > 0]
        if len(non_zero_distances) == 0: # Handle case with only one zone or all distances are 0
            return np.zeros_like(distances_matrix, dtype=int)

        max_dist = np.max(non_zero_distances)
        
        # Create bins. The last bin will include max_dist.
        # We want num_distance_buckets bins, so num_distance_buckets + 1 edges.
        bins = np.linspace(0, max_dist, num_distance_buckets + 1)
        
        # Use np.digitize to assign each distance to a bucket
        # np.digitize returns indices starting from 1, so subtract 1 to get 0-indexed buckets
        bucket_indices = np.digitize(distances_matrix, bins) - 1
        
        # Ensure indices are within valid range [0, num_distance_buckets - 1]
        bucket_indices[bucket_indices < 0] = 0
        bucket_indices[bucket_indices >= num_distance_buckets] = num_distance_buckets - 1
        
        return bucket_indices.astype(int)

    def _create_graph_from_distances(self, distances_km: np.ndarray, threshold_km: float = 2.0) -> torch.Tensor:
        """Creates a graph where nodes are connected if they are within a certain distance."""
        logger.info(f"Creating graph from distances with threshold {threshold_km} km...")
        adj_matrix = (distances_km > 0) & (distances_km < threshold_km)
        np.fill_diagonal(adj_matrix, 1) # Add self-loops
        edge_index = torch.tensor(np.array(np.where(adj_matrix)), dtype=torch.long)
        return edge_index

    def _calculate_normalized_adjacency(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Calculates the symmetrically normalized adjacency matrix."""
        logger.info("Calculating normalized adjacency matrix...")
        try:
            from torch_geometric.utils import to_dense_adj, degree
        except ImportError:
            logger.error("torch_geometric is not installed. Please install it to use GNN-based models.")
            return torch.empty(0)

        # Convert edge_index to a dense adjacency matrix
        adj = to_dense_adj(edge_index, max_num_nodes=self.num_zones)[0]
        
        # Calculate degree matrix
        deg = degree(edge_index[0], self.num_zones, dtype=torch.float32)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Symmetrically normalized adjacency matrix
        norm_adj = torch.eye(self.num_zones) + adj # Add self-loops before normalization
        norm_adj = deg_inv_sqrt.view(-1, 1) * norm_adj * deg_inv_sqrt.view(1, -1)
        
        return norm_adj

    def __len__(self):
        return len(self.valid_time_indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Generates a single training sample on-the-fly.
        Each sample contains historical data and targets for ALL zones at a given time slice.
        """
        anchor_time_idx = self.valid_time_indices[index]

        # --- Slice historical dynamic features for ALL zones ---
        start_idx = max(0, anchor_time_idx - self.look_back + 1)
        end_idx = anchor_time_idx + 1
        # Shape: (seq_len, num_zones, num_features)
        historical_data = self.dynamic_features_np[start_idx:end_idx, :, :]
        
        if historical_data.shape[0] < self.look_back:
            padding = np.zeros((self.look_back - historical_data.shape[0], self.num_zones, self.num_input_features), dtype=np.float32)
            historical_data = np.vstack([padding, historical_data])

        # --- Get targets for ALL zones ---
        # Shape: (num_zones, 3) - has_orders, trip_count, average_income
        targets_data = self.targets_np[anchor_time_idx, :, :].copy()
        targets_data[:, 1] = np.log1p(targets_data[:, 1]) # Log transform trip_count
        targets_data[:, 2] = np.log1p(targets_data[:, 2]) # Log transform avg_income
        # targets_data[:, 3] is total_income, no log transform needed for ground truth comparison

        # --- Contrastive Learning Data ---
        # Randomly select an anchor zone for contrastive learning
        anchor_zone_idx = np.random.randint(0, self.num_zones)
        positive_zone_idx = anchor_zone_idx # Positive sample is the anchor itself

        # Select negative samples, ensuring they are distinct from anchor
        negative_zone_indices = np.random.choice(
            [i for i in range(self.num_zones) if i != anchor_zone_idx],
            size=min(self.num_contrastive_negatives, self.num_zones - 1), # Ensure we don't ask for more negatives than available
            replace=False
        )
        
        # If there are no other zones (num_zones=1), or not enough negatives, handle gracefully
        if self.num_zones == 1:
            negative_static_features = torch.empty(0, self.num_static_features, dtype=torch.float32) # Empty tensor
        else:
            negative_static_features = self.static_features_tensor[negative_zone_indices]

        sample = {
            "dynamic_features": torch.from_numpy(historical_data.astype(np.float32)),
            "target": torch.from_numpy(targets_data[:, :3].astype(np.float32)), # Pass only the first 3 cols
            "total_income": torch.from_numpy(targets_data[:, 3].astype(np.float32)), # Pass total_income separately
            "zone_indices": torch.arange(self.num_zones, dtype=torch.long), # All zone indices for this sample
            "static_features": self.static_features_tensor, # All static features for the main model
            "haversine_distance_matrix": self.haversine_distance_matrix_static, # Pass the static matrix
            "static_features_anchor": self.static_features_tensor[anchor_zone_idx],
            "static_features_positive": self.static_features_tensor[positive_zone_idx],
            "static_features_negatives": negative_static_features,
            "time_slot": torch.tensor(self.valid_time_indices[index], dtype=torch.long),
            "adjacency_matrix": self.adjacency_matrix
        }
        
        return sample
