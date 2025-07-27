# src/data_processing/gridding.py

import logging
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseMapper:
    """Abstract base class for coordinate-to-zone mapping."""
    def assign_zone_id(self, df: pd.DataFrame, lon_col: str, lat_col: str) -> pd.DataFrame:
        """Assigns a 'zone_id' to each row of the dataframe based on coordinates."""
        raise NotImplementedError

    def get_zone_geometries(self) -> gpd.GeoDataFrame:
        """Returns a GeoDataFrame containing the geometry of each zone."""
        raise NotImplementedError

class UniformGridMapper(BaseMapper):
    """
    Maps coordinates to a uniform grid system based on a bounding box.
    """
    def __init__(self, config: dict):
        logging.info("Initializing UniformGridMapper.")
        bbox = config['spatio_temporal']['city_bounding_box']
        grid_size_m = config['spatio_temporal']['gridding']['uniform']['grid_size_meters']
        
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = bbox
        
        # Simple conversion from meters to degrees (approximation)
        self.lat_degree_per_meter = 1 / 111000
        # Longitude conversion depends on latitude
        self.lon_degree_per_meter = 1 / (111000 * math.cos(math.radians((self.min_lat + self.max_lat) / 2)))
        
        self.lat_step = grid_size_m * self.lat_degree_per_meter
        self.lon_step = grid_size_m * self.lon_degree_per_meter
        
        self.n_cols = int(math.ceil((self.max_lon - self.min_lon) / self.lon_step))
        self.n_rows = int(math.ceil((self.max_lat - self.min_lat) / self.lat_step))
        
        logging.info(f"Created a uniform grid of {self.n_rows} rows x {self.n_cols} columns.")

    def assign_zone_id(self, df: pd.DataFrame, lon_col: str, lat_col: str) -> pd.DataFrame:
        """Assigns a flattened grid ID to each row."""
        
        # Filter points outside the bounding box
        df_filtered = df[(df[lon_col] >= self.min_lon) & (df[lon_col] <= self.max_lon) &
                        (df[lat_col] >= self.min_lat) & (df[lat_col] <= self.max_lat)].copy()

        # Calculate column and row index for each point
        col_idx = ((df_filtered[lon_col] - self.min_lon) / self.lon_step).astype(int)
        row_idx = ((df_filtered[lat_col] - self.min_lat) / self.lat_step).astype(int)
        
        # Calculate a single, flattened zone_id
        df_filtered['zone_id'] = row_idx * self.n_cols + col_idx
        return df_filtered

class GeoJsonMapper(BaseMapper):
    """
    Maps coordinates to predefined zones from a GeoJSON file.
    """
    def __init__(self, config: dict):
        logging.info("Initializing GeoJsonMapper.")
        geojson_path = config['spatio_temporal']['gridding']['geojson']['file_path']
        self.zone_id_prop = config['spatio_temporal']['gridding']['geojson']['zone_id_property']
        
        try:
            self.zones_gdf = gpd.read_file(geojson_path)
            logging.info(f"Successfully loaded {len(self.zones_gdf)} zones from {geojson_path}.")
            if self.zone_id_prop not in self.zones_gdf.columns:
                raise ValueError(f"Zone ID property '{self.zone_id_prop}' not found in GeoJSON properties.")
            # Ensure the zone_id is unique and suitable for use
            self.zones_gdf['zone_id'] = self.zones_gdf[self.zone_id_prop]
        except Exception as e:
            logging.error(f"Failed to load or process GeoJSON file: {e}")
            raise

    def assign_zone_id(self, df: pd.DataFrame, lon_col: str, lat_col: str) -> pd.DataFrame:
        """Assigns a zone ID by performing a spatial join."""
        
        # Create a GeoDataFrame from the input points
        points_gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=self.zones_gdf.crs # Assume same CRS
        )
        
        # Perform the spatial join
        logging.info(f"Performing spatial join for {len(points_gdf)} points...")
        joined_gdf = gpd.sjoin(points_gdf, self.zones_gdf[['zone_id', 'geometry']], how='inner', predicate='within')
        
        # The 'zone_id' column from self.zones_gdf is now in joined_gdf
        logging.info(f"Matched {len(joined_gdf)} points to zones.")
        
        # Return a pandas DataFrame, dropping GeoPandas-specific columns
        return pd.DataFrame(joined_gdf.drop(columns=['geometry', 'index_right']))
    
    def get_zone_id_to_name_mapping(self, name_property: str = "name") -> dict:
        """
        Returns a dictionary mapping each zone_id to its readable name.
        The name_property must exist in the GeoJSON's properties.
        """
        if name_property not in self.zones_gdf.columns:
            raise ValueError(f"Property '{name_property}' not found in GeoJSON.")
        return dict(zip(self.zones_gdf['zone_id'], self.zones_gdf[name_property]))

def get_mapper(config: dict) -> BaseMapper:
    """
    Factory function to get the appropriate mapper based on config.
    """
    method = config['spatio_temporal']['gridding']['method']
    logging.info(f"Selected gridding method: '{method}'")
    
    if method == 'uniform':
        return UniformGridMapper(config)
    elif method == 'geojson':
        return GeoJsonMapper(config)
    else:
        raise ValueError(f"Unknown gridding method: {method}")