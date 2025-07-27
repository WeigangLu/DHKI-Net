# src/data_processing/loader.py

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_raw_data(file_path: str, sample_fraction: float = 1.0) -> pd.DataFrame:
    """
    Loads raw data from a CSV file into a pandas DataFrame and optionally samples it.

    Args:
        file_path (str): The path to the CSV file.
        sample_fraction (float): The fraction of data to sample. Defaults to 1.0 (all data).

    Returns:
        pd.DataFrame: The loaded (and possibly sampled) data.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    logging.info(f"Loading raw data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        
        if sample_fraction < 1.0:
            logging.warning(f"Using only a {sample_fraction:.0%} random sample of the data.")
            df = df.sample(frac=sample_fraction, random_state=42) # Use a fixed seed for reproducibility
            logging.info(f"Sampled data contains {len(df)} rows.")
        else:
            logging.info(f"Successfully loaded {len(df)} rows.")
        
        return df
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {file_path}")
        raise