import torch
import re
import numpy as np
import logging
from typing import Optional, Dict

from fire_prediction.utils.fds_parser import FDSParser

logger = logging.getLogger(__name__)

class StaticFeatureExtractor:
    """
    Extracts static features from experiment scenario names and FDS files.
    
    Features (Size: 12):
    [0-5] Fuel Type (One-hot: 6 classes)
    [6-8] Room Size (One-hot: 3 classes)
    [9]   Opening Factor (Scalar: 0-1)
    [10]  Wind Speed (Scalar: Normalized 0-1)
    [11]  Fire Size (Scalar: Normalized 0-1)
    """
    
    FUELS = ['N-HEPTANE', 'PROPANE', 'METHANE', 'ACETONE', 'ETHANOL', 'DIESEL']
    ROOMS = ['small', 'medium', 'large']
    
    def __init__(self, data_dir: str = r"D:\FDS\Small_project\fds_scenarios"):
        self.num_fuels = len(self.FUELS)
        self.num_rooms = len(self.ROOMS)
        self.output_dim = self.num_fuels + self.num_rooms + 3
        
        # Initialize Parser
        self.parser = FDSParser(data_dir)
        
    def extract(self, scenario_name: str) -> torch.Tensor:
        """
        Extract features for a given scenario.
        """
        name_lower = scenario_name.lower()
        
        # 1. Parse FDS File (Ground Truth)
        fds_params = self.parser.parse(scenario_name)
        
        # --- FEATURE 1: FUEL TYPE ---
        fuel_vec = [0.0] * self.num_fuels
        
        # Try FDS file first
        detected_fuel = False
        if 'fuel' in fds_params:
            fds_fuel = fds_params['fuel'].upper()
            if fds_fuel in self.FUELS:
                idx = self.FUELS.index(fds_fuel)
                fuel_vec[idx] = 1.0
                detected_fuel = True
        
        # Fallback to filename
        if not detected_fuel:
            for i, fuel in enumerate(self.FUELS):
                # 'n-heptane' vs 'heptane' check
                search_term = fuel.lower().replace('n-', '') 
                if search_term in name_lower:
                    fuel_vec[i] = 1.0
                    detected_fuel = True
                    break
                    
        # Final Fallback (Should typically not happen in valid dataset)
        if not detected_fuel:
            # logger.warning(f"Could not detect fuel for {scenario_name}, defaulting to Propane")
            fuel_vec[1] = 1.0 # Default Propane
            
        # --- FEATURE 2: ROOM SIZE ---
        room_vec = [0.0] * self.num_rooms
        
        # Use filename heuristic (Robust enough for now)
        if 'small' in name_lower: room_vec[0] = 1.0
        elif 'large' in name_lower: room_vec[2] = 1.0
        else: room_vec[1] = 1.0 # Default Medium
        
        # --- FEATURE 3: OPENING FACTOR ---
        opening = 0.5 
        
        # Regex parsing from filename
        if 'closed' in name_lower: opening = 0.0
        elif 'open' in name_lower and 'opening' not in name_lower: opening = 1.0 # "door_open"
        else:
            match = re.search(r'(?:opening|op|door)_?(\d+)', name_lower)
            if match:
                val = int(match.group(1))
                if val > 1: opening = val / 100.0
                else: opening = float(val)
                
        # --- FEATURE 4: WIND SPEED ---
        wind = 0.0
        match = re.search(r'wind_(\d+)ms', name_lower)
        if match:
            wind = float(match.group(1)) / 10.0
            
        # --- FEATURE 5: FIRE SIZE (Intensity) ---
        fire_size = 0.5
        
        # Use Mass Flux from FDS if available (More accurate)
        if 'mass_flux' in fds_params:
            # Normalize mass flux (example range 0.01 to 0.05)
            mf = fds_params['mass_flux']
            # Simple Min-Max scaling based on known dataset range
            fire_size = np.clip((mf - 0.01) / (0.05 - 0.01), 0.0, 1.0)
        else:
            # Fallback to filename
            match = re.search(r'(?:size|sz)_?(\d+)', name_lower) 
            if match:
                 val = int(match.group(1))
                 fire_size = val / 100.0
                 
        # Assemble Feature Vector
        features = []
        features.extend(fuel_vec)
        features.extend(room_vec)
        features.append(opening)
        features.append(wind)
        features.append(fire_size)
        
        return torch.tensor(features, dtype=torch.float32)

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    extractor = StaticFeatureExtractor()
    # Test known file
    name = "R1_n-heptane_medium_op50_sz72"
    feat = extractor.extract(name)
    print(f"Features for {name}:")
    print(feat)
    print(f"Shape: {feat.shape}")
