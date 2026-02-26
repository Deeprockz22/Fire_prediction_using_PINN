import re
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from fire_prediction.types import ScenarioParams

logger = logging.getLogger(__name__)

class FDSParser:
    """
    Robust parser for FDS input files (.fds).
    Extracts authoritative physics parameters directly from the simulation definition.
    """
    
    def __init__(self, scenarios_dir: str):
        self.scenarios_dir = Path(scenarios_dir)
        
    def parse(self, scenario_name: str) -> Dict[str, Any]:
        """
        Parse the .fds file for a given scenario.
        Returns a dictionary of extracted parameters.
        """
        # Check deployment structures first
        base_name = scenario_name.replace('_hrr', '')
        # Try both the original nested structure and the flat training_data/ structure
        fds_path_nested = self.scenarios_dir / base_name / f"{base_name}.fds"
        fds_path_flat = self.scenarios_dir.parent / "training_data" / f"{base_name}.fds"
        fds_path_input = self.scenarios_dir.parent / "Input" / f"{base_name}.fds"
        
        fds_path = None
        for p in [fds_path_nested, fds_path_flat, fds_path_input]:
            if p.exists():
                fds_path = p
                break
                
        if not fds_path:
            logger.debug(f"FDS file not found for: {base_name}. Extracting physics visually from filename string instead...")
            return {}
            
        params = {}
        
        try:
            content = fds_path.read_text(encoding='utf-8')
            
            # 1. Extract Fuel Type
            # Pattern: &REAC ... FUEL='N-HEPTANE' ... /
            fuel_match = re.search(r"&REAC\s+.*?FUEL\s*=\s*'([^']+)'", content, re.IGNORECASE)
            if fuel_match:
                params['fuel'] = fuel_match.group(1).upper()
                
            # 2. Extract Mass Flux (Fire Intensity)
            # Pattern: MASS_FLUX(1)=0.0244
            flux_match = re.search(r"MASS_FLUX(?:\(\d+\))?\s*=\s*([\d\.]+)", content, re.IGNORECASE)
            if flux_match:
                params['mass_flux'] = float(flux_match.group(1))
                
            # 3. Extract Mesh Bounds (Room Size indicator)
            # Pattern: &MESH ... XB=-1.5,1.5,-1.5,1.5,0.0,2.4
            mesh_match = re.search(r"&MESH\s+.*?XB\s*=\s*([^/]+)", content, re.IGNORECASE)
            if mesh_match:
                coords = [float(x) for x in mesh_match.group(1).split(',')]
                # Calculate volume or footprint approx
                # x_min, x_max, y_min, y_max, z_min, z_max
                if len(coords) >= 6:
                    x_span = coords[1] - coords[0]
                    y_span = coords[3] - coords[2]
                    z_span = coords[5] - coords[4]
                    params['domain_volume'] = x_span * y_span * z_span
                    
        except Exception as e:
            logger.error(f"Error parsing FDS file {fds_path}: {e}")
            
        return params

