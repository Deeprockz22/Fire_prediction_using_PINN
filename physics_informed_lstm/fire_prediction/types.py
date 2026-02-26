from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class ScenarioParams:
    """
    Strongly-typed storage for scenario parameters.
    """
    name: str
    fuel_type: str
    room_size: str # 'small', 'medium', 'large'
    opening_factor: float # 0.0 to 1.0
    wind_speed: float # Normalized 0.0 to 1.0
    fire_size: float # Normalized 0.0 to 1.0
    
    # Physics parameters from FDS file (Ground Truth)
    fds_fuel: Optional[str] = None
    fds_mass_flux: Optional[float] = None
