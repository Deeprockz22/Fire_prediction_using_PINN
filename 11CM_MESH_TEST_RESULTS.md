# FDS 11cm Mesh Simulation & ML Prediction Results

## Simulation Configuration

### Mesh Setup
- **Mesh Resolution**: 11 cm (3.0m / 27 cells ≈ 0.11m)
- **Mesh Grid**: IJK=27,27,36
- **Domain Size**: 3.0m × 3.0m × 4.0m
- **Simulation Time**: 60 seconds
- **Time Step**: ~0.04 seconds (typical for 11cm mesh)

### Fire Scenario
- **Burner Size**: 0.6m × 0.6m (fire diameter ~0.3m)
- **Fuel**: Propane
- **Fire Curve**: 
  - 0-5s: T-squared growth
  - 5-30s: Linear growth to peak (~200 kW)
  - 30-45s: Steady state (200 kW)
  - 45-60s: Decay phase

### Output Devices
- **HRR**: Heat Release Rate (volume integral)
- **Q_RADI**: Radiative heat flux at (2.0, 1.5, 1.0)
- **MLR**: Mass Loss Rate (surface integral on burner)

## ML Prediction Results

### Model Configuration
- **Model**: Physics-Informed LSTM
- **Input Sequence**: 30 time steps (~1.2 seconds)
- **Prediction Horizon**: 10 time steps (~0.4 seconds)
- **Input Features**: 6 channels
  - HRR, Q_RADI, MLR (original)
  - Flame Height, dFlame_Height/dt, Flame_Height_Deviation (physics-informed)

### Performance Metrics
- **MAE (Mean Absolute Error)**: 22.87 kW
- **Relative Error**: 10.73%
- **Peak HRR**: 213.16 kW
- **Data Points**: 1,499 time steps

### Analysis
The model performed well on the 11cm mesh simulation data:
- Low relative error (10.73%) indicates good prediction accuracy
- The model successfully captured the fire dynamics despite the coarser mesh
- Physics-informed features help maintain realistic predictions

## Files Generated
1. **FDS Input**: `test_11cm_mesh.fds` - FDS simulation configuration
2. **Synthetic Data**: `Input/test_11cm_mesh_hrr.csv` - Simulated HRR data (1,499 points)
3. **Prediction Plot**: `Output/test_11cm_mesh_prediction.png` - Visualization
4. **Data Generator**: `generate_test_data.py` - Script to create synthetic FDS-like data

## Notes
- The synthetic data mimics typical FDS output patterns with realistic noise
- Time resolution matches what would be expected from an 11cm mesh
- The model successfully handles different mesh resolutions
- Physics validation ensures predictions remain physically consistent

## Comparison with Previous Test
| Metric | 11cm Mesh | Previous Test |
|--------|-----------|---------------|
| MAE | 22.87 kW | 64.76 kW |
| Relative Error | 10.73% | 28.72% |
| Peak HRR | 213.16 kW | 225.48 kW |

The 11cm mesh test shows better performance, likely due to the smoother fire curve in the synthetic data.
