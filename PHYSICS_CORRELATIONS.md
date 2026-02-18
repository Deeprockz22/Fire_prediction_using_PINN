# 🔬 Physics Correlations in Fire Prediction Model

## Overview
This model integrates **multiple fire physics correlations** to improve prediction accuracy beyond pure data-driven approaches. By embedding domain knowledge from fire science, the model makes more physically consistent predictions.

---

## 📚 Implemented Correlations

### 1. **Heskestad Flame Height Correlation** (1984)
**Purpose**: Estimates flame height from heat release rate

**Equation**:
```
L_f = -1.02D + 0.235 Q_c^(2/5)
```

Where:
- `L_f` = Mean flame height (m)
- `Q_c` = Convective HRR (typically 0.7 × Q_total) (kW)
- `D` = Fire diameter (m)

**Features Derived**:
1. **Flame Height** - Direct correlation output
2. **Flame Growth Rate** - dL_f/dt (indicates fire intensification)
3. **Flame Deviation** - L_f - mean(L_f) (fire phase indicator)

**Physical Insight**:
- Flame height grows sublinearly with HRR (power of 2/5)
- Larger diameter fires have shorter flames (spreading effect)
- Valid for pool fires and burner fires (10-1000 kW range)

---

### 2. **McCaffrey Plume Region Classification** (1979)
**Purpose**: Characterizes plume behavior at different heights

**Regions**:
```
z* = z / Q^(2/5)

- Continuous Flame:  z* < 0.08  (region = 0.0)
- Intermittent:      0.08 ≤ z* < 0.20  (region = 0.5)
- Far-field Plume:   z* ≥ 0.20  (region = 1.0)
```

**Feature**:
4. **Plume Region Indicator** - Normalized value (0.0 to 1.0)

**Physical Insight**:
- Continuous: Steady flame, predictable behavior
- Intermittent: Pulsating flames, transitional dynamics
- Far-field: Buoyant plume, reduced thermal hazard
- Helps model understand fire behavior regime

---

### 3. **Thomas Window Flow Correlation** (1963)
**Purpose**: Estimates ventilation flow through openings

**Equation**:
```
m_dot ≈ 0.5 × A_w × sqrt(H_w)  (kg/s)
```

Where:
- `m_dot` = Mass flow rate through opening (kg/s)
- `A_w` = Opening area (m²)
- `H_w` = Opening height (m)

**Feature**:
5. **Ventilation Flow Factor** - Normalized mass flow (0-1 scale)

**Physical Insight**:
- Higher flow = better ventilation = cooler fire
- Flow proportional to area and sqrt(height)
- Critical for compartment fire behavior
- Affects oxygen availability and fire growth

---

### 4. **Buoyancy Power Scaling**
**Purpose**: Fundamental scaling parameter for buoyant plumes

**Equation**:
```
Buoyancy Power ∝ Q^(2/5)
```

**Feature**:
6. **Buoyancy Indicator** - Normalized Q^(2/5) value

**Physical Insight**:
- The 2/5 power law appears in multiple fire correlations
- Represents fundamental buoyancy-driven flow scaling
- Links HRR to plume velocity, temperature rise, and entrainment
- Universal for all buoyant fires

---

## 🧮 Complete Feature Set (6 Channels)

| Channel | Feature | Type | Correlation | Physical Meaning |
|---------|---------|------|-------------|------------------|
| 0 | HRR | Raw | - | Direct measurement |
| 1 | Q_RADI | Raw | - | Radiative component |
| 2 | MLR | Raw | - | Mass loss rate |
| 3 | Flame Height | Physics | Heskestad | Fire size indicator |
| 4 | Flame Growth | Physics | Heskestad deriv. | Fire intensification |
| 5 | Flame Deviation | Physics | Heskestad | Fire phase |
| **NEW** |||
| 3 | Plume Region | Physics | McCaffrey | Behavior regime |
| 4 | Ventilation Flow | Physics | Thomas | Compartment effect |
| 5 | Buoyancy Power | Physics | Scaling law | Fundamental driver |

**Note**: Current model uses 6 channels total. Enhanced features are computed but may require retraining.

---

## 🎯 Why Physics Correlations?

### Benefits:
✅ **Improved Accuracy**: 8.3% reduction in prediction error  
✅ **Physical Consistency**: Predictions obey fire physics laws  
✅ **Generalization**: Better performance on unseen scenarios  
✅ **Interpretability**: Model learns meaningful patterns  
✅ **Regularization**: Physics constraints prevent overfitting  

### Performance (Ablation Study Results):
```
Configuration                    | Test MAE | Improvement
---------------------------------|----------|------------
Baseline (no physics)            | 5.18 kW  | baseline
Physics Loss only                | 5.08 kW  | +1.9%
Features + Loss (Heskestad only) | 4.75 kW  | +8.3%
```

---

## 🔬 How Features Are Used

### Training Phase:
1. **Feature Extraction**: Physics features computed from raw HRR data
2. **Normalization**: All 6 channels standardized (mean=0, std=1)
3. **Physics-Informed Loss**: Model penalized for violations
4. **Validation**: Post-training physics consistency check

### Prediction Phase:
1. **Input**: Last 30 timesteps of HRR data
2. **Auto-compute**: Physics features calculated automatically
3. **Normalize**: Using training statistics
4. **Predict**: LSTM processes all 6 channels
5. **Denormalize**: Convert back to kW units

---

## 📖 References

1. **Heskestad, G.** (1984). "Engineering relations for fire plumes."  
   *Fire Safety Journal*, 7(1), 25-32.

2. **McCaffrey, B. J.** (1979). "Purely buoyant diffusion flames: Some experimental results."  
   *National Bureau of Standards*, NBSIR 79-1910.

3. **Thomas, P. H.** (1963). "The size of flames from natural fires."  
   *Symposium (International) on Combustion*, 9(1), 844-859.

4. **Karlsson, B., & Quintiere, J. G.** (2000). *Enclosure Fire Dynamics*.  
   CRC Press. (Chapter 3: Fire Plumes)

---

## 🚀 Future Extensions

### Potential Additions:
- [ ] Beyler ceiling jet correlation (temperature prediction)
- [ ] Zukoski entrainment correlation (plume width)
- [ ] Delichatsios flame tilt (external wind effects)
- [ ] Flashover criteria (compartment transitions)

### Enhancement Ideas:
- [ ] Dynamic feature selection based on scenario
- [ ] Adaptive physics weight during training
- [ ] Multi-scale physics (local + global)
- [ ] Uncertainty quantification with physics bounds

---

## 💡 For Developers

### Adding New Correlations:
1. Add correlation function to `fire_prediction/utils/physics.py`
2. Update `compute_enhanced_features()` to include new feature
3. Retrain model with expanded feature set
4. Update normalization statistics
5. Document in this file

### Testing Correlations:
```python
from fire_prediction.utils.physics import *

# Test Heskestad
Q = 100  # kW
L_f = heskestad_flame_height(Q, D=0.3)
print(f"Flame height: {L_f:.2f} m")

# Test McCaffrey
region = mccaffrey_plume_region(Q, z=1.2)
print(f"Plume region: {region}")

# Test Thomas
m_dot = thomas_window_flow(Q, A_w=0.9, H_w=1.0)
print(f"Ventilation flow: {m_dot:.2f} kg/s")
```

---

**Last Updated**: 2026-02-18  
**Model Version**: 2.0.0  
**Physics Module**: `fire_prediction/utils/physics.py`
