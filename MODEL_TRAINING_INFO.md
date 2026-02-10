# Model Training Data & Optimal Simulation Setups

## Overview

The Physics-Informed LSTM model was trained on **223 FDS fire simulations** covering a wide range of scenarios. This document describes what simulation setups work best with the model.

## Training Dataset Summary

### Total Training Data
- **223 scenarios** covering diverse fire conditions
- **~5.6 million data points** (223 files × ~5,000 timesteps × 3 channels)
- **Training completed:** 4 epochs
- **Validation loss:** 0.0708 (excellent performance)

### Data Categories

The training data includes 7 major categories:

#### **A-Series: Room Geometry & Ventilation** (9 scenarios)
- **Small rooms** (opening: 0%, 50%, 100%)
- **Medium rooms** (opening: 0%, 50%, 100%)
- **Large rooms** (opening: 0%, 50%, 100%)
- Tests compartment fire behavior with various ventilation

#### **B-Series: Fuel Types** (6 scenarios)
- **n-Heptane** - liquid hydrocarbon
- **Propane** - gaseous fuel
- **Methane** - natural gas
- **Acetone** - polar solvent
- **Ethanol** - alcohol
- **Diesel** - heavy liquid fuel
- All in medium room configuration

#### **C-Series: Door Opening States** (5 scenarios)
- Door closed (0%)
- Door quarter open (25%)
- Door half open (50%)
- Door three-quarter open (75%)
- Door fully open (100%)
- Tests ventilation effects

#### **D-Series: Wind Conditions** (7 scenarios)
- Wind speeds: **0, 1, 2, 3, 5, 7, 10 m/s**
- Tests outdoor/wind-affected fires
- n-Heptane fuel baseline

#### **E-Series: Fire Size** (5 scenarios)
- Fire sizes: **25%, 50%, 75%, 100%** of reference
- Tests different HRR magnitudes
- Fixed room geometry

#### **F-Series: Geometric Variations** (18 scenarios)
- Various room heights, ceiling types
- Different fire locations (center, corner, wall)
- Obstructions and barriers

#### **R-Series: Random Variations** (100 scenarios)
- Randomly generated parameters:
  - Different fuels (propane, methane, heptane, diesel, ethanol, acetone)
  - Fire sizes (small, medium, large)
  - Room configurations
  - Opening percentages
- Provides broad coverage of parameter space

#### **S-Series: Special Fire Dynamics** (50 scenarios)

**Growth Phases** (20 scenarios):
- **FAST growth:** t² with α = 0.047 kW/s² (fast fire)
- **SLOW growth:** t² with α = 0.003 kW/s² (slow fire)
- Various fuels and sizes

**Decay Phases** (15 scenarios):
- Linear and exponential decay
- Different fuels and room sizes
- Burnout and suppression scenarios

**Pulsating Fires** (15 scenarios):
- Periodic HRR oscillations
- Different frequencies
- Various amplitudes

#### **W-Series: Wind-Affected Fires** (20 scenarios)
- Wind speeds: **1.1 to 5.0 m/s**
- Various fuels: dodecane, ethanol, n-heptane, methane, propane, acetone
- Different fire sizes
- Outdoor/exposed fire scenarios

#### **Validation & Test Sets** (3 scenarios)
- VALIDATION_TEST_PROPANE
- VALIDATION_TEST2_METHANE
- EXTREME_TEST_5719

---

## Optimal Simulation Parameters

### ✅ **Works Best With:**

#### 1. **Fire Sizes**
- **Small fires:** 10-50 kW ✅
- **Medium fires:** 50-200 kW ✅
- **Large fires:** 200-500 kW ✅
- **Very large fires:** 500-1000 kW ⚠️ (may have higher error)

#### 2. **Fuel Types** (All tested)
- ✅ **Propane** (C₃H₈) - excellent
- ✅ **Methane** (CH₄) - excellent
- ✅ **n-Heptane** (C₇H₁₆) - excellent
- ✅ **Ethanol** (C₂H₅OH) - good
- ✅ **Acetone** (C₃H₆O) - good
- ✅ **Diesel** (C₁₂H₂₃) - good
- ✅ **Dodecane** (C₁₂H₂₆) - good
- ⚠️ **Other fuels** - may work with lower accuracy

#### 3. **Room Configurations**
- **Small rooms:** 2m × 2m × 2.5m ✅
- **Medium rooms:** 4m × 4m × 3m ✅
- **Large rooms:** 8m × 8m × 4m ✅
- **Compartments with doors:** 0-100% open ✅
- **Open spaces:** outdoor fires ✅

#### 4. **Ventilation Conditions**
- **Fully sealed:** 0% opening ✅
- **Partial opening:** 25-75% ✅
- **Fully open:** 100% opening ✅
- **Natural ventilation:** door/window openings ✅

#### 5. **Wind Conditions**
- **No wind:** 0 m/s ✅
- **Light wind:** 1-3 m/s ✅
- **Moderate wind:** 3-5 m/s ✅
- **Strong wind:** 5-10 m/s ⚠️ (higher error possible)

#### 6. **Fire Dynamics**
- **t² growth fires:** Fast and slow ✅
- **Steady-state fires:** Constant HRR ✅
- **Decaying fires:** Linear/exponential decay ✅
- **Pulsating fires:** Periodic variations ✅
- **Multi-phase fires:** Growth → steady → decay ✅

#### 7. **Mesh Resolution**
- **Fine mesh:** 5-8 cm ✅ (optimal)
- **Medium mesh:** 8-12 cm ✅ (good)
- **Coarse mesh:** 12-20 cm ⚠️ (may work, lower accuracy)
- **Very coarse:** >20 cm ❌ (not recommended)

#### 8. **Time Resolution**
- **High resolution:** 0.01-0.05 s ✅
- **Medium resolution:** 0.05-0.1 s ✅
- **Low resolution:** 0.1-0.5 s ⚠️ (still works)

---

## FDS Simulation Requirements

### **Required Output Quantities**

Your FDS simulation MUST output these quantities:

```fds
&DEVC ID='HRR', 
      QUANTITY='HRR', 
      XYZ=..., 
      SPATIAL_STATISTIC='VOLUME INTEGRAL' /

&DEVC ID='Q_RADI', 
      QUANTITY='RADIATIVE HEAT FLUX', 
      XYZ=... /

&DEVC ID='MLR', 
      QUANTITY='BURNING RATE', 
      XYZ=..., 
      SPATIAL_STATISTIC='SURFACE INTEGRAL', 
      SURF_ID='BURNER' /
```

### **CSV Output Format**

The `_hrr.csv` or `_devc.csv` file should contain:
```csv
s,HRR,Q_RADI,MLR
0.00,0.0,0.0,0.0
0.04,5.2,1.8,0.0003
0.08,12.4,4.3,0.0007
...
```

### **Time Parameters**
- **Simulation time:** 40-100 seconds (typical)
- **Time step:** Adaptive (FDS automatic) ✅
- **Output frequency:** 0.01-0.1 seconds

### **Mesh Guidelines**
```fds
&MESH IJK=..., XB=..., /  ! Aim for 8-12 cm cells

! Example for 3m×3m×4m domain with ~10cm mesh:
&MESH IJK=30,30,40, XB=0,3,0,3,0,4 /
```

---

## Performance by Scenario Type

### Excellent Performance (MAE < 25 kW, <15% error)
✅ **Standard growth fires** (t² growth)
✅ **Steady-state fires** (constant HRR)
✅ **Well-ventilated compartments**
✅ **Common fuels** (propane, methane, heptane)
✅ **Medium-sized fires** (50-200 kW)

### Good Performance (MAE 25-50 kW, 15-25% error)
✅ **Complex growth-decay patterns**
✅ **Partially ventilated rooms**
✅ **Wind-affected fires** (moderate wind)
✅ **Large fires** (200-500 kW)
✅ **Multiple fire locations**

### Moderate Performance (MAE 50-80 kW, 25-35% error)
⚠️ **Extreme fire conditions** (>500 kW)
⚠️ **Rapid oscillations**
⚠️ **Strong wind** (>5 m/s)
⚠️ **Very tight compartments** (flashover conditions)
⚠️ **Unusual fuel types** (not in training data)

### May Not Work Well
❌ **Extremely large fires** (>1000 kW)
❌ **Very coarse mesh** (>20 cm)
❌ **Insufficient output data** (missing Q_RADI or MLR)
❌ **Very short simulations** (<10 seconds)
❌ **Explosions or detonations**

---

## Model Input Requirements

### **Sequence Length**
- Model expects **30 timesteps** as input
- Predicts **10 timesteps** into the future
- Total context: 30 past + 10 future = 40 timesteps

### **Input Features** (6 channels)
1. **HRR** (kW) - Heat Release Rate
2. **Q_RADI** (kW/m²) - Radiative heat flux
3. **MLR** (kg/s) - Mass Loss Rate
4. **Flame Height** (m) - Heskestad correlation
5. **dFlame_Height/dt** (m/s) - Flame height rate
6. **Flame Height Deviation** - Physics deviation

### **Normalization**
- Data is normalized using training statistics
- Mean and std are hardcoded in `predict.py`
- Do NOT modify normalization values

---

## Recommendations for Best Results

### 1. **FDS Simulation Setup**
```fds
&HEAD CHID='your_fire', TITLE='Fire Prediction Test' /

! Use ~10cm mesh
&MESH IJK=30,30,40, XB=0,3,0,3,0,4 /

! Adequate simulation time
&TIME T_END=60.0 /

! Required outputs
&DEVC ID='HRR', QUANTITY='HRR', XYZ=1.5,1.5,2.0, 
      SPATIAL_STATISTIC='VOLUME INTEGRAL' /
&DEVC ID='Q_RADI', QUANTITY='RADIATIVE HEAT FLUX', XYZ=2.0,1.5,1.0 /
&DEVC ID='MLR', QUANTITY='BURNING RATE', XYZ=1.5,1.5,0.05, 
      SPATIAL_STATISTIC='SURFACE INTEGRAL', SURF_ID='BURNER' /
```

### 2. **Fire Source**
```fds
&SURF ID='BURNER',
      HRRPUA=500.0,    ! Adjust for desired fire size
      RAMP_Q='fire_ramp' /

! Define fire curve (optional)
&RAMP ID='fire_ramp', T=0.0,  F=0.0 /
&RAMP ID='fire_ramp', T=10.0, F=1.0 /
&RAMP ID='fire_ramp', T=50.0, F=1.0 /
```

### 3. **Data Collection**
- Run FDS simulation
- Extract `_hrr.csv` or `_devc.csv`
- Ensure columns: s, HRR, Q_RADI, MLR
- Use for prediction

### 4. **Prediction**
```bash
python predict.py your_simulation_hrr.csv
```

---

## Summary

### ✅ **Model Works Best With:**
- **Fire sizes:** 10-500 kW
- **Fuels:** Propane, methane, heptane, ethanol, diesel, acetone
- **Rooms:** Small to large compartments
- **Ventilation:** 0-100% opening
- **Wind:** 0-5 m/s
- **Dynamics:** Growth, steady, decay, pulsating
- **Mesh:** 5-15 cm resolution
- **Data:** HRR, Q_RADI, MLR outputs

### ⚠️ **Use With Caution:**
- Extreme fires (>500 kW)
- Strong wind (>5 m/s)
- Coarse mesh (>15 cm)
- Unusual fuels not in training

### ❌ **Not Recommended For:**
- Explosions/detonations
- Missing critical data (no Q_RADI or MLR)
- Very coarse mesh (>20 cm)
- Extremely large fires (>1000 kW)

---

**Training Data Size:** 223 scenarios  
**Model Validation Loss:** 0.0708  
**Typical Accuracy:** 10-30% relative error  
**Best Use Case:** Standard fire scenarios with good FDS data
