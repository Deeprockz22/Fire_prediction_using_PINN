# 🔥 Fire Prediction Using Physics-Informed Neural Networks

This repository contains two complementary approaches for machine learning-based fire dynamics prediction, both validated on comprehensive Fire Dynamics Simulator (FDS) datasets.

---

## 📁 Repository Structure

```
Fire_prediction_using_PINN/
├── physics_informed_lstm/     ← Physics-Informed LSTM (Main Method)
│   ├── fire_prediction/       ← Core package
│   ├── training_data/         ← 221 FDS scenarios
│   ├── checkpoints/           ← Trained models
│   └── docs/                  ← Documentation
│
└── koopman_operator/          ← Koopman Operator Approach
    ├── models/                ← Koopman model implementations
    ├── data/                  ← Data processing
    ├── checkpoints/           ← Trained models
    └── runs/                  ← Training logs
```

---

## 🎯 Project 1: Physics-Informed LSTM (Primary)

### **Status**: ✅ Production-Ready, Paper Under Review

**Heat Release Rate (HRR) prediction** using physics-informed LSTM with Heskestad fire plume correlations.

### **Key Results** (221 FDS Scenarios):
- **Test MAE**: 3.28 kW (2.52% relative error)
- **Improvement**: 36.8% over baseline (5.18 kW → 3.28 kW)
- **Dataset**: 221 scenarios, 57,033 time-sequence samples
- **Speedup**: ~15,000× faster than FDS (hours → <1 second)
- **Paper**: Under review at *Fire Safety Journal*

### **Physics Integration**:
1. **Heskestad Correlations**: Flame height, growth rate, deviation features
2. **Physics-Constrained Loss**: MSE + physics violation penalties
3. **Post-Prediction Validation**: Ensures physical plausibility

### **Quick Start**:
```bash
cd physics_informed_lstm
python fire_predict.py --input your_scenario.csv --output prediction.png
```

### **Dataset**:
- **221 FDS scenarios**: 6 fuels × 3 room sizes × diverse conditions
- **Fuel types**: Propane, Methane, N-Heptane, Ethanol, Acetone, Diesel
- **Fire behaviors**: Growth, decay, pulsating, steady-state, wind-affected
- **Training**: 154 scenarios (39,933 samples)
- **Validation**: 33 scenarios (8,500 samples)
- **Test**: 34 scenarios (8,600 samples)

### **Publications**:
- 📄 Paper submitted to *Fire Safety Journal* (under review)
- 🚀 PhysicsNeMo contribution proposal prepared

---

## 🎯 Project 2: Koopman Operator Approach (Experimental)

### **Status**: 🔬 Research/Experimental

**Dynamic system modeling** using Koopman operator theory for fire dynamics.

### **Approach**:
- Learns linear representations of nonlinear fire dynamics
- Uses eigenfunction decomposition for prediction
- Explores spectral methods for long-term forecasting

### **Key Features**:
- Koopman operator with LSTM encoder
- Eigenvalue analysis of fire dynamics
- Comparison with Physics-Informed LSTM

### **Quick Start**:
```bash
cd koopman_operator
python train.py --data ../physics_informed_lstm/training_data
```

---

## 🚀 Installation

### **Requirements**:
```bash
# Create environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision pytorch-lightning
pip install pandas numpy matplotlib scikit-learn
pip install tensorboard
```

### **Verify Installation**:
```bash
cd physics_informed_lstm
python fire_predict.py --help
```

---

## 📊 Performance Comparison

| Method | Test MAE | Improvement | Inference Time | Status |
|--------|----------|-------------|----------------|--------|
| **FDS (Ground Truth)** | 0 kW | --- | 4-8 hours | Baseline |
| **Baseline LSTM** | 5.18 kW | --- | <1 second | Baseline ML |
| **Physics-Informed LSTM** | **3.28 kW** | **36.8%** | **<1 second** | ✅ **Production** |
| **Koopman Operator** | ~4.5 kW | ~13% | <1 second | 🔬 Experimental |

---

## 📖 Documentation

### **Physics-Informed LSTM**:
- `physics_informed_lstm/docs/README.md` - Complete guide
- `physics_informed_lstm/docs/START_HERE.txt` - Quick start
- `physics_informed_lstm/docs/CORRELATIONS_QUICK_REFERENCE.txt` - Physics equations

### **Koopman Operator**:
- `koopman_operator/README.md` - Method overview
- `koopman_operator/docs/` - Implementation details

---

## 🎓 Citation

If you use this work, please cite:

```bibtex
@article{physics_informed_fire_2026,
  title={Physics-Informed Deep Learning for Real-Time Fire Heat Release Rate Prediction: 
         Integrating Heskestad Correlations with LSTM Networks},
  author={[Authors]},
  journal={Fire Safety Journal},
  year={2026},
  status={Under Review}
}
```

---

## 🤝 Contributing to PhysicsNeMo

This work is being proposed as a contribution to **NVIDIA PhysicsNeMo** - the first fire/combustion module for the framework.

**Proposal Status**: 📝 Prepared and ready for submission

**Why This Matters**:
- First fire dynamics module in PhysicsNeMo
- 221-scenario professional-scale validation
- 36.8% improvement over baseline
- Real-time inference capability
- Extensible to FNO/MeshGraphNet architectures

---

## 📞 Contact & Support

- **GitHub Issues**: Report bugs or request features
- **Paper**: Under review at *Fire Safety Journal*
- **PhysicsNeMo Proposal**: Available in documentation

---

## 📜 License

Apache 2.0 License (to match PhysicsNeMo)

---

## 🙏 Acknowledgments

- **FDS Development Team**: NIST for Fire Dynamics Simulator
- **Heskestad's Work**: Foundational fire plume correlations
- **NVIDIA PhysicsNeMo**: Target framework for contribution
- **Fire Safety Research Community**: For validation and feedback

---

## 🔥 Quick Links

- 🌐 **Live Demo**: [Coming Soon]
- 📊 **Results Dashboard**: See `physics_informed_lstm/logs/`
- 🎯 **PhysicsNeMo**: https://github.com/NVIDIA/physicsnemo
- 📄 **Paper**: [Link when published]

---

**⭐ Star this repository if you find it useful!**

**🚀 221 Scenarios • 36.8% Improvement • Real-Time Prediction • Physics-Informed** 🔥
