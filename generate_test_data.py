"""
Generate synthetic FDS-like data for 11cm mesh simulation
"""
import numpy as np
import pandas as pd

# Time parameters
dt = 0.04  # 40ms time step (typical for 11cm mesh)
t_end = 60.0
time = np.arange(0, t_end, dt)

# Fire scenario: growth, steady, decay
def fire_curve(t):
    if t < 5:
        return 50 * (t/5)**2  # t-squared growth
    elif t < 30:
        return 50 + 150 * (t-5)/25  # linear growth to peak
    elif t < 45:
        return 200  # steady state
    elif t < 60:
        return 200 - 150 * (t-45)/15  # decay
    else:
        return 50

# Generate HRR with some noise
np.random.seed(42)
hrr = np.array([fire_curve(t) for t in time])
hrr += np.random.normal(0, 5, len(time))  # Add noise
hrr = np.maximum(hrr, 0)  # No negative values

# Q_RADI (approximately 30-40% of HRR)
q_radi = 0.35 * hrr + np.random.normal(0, 2, len(time))
q_radi = np.maximum(q_radi, 0)

# MLR (Mass Loss Rate) - related to HRR
# Typical effective heat of combustion ~20 MJ/kg
mlr = (hrr / 20000) + np.random.normal(0, 0.002, len(time))
mlr = np.maximum(mlr, 0)

# Create DataFrame
df = pd.DataFrame({
    's': time,
    'HRR': hrr,
    'Q_RADI': q_radi,
    'MLR': mlr
})

# Save to CSV
output_file = 'Input/test_11cm_mesh_hrr.csv'
df.to_csv(output_file, index=False)
print(f"Generated {len(df)} data points")
print(f"Time range: {time[0]:.2f}s - {time[-1]:.2f}s")
print(f"Peak HRR: {hrr.max():.2f} kW")
print(f"Saved to: {output_file}")
