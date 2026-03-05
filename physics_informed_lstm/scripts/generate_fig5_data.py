import torch
import numpy as np
import pandas as pd
from fire_prediction.models.train_physics_full import PhysicsInformedLSTM

model = PhysicsInformedLSTM.load_from_checkpoint('checkpoints/last-v3.ckpt')
model.eval()

df = pd.read_csv(r'd:\FDS\Small_project\ml_data\S10_GROWTH_FAST_ETHANOL_large_timeseries.csv')
hrr = df['HRR'].values if 'HRR' in df.columns else df['hrr'].values

hist = hrr[30:60]
fut = hrr[60:70]
mean, std = np.mean(hist), np.std(hist)+1e-6
hist_n = (hist-mean)/std

x = torch.zeros((1, 30, 9)).to(model.device)
x[0, :, 0] = torch.tensor(hist_n).to(model.device)
with torch.no_grad():
    pred_n = model(x)
pred = (pred_n[0, :, 0].cpu().numpy() * std) + mean

true_arr = np.concatenate([hist, fut])
pred_arr = np.concatenate([hist, pred])

print("true_hrr = np.array(" + str(list(np.round(true_arr, 2))) + ")")
print("physics_pred = np.array(" + str(list(np.round(pred_arr, 2))) + ")")
