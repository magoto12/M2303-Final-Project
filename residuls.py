import numpy as np
import pandas as pd
from single_linear_regress import merged
from multiple_linear_work import newbar
from single_linear_regress import ppg_Y
from single_linear_regress import season_X
from single_linear_regress import beta_ppg
#print(merged)

#Y_hat = X * C

#For Model P
y_hats = season_X @ beta_ppg

Es = ppg_Y - y_hats

SSEs = np.sum(Es**2)
Ybars = ppg_Y.mean(axis=0)
TSSs = np.sum((ppg_Y - Ybars)**2)
r_sqs = 1 - SSEs / TSSs

#FOR Model Q
ym = merged['predicted_PPG%'].values
ym_hat = newbar['Predicted_PPG'].values
Em = ym - ym_hat
 
SSEm = np.sum(Em**2)
Ybarm = ym.mean(axis=0)
TSSm = np.sum((ym - Ybarm)**2)

r_sqm = 1 - SSEm / TSSm

print(r_sqm)
print(r_sqs)



