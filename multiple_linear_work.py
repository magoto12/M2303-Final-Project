import pandas as pd
import numpy as np
from functools import reduce
from single_linear_regress import ff


s2020 = pd.DataFrame(pd.read_excel('nhlplaystat_2020-21.xlsx'))
s2021 = pd.DataFrame(pd.read_excel('nhlplaystat_2021-22.xlsx'))
s2022 = pd.DataFrame(pd.read_excel('nhlplaystat_2022-23.xlsx'))
s2023 = pd.DataFrame(pd.read_excel('nhlplaystat_2023-24.xlsx'))
s2024 = pd.DataFrame(pd.read_excel('nhlplaystat_2024-25.xlsx'))

s2020['Season'] = 2020
s2021['Season'] = 2021
s2022['Season'] = 2022
s2023['Season'] = 2023
s2024['Season'] = 2024

df = pd.concat([s2020, s2021, s2022, s2023, s2024], ignore_index=True)

newbar = df.groupby('Player')[['P/GP', 'S%']].mean()
predicted_y = ff['predicted_PPG%']
newbar.insert(0, 'Intercept', 1)


'''
MULTIPLE LINEAR REGRESSION
'''
mlr_X = newbar.values
mlr_Y = predicted_y.values
mlr_Y = mlr_Y.T

yoda = np.linalg.inv(mlr_X.T @ mlr_X) @ mlr_X.T @ mlr_Y

y_hat = mlr_X @ yoda 

newbar['Predicted_PPG'] = y_hat

#print(newbar['Predicted_PPG'].sort_values(ascending=False).head(10))
print(predicted_y)