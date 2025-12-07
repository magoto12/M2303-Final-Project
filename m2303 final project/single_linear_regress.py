import pandas as pd
import numpy as np
from functools import reduce

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

merged = df.pivot_table(
    index='Player',
    columns='Season',
    values=['P/GP', 'G', 'A']
)

merged = merged.fillna(merged.min()) #nulls are now the lowest value in that column
#print(merged)

'''
LEAST SQUARES METHOD: PREDICTING ASSIST COUNT PER PLAYER
This takes in the assist counts from the past 5 years. 

Parameters: 
    season_X : design matrix
    assist_Y : observation matrix
Result: 
    beta_assist: 2 x 111 matrix, contains B_0 and B_1.        
'''
season_X = np.array([[1,2020], [1,2021], [1,2022], [1,2023], [1,2024]]) #A
assist_Y = merged['A'].values #gives us a 111 x 5, so trapose 
assist_Y = assist_Y.T #now we have 5 x 111

beta_assist = np.linalg.inv(season_X.T @ season_X) @ (season_X.T @ assist_Y)

p_s2025_a = beta_assist[0, :] + beta_assist[1, :] * 2025

merged['predict_A_2025'] = p_s2025_a

'''
LEAST SQUARES METHOD FOR TOTAL GOALS
'''
goal_Y = merged['G'].values 
goal_Y = goal_Y.T 

beta_goal = np.linalg.inv(season_X.T @ season_X) @ (season_X.T @ goal_Y)

p_s2025_g = beta_goal[0, :] + beta_goal[1, :] * 2025
merged['predict_G_2025'] = p_s2025_g

'''
LEAST SQUARES METHOD FOR P/GP %
'''
ppg_Y = merged['P/GP'].values
ppg_Y = ppg_Y.T

beta_ppg = np.linalg.inv(season_X.T @ season_X) @ (season_X.T @ ppg_Y)
p_s2025_ppg = beta_ppg[0, :] + beta_ppg[1, :] * 2025
merged['predicted_PPG%'] = p_s2025_ppg

ff=merged

print(merged[['predict_A_2025', 'predict_G_2025', 'predicted_PPG%']] \
    .sort_values(by='predicted_PPG%', ascending=False) \
    .head(10))

#print(merged['2020loc[merged['G'].min])
print(assist_Y.size)