import pandas as pd
import numpy as np
from functools import reduce

s2020 = pd.DataFrame(pd.read_excel('nhlplaystat_2020-21.xlsx'))
s2021 = pd.DataFrame(pd.read_excel('nhlplaystat_2021-22.xlsx'))
s2022 = pd.DataFrame(pd.read_excel('nhlplaystat_2022-23.xlsx'))
s2023 = pd.DataFrame(pd.read_excel('nhlplaystat_2023-24.xlsx'))
s2024 = pd.DataFrame(pd.read_excel('nhlplaystat_2024-25.xlsx'))




pgpdf = [
    s2020[['Player','P/GP', 'G', 'A']],
    s2021[['Player','P/GP', 'G', 'A']],
    s2022[['Player','P/GP', 'G', 'A']],
    s2023[['Player','P/GP', 'G', 'A']],
    s2024[['Player','P/GP', 'G', 'A']]
]

print(pgpdf)



dfs = []
for year, df in zip(range(2020, 2026), [s2020, s2021, s2022, s2023, s2024]):
    tmp = df[['Player', 'P/GP']].copy()
    tmp.rename(columns={'P/GP': str(year)}, inplace=True)
    dfs.append(tmp)

#df_all = reduce(lambda left, right: pd.merge(left, right, on='Player'), dfs)
df_all = reduce(
    lambda left, right: pd.merge(left, right, on='Player', how='outer'),
    dfs
)
#fill nulls with column minimum, there are 119 players
df_all['2020'] = df_all['2020'].fillna(df_all['2020'].min())
df_all['2021'] = df_all['2021'].fillna(df_all['2021'].min())
df_all['2022'] = df_all['2022'].fillna(df_all['2022'].min())
df_all['2023'] = df_all['2023'].fillna(df_all['2023'].min())
df_all['2024'] = df_all['2024'].fillna(df_all['2024'].min())


#turn df_all into a numpy

y = df_all[['2020', '2021', '2022', '2023', '2024']].to_numpy()



A_T = np.array([
    [1, 1, 1, 1, 1],
    [2020, 2021, 2022, 2023, 2024]
])

ATA = A_T @ A_T.T  #A^T * A

ATA_inv = np.linalg.inv(ATA)
#print(ATA_inv)

ATy = A_T @ y.T

B = ATA_inv @ ATy
y_2025 = B[0, :] + B[1, :] * 2025
y_2026 = B[0, :] + B[1, :] * 2026
y_2027 = B[0, :] + B[1, :] * 2027
y_2028 = B[0, :] + B[1, :] * 2028


df_all['2025'] = y_2025
df_all['2026'] = y_2026
df_all['2027'] = y_2027
df_all['2028'] = y_2028

#print(df_all.sort_values(by='2025', ascending=False))
#print(df_all.loc[df_all['2025'], 'Player'])
top10 = df_all.nlargest(10, "2025")[["Player", "2025"]]
print(df_all)

