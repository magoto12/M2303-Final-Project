from multiple_linear_work import df_panel


print(df_panel[df_panel['Season'] == 2020][['P/GP', 'S%', 'TOI_float']])
