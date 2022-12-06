import pandas as pd
import numpy as np
from IPython.core.display_functions import display

df = pd.read_csv('mike_data3.csv')

animals = df['animal'].unique()

bf_devs = []
for anim in animals:
    df_bf = df.loc[df['animal'] == anim]
    bfs = df_bf['bf'].values.tolist()
    avg = df_bf['bf'].mean()
    bfdevs = bfs - avg
    for val in bfdevs:
        bf_devs.append(val)

df['bf_deviation'] = bf_devs

df.to_csv('mike_data_bfdev.csv')
