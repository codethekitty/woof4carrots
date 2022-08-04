import pandas as pd
import numpy as np
from IPython.core.display_functions import display

df = pd.read_csv('test_set_20.csv')

animals = df['animal'].unique()

bf_devs = []
for anim in animals:
    df_bf = df.loc[df['animal'] == anim]
    bfs = df_bf['bf'].values.tolist()
    avg = df_bf['bf'].mean()
    df_bf['bf_deviation'] = bfs - avg
    vals = df_bf['bf_deviation'].values.tolist()
    for val in vals:
        bf_devs.append(val)

df['bf_deviation'] = bf_devs

df.to_csv('test_set_21.csv')
