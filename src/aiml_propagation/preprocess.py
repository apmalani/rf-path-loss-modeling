import pandas as pd
import numpy as np
import os

# replace with the directory of the new data if you want to preprocess the new data and load it into the model (a few minutes, max)
# keep the same and add the new data to this file if you want to retrain the model (overnight process)
dir = input()
dfs = []

for f in os.listdir(dir):
    p = os.path.join(dir, f)
    if (os.path.isdir(p)):
        for fn in os.listdir(p):
            pn = os.path.join(p,fn)
            df = pd.read_csv(pn)
            df['city'] = f[7:-8]
            dfs.append(df)
    else:
        df = pd.read_csv(p)
        dfs.append(df)

data = pd.concat(dfs, ignore_index = True)

data = data.drop(data[data['Route Type'] == 'LOS'].index)
data = data.drop(data[data['Route Type'] == 'NLOS'].index)

data['gain_error'] = -1 * data['Total Gain'] - data['measured_values']

data['gamma_ideal'] = np.where(data['Route Type'] == 'ONE_TURN', 
                            (20 * np.log10(data['Diffraction C']) + data['gain_error']),
                            (20 * np.log10(data['Diffraction C']) + data['gain_error']/2)
                        )

data = data.drop(data[~(data['Hor Gain'] > data['Vert Gain'] + 3)].index)

rmv = [ 
    'Tx Latitude',
    'Tx Longitude',
    'Tx Ant Hgt',
    'Rx Latitude',
    'Rx Longitude',
    'Rx Ant Hgt',
    'DiffOption',
    'Diffraction C',
    'Bld Reflect C',
    'erg',
    'conductivityg',
    'erw',
    'conductivityw',
    'v_main',
    'hv_main',
    'dv_main',
    'v_1',
    'hv_1',
    'dv_1',
    'v_2',
    'hv_2',
    'dv_2',
    'v_last',
    'hv_last',
    'dv_last',
    'n_e',
    'rxAntHgtAdj',
    'Hor Gain',
    'Vert Gain',
    'Total Gain',
    'measured_values',
    'model',
    'gain_error',
    'ID',
    'Edge Total',
    '2-T Avg Ver BD',
    '2-T Avg Hor BD',
    'Avg Tree Hgt',
    '1-T DDist Rx',
    'TX Ant Hgt AGL',
    '1-T DDist Tx',
    'Route Type',
    'RX Ant Hgt AGL'
]

data = data.drop(rmv, axis = 1)

data.to_csv('src/aiml_propagation/combined_output_data.csv', index = False)
