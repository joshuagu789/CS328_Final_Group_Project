"""
Extracts features and saves processed data
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd

def calc_magnitude(data):
    # Calculate magnitude  
    data = data.copy() # create copy to remove warnings
    data['accel_mag'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2) # absolute accel magnitude
    data['accel_mag'] = data['accel_mag'] - data['accel_mag'].mean() # detrend: "remove gravity"

    return data

def remove_noise(data,sampling_rate):
    # Low pass filter
    cutoff = 5 # Hz
    order = 2
    b, a = butter(order, cutoff/(sampling_rate/2), btype='lowpass')
    data['filtered_accel_mag'] = filtfilt(b, a, data['accel_mag'])

    return data

def add_features(window: pd.DataFrame):
    data = {
        'avg': [window['filtered_accel_mag'].mean()], 
        'max': [window['filtered_accel_mag'].max()],
        'med': [window['filtered_accel_mag'].median()], 
        'min': [window['filtered_accel_mag'].min()],
        'q25': [window['filtered_accel_mag'].quantile(0.25)],
        'q75': [window['filtered_accel_mag'].quantile(0.75)],
        'std': [window['filtered_accel_mag'].std()]
    }
    return pd.DataFrame(data)

# Function to extract windows and features 
def extract_features(data: pd.DataFrame, window_sec, sample_rate):
    feature_dfs = []
    resampled =  data.resample(f'{window_sec}s')
    for timestamp, window in resampled:
        window_with_features = add_features(window.copy())
        window_with_features.insert(0, 'time', timestamp)
        window_with_features['label'] = 1 if window['annotation'].sum() > 4 else 0
        feature_dfs.append(window_with_features)
    return pd.concat(feature_dfs)