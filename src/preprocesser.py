"""
Extracts features and saves processed data
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
import matplotlib.pyplot as plt

import glob
import os

import evaluater
from abc import ABC, abstractmethod

class DataFrameBuilder(ABC):
    """
    Abstract Fluent Builder for manipulating pandas DataFrames
    """

    def __init__(self, initial_dataframe: pd.DataFrame = pd.DataFrame(), sample_rate: int = 100):
        if type(self) == DataFrameBuilder:
            raise TypeError("Cannot instantiate class  directly")

        self.dataframe = initial_dataframe
        self.sample_rate = sample_rate

    def finish_build(self):
        return self.dataframe

class CSV_Builder(DataFrameBuilder):
    """
    Fluent Builder for initial gathering of CSV files and operations done on them (such as trimming first and last second of data)
    """
    def __init__(self, initial_dataframe: pd.DataFrame = None, sample_rate: int = 100):
        super().__init__(initial_dataframe, sample_rate)

    def add_csv_files_in_directory(self, relative_path: str, keyword: str, trim: bool):
        """
        All this does is collect csv files into one big dataframe, no processing is done apart from potential trimming
        keyword: specific word to look in file names when selecting files
        trim: whether to remove first and last few seconds of data from each csv
        """
        all_data = pd.DataFrame()

        all_dfs = []
        #fns = glob.glob(f"data/Activities/*/*.csv")
        #fns = glob.glob(f"{root}/Activities/*/*.csv")
        fns = glob.glob(f"{relative_path}/**/*{keyword}*.csv", recursive=True)
        # x = f"{relative_path}/*.csv"
        # fns = glob.glob(x, recursive=True)
        # fns = glob.glob(path1, recursive=True)

        for fn in fns:
            df = pd.read_csv(fn)
            print("file name is " + str(fn))
            if trim:
                df = df.iloc[3 * self.sample_rate:] # remove first 3 seconds
                df = df.iloc[:-3 * self.sample_rate] # remove last 3 seconds
            all_dfs.append(df)
        
        # print("concating: " + str(all_dfs))
        data = pd.concat(all_dfs)
        all_data = pd.concat([all_data, data])

        # all_data.to_csv(out_path, index=False)

        self.dataframe = pd.concat([self.dataframe, all_data])

        return self
        
    def finish_build(self):
        return self.dataframe

class Preprocessor(DataFrameBuilder):
    """
    Fluent Builder for initial preprocessing of DataFrames (magnitude, removing noise)
    """
    def __init__(self, initial_dataframe: pd.DataFrame, sample_rate: int):
        super().__init__(initial_dataframe, sample_rate)
        # self.dataframe = initial_dataframe
        # self.sample_rate = sample_rate

    def calc_magnitude(self, output_name: str, remove_gravity = True):
        """
        Assumes dataframe has x, y, and z column, calculates magnitude of these 3 coordinates
        """ 
        data = self.dataframe.copy() # create copy to remove warnings

        data[output_name] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2) # absolute accel magnitude
        # print("before detrend: " + str(data[output_name].mean()))
        if remove_gravity:
            data[output_name] = data[output_name] - data[output_name].mean() # detrend: "remove gravity"
        # print("after detrend: " + str(data[output_name].mean()))

        return Preprocessor(data, self.sample_rate)

    def remove_noise(self, column_name: str, output_name: str, order: int = 2, cutoff: int = 5):
        """
        Remove noise of specified column by frequency
        """
        # Low pass filter
        # cutoff = 5 # Hz
        # order = 2
        b, a = butter(order, cutoff/(self.sample_rate/2), btype='lowpass')

        data = self.dataframe.copy()
        data[output_name] = filtfilt(b, a, self.dataframe[column_name])

        return Preprocessor(data, self.sample_rate)

    def to_date_time(self):
        """
        Assumes current dataframe has 'time' column 
        """
        self.dataframe['timestamp'] = pd.to_datetime(self.dataframe['time'])
        self.dataframe = self.dataframe.set_index('timestamp')
        return Preprocessor(self.dataframe, self.sample_rate)

class FeatureExtractor(DataFrameBuilder):
    """
    Fluent Builder for window resampling and feature extraction of DataFrames,
    outputs dataframe in following format: first column is 'time' using data time, last column is 'activity' which is string,
    all other columns are features
    """

    # def __init__(self, initial_dataframe: pd.DataFrame, sample_rate: int, window_sec: int, is_resampled: bool = False):
    def __init__(self, initial_dataframe: pd.DataFrame, sample_rate: int, window_sec: int):
        super().__init__(initial_dataframe, sample_rate)
        self.window_sec = window_sec
        self.resampled = self.dataframe.resample(f'{self.window_sec}s')
        self.features: pd.DataFrame = pd.DataFrame()

    def extract_basic_features(self, column_name: str, activity: str):
        """
        adds following columns: mean, med (median), std (standard deviation), variance using the specified column name, other features include min, max, quartiles
        also labels windows with specified activity name
        """
        # data has filtered acceleration magnitude and annotated step locations
        feature_dfs = []
        # resampled =  self.dataframe.resample(f'{self.window_sec}s')
        for timestamp, window in self.resampled:
            window_with_features = self.__add_features(window.copy(), column_name)
            window_with_features['activity'] = activity
            window_with_features.insert(0, 'time', timestamp)
            # window_with_features['label'] = 1 if window['annotation'].sum() > 4 else 0
            feature_dfs.append(window_with_features)

        new_features = pd.concat(feature_dfs)
        # self.features = self.features.join(new_features) 
        self.features = pd.concat([self.features, new_features], axis=1, ignore_index=False)
        self.features = self.features.set_index('time')

        return self

    def __add_features(self, window: pd.DataFrame, column_name: str):
        """
        Adds features mean, max, med, min, q25, q75, and std for the specified column_name
        NOTE: Erases all other columns on returned object, store original DataFrameBuilder in variable if want to access original labels
        """

        data = {
            'mean': window[column_name].mean(), 
            'max': window[column_name].max(),
            'med': window[column_name].median(), 
            'min': window[column_name].min(),
            'q25': window[column_name].quantile(0.25),
            'q75': window[column_name].quantile(0.75),
            'std': window[column_name].std(),
            'variance': window[column_name].var()
        }
        df = pd.DataFrame()
        df = df._append(data,ignore_index=True)
        return df
        # return Preprocessor(pd.Dataframe(data), self.sample_rate)

    def finish_build(self):
        return self.features

if __name__ == "__main__":
    # df = pd.read_csv("/Users/joshuagu/CICS328_Assignments/cs328-projectproposal-group-2/data/walking/abnormal/limp_walking/joshua_limp1_accelerometer.csv") 
    # df = pd.read_csv("/Users/joshuagu/CICS328_Assignments/cs328-projectproposal-group-2/data/walking/normal/joshua_normal_Accelerometer1.csv") 
    
    combined_dataframe = CSV_Builder(sample_rate=100).add_csv_files_in_directory(
        # relative_path="/Users/joshuagu/CICS328_Assignments/cs328-projectproposal-group-2/data/walking/abnormal/limp_walking",
        relative_path="./data/walking/abnormal/limp_walking",
        # relative_path="./data/walking/normal",
        # relative_path="./data/not_walking",
        # relative_path="./data/walking/abnormal/duck_walking",
        # keyword="ccelerometer",
        keyword="yroscope",
        trim=True
    ).finish_build()

    # print("COMBINED IS" + str(combined_dataframe))
    print("combined has length " + str(len(combined_dataframe)))

    final_df = Preprocessor(combined_dataframe, 100).calc_magnitude("accel_mag", remove_gravity=False).remove_noise("accel_mag", "filtered_accel_mag").to_date_time().finish_build()
    # print(final_df)
    print("LENGTH OF DATAFRAME IS " + str(len(final_df)))

    feature_df = FeatureExtractor(final_df, 100, 5).extract_basic_features("filtered_accel_mag", "limping").finish_build()
    # print(feature_df)
    print("LENGTH OF DATAFRAME AFTER WINDOWS IS " + str(len(feature_df)))

    evaluater.plot_column(feature_df, ["mean", "variance", "max", "min"], "Window Number", "Angular Velocity (m/s)", "Rotation Features for Limp Walk", "basic_rotation_limping")
    # array = feature_df["mean"].to_numpy()
    # array = array[~np.isnan(array)]

    # plt.plot(array)
    # plt.show()
    print("done")

# def calc_magnitude(data: pd.DataFrame):
#     # Calculate magnitude  
#     data = data.copy() # create copy to remove warnings
#     data['accel_mag'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2) # absolute accel magnitude
#     data['accel_mag'] = data['accel_mag'] - data['accel_mag'].mean() # detrend: "remove gravity"

#     return data

# def remove_noise(data,sampling_rate):
#     # Low pass filter
#     cutoff = 5 # Hz
#     order = 2
#     b, a = butter(order, cutoff/(sampling_rate/2), btype='lowpass')
#     data['filtered_accel_mag'] = filtfilt(b, a, data['accel_mag'])

#     return data

# def add_features(window: pd.DataFrame):
#     data = {
#         'avg': [window['filtered_accel_mag'].mean()], 
#         'max': [window['filtered_accel_mag'].max()],
#         'med': [window['filtered_accel_mag'].median()], 
#         'min': [window['filtered_accel_mag'].min()],
#         'q25': [window['filtered_accel_mag'].quantile(0.25)],
#         'q75': [window['filtered_accel_mag'].quantile(0.75)],
#         'std': [window['filtered_accel_mag'].std()]
#     }
#     return pd.DataFrame(data)

# # Function to extract windows and features 
# def extract_features(data: pd.DataFrame, window_sec, sample_rate):
#     feature_dfs = []
#     resampled =  data.resample(f'{window_sec}s')
#     for timestamp, window in resampled:
#         window_with_features = add_features(window.copy())
#         window_with_features.insert(0, 'time', timestamp)
#         window_with_features['label'] = 1 if window['annotation'].sum() > 4 else 0
#         feature_dfs.append(window_with_features)
#     return pd.concat(feature_dfs)