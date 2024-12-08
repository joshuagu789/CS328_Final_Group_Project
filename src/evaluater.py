"""
Runs trained model and displays model performance such as confusion matrix
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_column(dataframe: pd.DataFrame, column_name: str, x_name: str, y_name: str, title: str):
    array = dataframe["mean"].to_numpy()
    array = array[~np.isnan(array)] # sometimes array has NaN

    plt.plot(array)
    plt.show()