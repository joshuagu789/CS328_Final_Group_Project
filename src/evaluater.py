"""
Runs trained model and displays model performance such as confusion matrix
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_column(dataframe: pd.DataFrame, column_names: str, x_name: str, y_name: str, title: str, output_name: str):

    # np_dict = {}
    where_are_nans = []

    for column_name in column_names:
        np_array = dataframe[column_name].to_numpy()
        # np_dict[column_name] = np_array 
        nans = np.isnan(np_array)

        if len(nans) > len(where_are_nans):
            where_are_nans = nans



    for column_name in column_names:

        array = dataframe[column_name].to_numpy()
        # array = array[~np.isnan(array)] # sometimes array has NaN
        array = array[~where_are_nans] # sometimes array has NaN

        # x = np.arange(1, len(array)+1)
        print("length is " + str(len(array)) + " for " + column_name)
        plt.plot(array, label = column_name)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.legend()

    path = "./images/" + output_name + ".png"
    if os.path.exists(path):
        print("DID NOT SAVE IMAGE, IT ALREADY EXISTS")
    else:
        plt.savefig("./images/" + output_name + ".png")
    # plt.savefig('my_figure.png') 

    plt.show()
