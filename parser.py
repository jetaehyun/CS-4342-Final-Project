import pandas
import numpy as np

def importCSV(file_name):
    return pandas.read_csv(file_name)


def getLabels(data_frame):
    return data_frame.iloc[:,0].to_numpy()


def getData(data_frame):
    return data_frame.drop(data_frame.columns[0], axis=1).to_numpy()


def getIndicesOfLabel(data_frame, label):
    rows = np.where(data_frame.label == label)
    idx = data_frame.iloc[rows].to_numpy()
    return idx[:,0]


def getDataAtLabel(data_frame, label):
    rows = np.where(data_frame.label==label)
    desired_rows = data_frame.iloc[rows].to_numpy()

    return desired_rows[:,1:]
