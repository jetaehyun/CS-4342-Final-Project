import pandas
import numpy as np


def importCSV(file_name):
    return pandas.read_csv(file_name)


def getLabels(data_frame):
    return data_frame.iloc[:,0].to_numpy()


def getData(data_frame):
    return data_frame.drop(data_frame.columns[0], axis=1).to_numpy()


def getDataAtLabel(data_frame, label):
    rows = np.where(data_frame.label==label)

    desired_rows = data_frame.iloc[rows].to_numpy()

    return desired_rows[:,1:]


def dataAugmentationFlip(X1, X2, y1, label):
    X2 = np.reshape(X2, (X2.shape[0], 28, 28))

    for i in range(X2.shape[0]):
        X2[i] = np.fliplr(X2[i])

    X2 = np.reshape(X2, (X2.shape[0], 784))
    x_new = np.vstack((X1, X2))

    y2 = np.full((X2.shape[0],), label)
    y_new = np.hstack((y1, y2))

    shuffler = np.random.permutation(x_new.shape[0])
    # return x_new[shuffler], y_new[shuffler]
    return x_new, y_new





# d = importCSV('train.csv')
# print(getLabels(d))
# print(getData(d))
