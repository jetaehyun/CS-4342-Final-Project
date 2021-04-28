import pandas


def importCSV(file_name):
    return pandas.read_csv(file_name)


def getLabels(data_frame):
    return data_frame.iloc[:,0].to_numpy()


def getData(data_frame):
    return data_frame.drop(data_frame.columns[0], axis=1).to_numpy()

# d = importCSV('train.csv')
# print(getLabels(d))
# print(getData(d))
