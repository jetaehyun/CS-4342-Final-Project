import pandas
import sklearn.svm

if __name__ == "__main__":
    d_train = pandas.read_csv("train.csv")
    y_train = d_train.label.to_numpy()
    X_train = d_train.values[:,1:]

    d_test = pandas.read_csv("test.csv")
    ID = d_test.id.to_numpy()
    X_test = d_test.values[:,1:]

    svm = sklearn.svm.SVC(kernel='rbf', gamma=0.001)
    svm.fit(X_train, y_train)
    yHat = svm.predict(X_test)
    df = pandas.DataFrame({'id': ID,
                           'label': yHat})
    df.to_csv('predictions.csv',index=False)
    print("Done")