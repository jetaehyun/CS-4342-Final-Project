from sklearn.metrics import accuracy_score
from parser import *
from sklearn.ensemble import GradientBoostingClassifier


def setupData(train_d, test_d):
    train_x = getData(train_d)
    train_y = getLabels(train_d)
    test_x = getData(test_d)
    test_y = getLabels(test_d)

    return train_x, test_x, train_y, test_y

def runModel(train_x, train_y, test_x, test_y):
    clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=1, random_state=0).fit(train_x, train_y)
    gradOutput = clf.predict(test_x)
    print(accuracy_score(test_y, gradOutput))

def runBoosting(train_d, test_d):
    print("Running Models")
    train_x, test_x, train_y, test_y, = setupData(train_d, test_d)
    runModel(train_x, train_y, test_x, test_y)