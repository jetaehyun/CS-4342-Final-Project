from parser import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# tuned using GridSearchCV
def tune_random_forest(train_d, test_d, params):

    Xtr = getData(train_d)
    ytr = getLabels(train_d)
    Xte = getData(test_d)
    yte = getLabels(test_d)

    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, params)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print(f'tuned: {metrics.accuracy_score(yte, pred)}')

    return pred


def run_random_forest(train_d, test_d, n_estimators=100):
    Xtr = getData(train_d)
    ytr = getLabels(train_d)
    Xte = getData(test_d)
    yte = getLabels(test_d)

    Xtr, Xte_ex, ytr, yte_ex = train_test_split(Xtr, ytr, test_size=0.2, random_state=42)


    # n_estimators is number of trees
    rand_forest = RandomForestClassifier(n_estimators=n_estimators)
    rand_forest.fit(Xtr, ytr)
    pred = rand_forest.predict(Xte_ex)
    print(f'split test set accuracy: {metrics.accuracy_score(yte_ex, pred)}')

    pred = rand_forest.predict(Xte)
    print(f'test set accuracy: {metrics.accuracy_score(yte, pred)}')

    return pred
