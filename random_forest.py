from parser import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# import numpy as np


def run_random_forest(train_d, test_d, n_estimators):
    Xtr = getData(train_d)
    Ytr = getLabels(train_d)
    Xte = getData(test_d)
    Yte = getLabels(test_d)

    # n_estimators is number of trees
    rand_forest = RandomForestClassifier(n_estimators=n_estimators)
    rand_forest.fit(Xtr, Ytr)
    pred = rand_forest.predict(Xte)
    print(f'accuracy: {metrics.accuracy_score(Yte, pred)}')

    # shuffle_tr = np.random.permutation(len(Xtr))
    # Xtr = Xtr[shuffle_tr]
    # Ytr = Ytr[shuffle_tr]
    #
    # rand_forest_mix = RandomForestClassifier(n_estimators=n_estimators)
    # rand_forest_mix.fit(Xtr, Ytr)
    # pred = rand_forest_mix.predict(Xte)
    # print(f'accuracy: {metrics.accuracy_score(Yte, pred)}')

    return pred
