from parser import *
import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# tuned using GridSearchCV
# test_d is to make prediction, not the validation data
# best params: {'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
def tune_random_forest(train_d, test_d, params):

    Xtr = getData(train_d)
    ytr = getLabels(train_d)

    Xtr, Xte_val, ytr, yte_val = train_test_split(Xtr, ytr, test_size=0.2, random_state=42)

    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, params)
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte_val)
    print(f'prediction on val data: {metrics.accuracy_score(yte_val, pred)}')

    best_params = clf.best_params_
    results = clf.cv_results_
    print(f'best params: {best_params}')
    pprint.pprint(results)

    return best_params


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
