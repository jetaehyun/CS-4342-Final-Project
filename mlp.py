from parser import *
from data_augmentation import *

import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def one_hot_label(label):
    shape = (label.size, label.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(label.size)
    one_hot[rows, label] = 1

    return one_hot


def create_model(learning_rate=0.01, momentum=0, activation='relu', dropout_rate=0.1, neurons=512):
    # create model
    model = Sequential()

    model.add(Dense(neurons, activation=activation, input_dim=784))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def tune_MLP(train_d, epochs, batch_size, learning_rate, momentum, activation, dropout_rate, neurons):

    Xtr = getData(train_d)
    ytr = getLabels(train_d)
    ytr = one_hot_label(ytr)
    Xtr = np.divide(Xtr, 255)

    Xtr, Xte_val, ytr, yte_val = train_test_split(Xtr, ytr, test_size=0.2, random_state=42)

    model = KerasClassifier(build_fn=create_model, verbose=True)

    params = dict(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, momentum=momentum, neurons=neurons, dropout_rate=dropout_rate, activation=activation)

    clf = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=3)
    clf_results = clf.fit(Xte_val, yte_val)

    best_params = clf.best_params_
    print(f'best params: {best_params}')

# best params: {'activation': 'relu', 'batch_size': 50, 'dropout_rate': 0.2, 'learning_rate': 0.1, 'momentum': 0.8, 'neurons': 512}
def run_MLP(train_d, test_d, activation='relu', batch_size=50, dropout_rate=0.2, learning_rate=0.1, momentum=0.8, neurons=512, epochs=50, num_samples=60000):
    Xtr = getData(train_d)[0:num_samples:]
    ytr = getLabels(train_d)[0:num_samples:]

    idx = getIndicesOfLabel(train_d, 5)
    Xtr = dataAugmentationColor(Xtr, idx, 128)

    data_to_augment = getDataAtLabel(train_d, 6)
    Xtr, ytr = dataAugmentationRotate(Xtr, data_to_augment, ytr, 6)
    data_to_augment = getDataAtLabel(train_d, 5)
    Xtr, ytr = dataAugmentationRotate(Xtr, data_to_augment, ytr, 5)
    data_to_augment = getDataAtLabel(train_d, 5)
    Xtr, ytr = dataAugmentationTranslation(Xtr, data_to_augment, ytr, 5)

    Xte = getData(test_d)
    yte = getLabels(test_d)

    yte = one_hot_label(yte)
    ytr = one_hot_label(ytr)


    Xte = np.divide(Xte, 255)
    Xtr = np.divide(Xtr, 255)

    Xtr, Xte_val, ytr, yte_val = train_test_split(Xtr, ytr, test_size=0.2, random_state=42)

    model = Sequential()

    model.add(Dense(neurons, activation=activation, input_dim=784))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))


    sgd = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) # 92.44%


    model.fit(Xtr, ytr, epochs=epochs, batch_size=batch_size, validation_data=(Xte_val, yte_val))

    fCE, accuracy = model.evaluate(Xte, yte, batch_size=batch_size)

    print(f'fCE={fCE}, accuracy={accuracy}')

    yhat = np.argmax(model.predict(Xte, batch_size=batch_size), axis=-1)
    yte = np.argmax(yte, axis=-1)
    print(classification_report(yte, yhat))

    return yhat
