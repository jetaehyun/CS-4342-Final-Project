from parser import *
import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

def one_hot_label(label):
    shape = (label.size, label.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(label.size)
    one_hot[rows, label] = 1

    return one_hot

def run_MLP(train_d, test_d):
    Xtr = getData(train_d)
    ytr = getLabels(train_d)
    Xte = getData(test_d)
    yte = getLabels(test_d)

    yte = one_hot_label(yte)
    ytr = one_hot_label(ytr)


    Xte = np.divide(Xte, 255)
    Xtr = np.divide(Xtr, 255)

    Xtr, Xte_val, ytr, yte_val = train_test_split(Xtr, ytr, test_size=0.2, random_state=42)

    model = Sequential()

    model.add(Dense(512, activation='relu', input_dim=784))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))


    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) # 92.44%


    model.fit(Xtr, ytr, epochs=20, batch_size=100, validation_data=(Xte_val, yte_val))

    fCE, accuracy = model.evaluate(Xte, yte, batch_size=100)

    print(f'fCE={fCE}, accuracy={accuracy}')

    yhat = np.argmax(model.predict(Xte, batch_size=100), axis=-1)
    # print(yhat)

    return yhat
