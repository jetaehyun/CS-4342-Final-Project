from keras.utils.np_utils import to_categorical
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from parser import *
from data_augmentation import *

# load train and test dataset
def load_dataset(train_d, test_d, sampleSize=60000):

    # load dataset -> Not Augmented
    # Xtr = getData(train_d)
    # ytr = getLabels(train_d)
    # Xte = getData(test_d)
    # yte = getLabels(test_d)

    # Load Dataset -> Augmented
    Xtr = getData(train_d)[0:sampleSize:]
    ytr = getLabels(train_d)[0:sampleSize:]

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

    # reshape dataset to have a single channel
    trainX = Xtr.reshape((Xtr.shape[0], 28, 28, 1))
    testX = Xte.reshape((Xte.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(ytr)
    testY = to_categorical(yte)
    return trainX, trainY, testX, testY


# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # define model
        model = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories


# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()


# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    # box and whisker plots of results
    # pyplot.boxplot(scores)
    # pyplot.show()


# run the test harness for evaluating a model
def run_test_harness(train_d, test_d):
    # load dataset
    trainX, trainY, testX, testY = load_dataset(train_d, test_d)
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # evaluate model
    scores, histories = evaluate_model(trainX, trainY)
    # learning curves
    # summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)

