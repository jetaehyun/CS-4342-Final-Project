from parser import *

import numpy as np
import pandas
import matplotlib.pyplot as plt


def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None):

    w = 0.01 * np.random.rand(trainingImages.shape[0], trainingLabels.shape[0])
    rnd = int(trainingImages.shape[1] / batchSize)
    numEpoch = 10

    # fCE = []
    for epoch in range(numEpoch):
        for i in range(rnd):

            start = i * batchSize
            end = start + batchSize

            x, y = trainingImages[:,start:end], trainingLabels[:,start:end]
            gradf = gradient(x, y, w, batchSize)

            w -= epsilon * gradf
            # if epoch == 0 and i < 20:
                # print(CELoss(x, y, w))
            # if epoch == numEpoch - 1 and i > rnd - 21:
                # print(CELoss(x, y, w))

        # fCE.append(CELoss(trainingImages, trainingLabels, w))

    print(f'test set: CE loss = {CELoss(testingImages, testingLabels, w)}, PC = {accuracy(testingImages, testingLabels, w)*100}%')

    # plotGraph([i + 1 for i in range(numEpoch)], fCE, "epoch", "fCE")

    return w

def plotGraph(xData, yData, xAxisName, yAxisName):
    plt.plot(xData, yData)
    plt.ylabel(yAxisName)
    plt.xlabel(xAxisName)
    plt.show()

def CELoss(x, y, w):
    yhat_k = np.log(softmax(x, w))

    summation = np.sum(yhat_k * y)

    return (-1 / x.shape[1]) * summation

def accuracy(x, y, w):
    yhat = softmax(x, w)

    return np.mean(np.argmax(y, axis=0)==np.argmax(yhat, axis=0))

def gradient(x, y, w, batchSize):
    yhat_k = softmax(x, w)
    gradf = x.dot(yhat_k.T - y.T) / batchSize

    return gradf

def softmax(x, w):
    Z = x.T.dot(w)
    preActivScores = np.exp(Z)

    return preActivScores.T / np.sum(preActivScores, axis=1)

def one_hot_label(label, shuffle):
    shape = (label.size, label.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(label.size)
    one_hot[rows, label] = 1

    if len(shuffle) == 0:
        return one_hot.T

    return one_hot[shuffle].T

def transImage(images, shuffle):
    row = np.ones((1, images.shape[0])).T
    images = np.hstack((images, row))

    if len(shuffle) == 0:
        return images.T

    return images[shuffle].T

if __name__ == "__main__":

    train_d = importCSV('train.csv')
    test_d = importCSV('Dig-MNIST.csv')

    trainingImages = getData(train_d)
    trainingLabels = getLabels(train_d)
    testingImages = getData(test_d)
    testingLabels = getLabels(test_d)

    trainingImages = np.divide(trainingImages, 255.0)
    testingImages = np.divide(testingImages, 255.0)

    # shuffle permuations
    shuffler_tr = np.random.permutation(len(trainingImages))

    # append bias and create one hot vector for training and testing images
    trainingImages = transImage(trainingImages, shuffler_tr)
    trainingLabels = one_hot_label(trainingLabels, shuffler_tr)

    testingImages = transImage(testingImages, [])
    testingLabels = one_hot_label(testingLabels, [])

    W = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=500).T

    # plot the 10 classes
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(np.reshape(W[i][:-1], (28, 28)), cmap='gray')
        plt.axis('off')
    plt.show()
