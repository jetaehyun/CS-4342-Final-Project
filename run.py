from softmax_regression import *
from random_forest import *
from cnn import *
from boosting import *
from google_net import *
from mlp import *
import os

if __name__ == '__main__':
    # hide error msgs from tensorflow regarding CPU incapabilities for performance
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    train_d = importCSV('train.csv')
    test_d = importCSV('Dig-MNIST.csv')

    # run_softmax_reg(train_d, test_d, 0.1, 500)
    # run_test_harness(train_d, test_d)

    # random_forest
    # params = {'n_estimators': [50, 100, 200], 'min_samples_leaf': [1, 2], 'min_samples_split': [2, 3]}
    # tune_random_forest(train_d, test_d, params)
    # run_random_forest(train_d, test_d)
    # run_test_harness(train_d, test_d)
    # params = {'n_estimators': [50, 200], 'min_samples_leaf': [1, 2], 'min_samples_split': [2, 3]}
    # tune_random_forest(train_d, test_d, params)
    # runBoosting(train_d, test_d)

    # GoogLeNet
    run_GNet(train_d, test_d)



    # mlp
    # batch_size = [50, 100, 500]
    # epochs = [10, 50]
    # activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    # dropout_rate = [0.0, 0.2, 0.4]
    # neurons = [8, 16, 32, 64, 128, 256, 512]
    # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    # learn_rate = [0.001, 0.01, 0.1]
    # tune_MLP(train_d, epochs=epochs,batch_size=batch_size,learning_rate=learn_rate, momentum=momentum, activation=activation, dropout_rate=dropout_rate, neurons=neurons)
    # best params: {'activation': 'relu', 'batch_size': 50, 'dropout_rate': 0.2, 'learning_rate': 0.1, 'momentum': 0.8, 'neurons': 512}
    # run_MLP(train_d, test_d)
