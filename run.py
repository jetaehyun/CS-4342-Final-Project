from softmax_regression import *
from random_forest import *
from cnn import *

if __name__ == '__main__':
    train_d = importCSV('train.csv')
    test_d = importCSV('Dig-MNIST.csv')

    # run_softmax_reg(train_d, test_d, 0.1, 500)
    # run_random_forest(train_d, test_d)
    run_test_harness(train_d, test_d)
