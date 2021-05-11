from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

def dataAugmentationRotate(X1, X2, y1, label, degree=20):
    X2 = np.reshape(X2, (X2.shape[0], 28, 28))

    for i in range(X2.shape[0]):
        X2[i] = ndimage.rotate(X2[i], degree, reshape=False)


    X2 = np.reshape(X2, (X2.shape[0], 784))
    x_new = np.vstack((X1, X2))

    y2 = np.full((X2.shape[0],), label)
    y_new = np.hstack((y1, y2))

    shuffler = np.random.permutation(x_new.shape[0])

    return x_new, y_new


def dataAugmentationTranslation(X1, X2, y1, label, shift=-2):
    X2 = np.reshape(X2, (X2.shape[0], 28, 28))

    for i in range(X2.shape[0]):
        X2[i] = np.roll(X2[i], shift, axis=1)

    X2 = np.reshape(X2, (X2.shape[0], 784))
    x_new = np.vstack((X1, X2))

    y2 = np.full((X2.shape[0],), label)
    y_new = np.hstack((y1, y2))

    shuffler = np.random.permutation(x_new.shape[0])

    return x_new, y_new


def showTwoImages(img1, img2):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img1)
    axarr[1].imshow(img2)

    plt.show()
