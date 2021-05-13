import tensorflow as tf
from tensorflow.keras import layers, losses, Model
from parser import *
from data_augmentation import *

# Within this load_dataset I also am reshaping the data to suit the GoogLeNet CNN
# The images need to be reshaped to 224x224x3 to suit the LeNet format
# See article below for reference
# https://ai.plainenglish.io/googlenet-inceptionv1-with-tensorflow-9e7f3a161e87

def load_dataset(train_d, test_d, sampleSize=60000):
    # load dataset -> Not Augmented
    # Xtr = getData(train_d)[0:sampleSize:]
    # ytr = getLabels(train_d)[0:sampleSize:]
    # Xte = getData(test_d)[0:sampleSize:]
    # yte = getLabels(test_d)[0:sampleSize:]

    # Load Data -> Augmented
    Xtr = getData(train_d)[0:sampleSize:]
    ytr = getLabels(train_d)[0:sampleSize:]

    idx = getIndicesOfLabel(train_d, 5)
    Xtr = dataAugmentationColor(Xtr, idx, 128)
    #
    data_to_augment = getDataAtLabel(train_d, 6)
    Xtr, ytr = dataAugmentationRotate(Xtr, data_to_augment, ytr, 6)
    data_to_augment = getDataAtLabel(train_d, 5)
    Xtr, ytr = dataAugmentationRotate(Xtr, data_to_augment, ytr, 5)
    data_to_augment = getDataAtLabel(train_d, 5)
    Xtr, ytr = dataAugmentationTranslation(Xtr, data_to_augment, ytr, 5)

    Xte = getData(test_d)
    yte = getLabels(test_d)

    Xtr = np.reshape(Xtr, (Xtr.shape[0], 28, 28))
    Xte = np.reshape(Xte, (Xte.shape[0], 28, 28))

    # reshape dataset to suit the googLeNet model
    Xtr = tf.pad(Xtr, [[0, 0], [2, 2], [2, 2]]) / 255
    Xte = tf.pad(Xte, [[0, 0], [2, 2], [2, 2]]) / 255
    Xtr = tf.expand_dims(Xtr, axis=3, name=None)
    Xte = tf.expand_dims(Xte, axis=3, name=None)
    Xtr = tf.repeat(Xtr, 3, axis=3)
    Xte = tf.repeat(Xte, 3, axis=3)
    xVal = Xtr[-2000:, :, :, :]
    yVal = ytr[-2000:]
    Xtr = Xtr[:-2000, :, :, :]
    ytr = ytr[:-2000]

    return Xtr, ytr, xVal, yVal

def inception(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool):
    path1 = layers.Conv2D(filters_1x1, (1, 1), padding='same',    activation='relu')(x)
    path2 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    path2 = layers.Conv2D(filters_3x3, (1, 1), padding='same', activation='relu')(path2)
    path3 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    path3 = layers.Conv2D(filters_5x5, (1, 1), padding='same', activation='relu')(path3)
    path4 = layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = layers.Conv2D(filters_pool, (1, 1), padding='same', activation='relu')(path4)
    return tf.concat([path1, path2, path3, path4], axis=3)

def initializeModel(Xtr):
    inp = layers.Input(shape=(32, 32, 3))
    input_tensor = layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=Xtr.shape[1:])(inp)

    x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(input_tensor)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.Conv2D(64, 1, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(192, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = inception(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool=32)
    x = inception(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool=64)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = inception(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16, filters_5x5=48, filters_pool=64)

    aux1 = layers.AveragePooling2D((5, 5), strides=3)(x)
    aux1 =layers.Conv2D(128, 1, padding='same', activation='relu')(aux1)
    aux1 = layers.Flatten()(aux1)
    aux1 = layers.Dense(1024, activation='relu')(aux1)
    aux1 = layers.Dropout(0.7)(aux1)
    aux1 = layers.Dense(10, activation='softmax')(aux1)

    x = inception(x, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5_reduce=24, filters_5x5=64, filters_pool=64)
    x = inception(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256, filters_5x5_reduce=24, filters_5x5=64, filters_pool=64)
    x = inception(x, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288, filters_5x5_reduce=32, filters_5x5=64, filters_pool=64)

    aux2 = layers.AveragePooling2D((5, 5), strides=3)(x)
    aux2 = layers.Conv2D(128, 1, padding='same', activation='relu')(aux2)
    aux2 = layers.Flatten()(aux2)
    aux2 = layers.Dense(1024, activation='relu')(aux2)
    aux2 = layers.Dropout(0.7)(aux2)
    aux2 = layers.Dense(10, activation='softmax')(aux2)

    x = inception(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool=128)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = inception(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool=128)
    x = inception(x, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384, filters_5x5_reduce=48, filters_5x5=128, filters_pool=128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(10, activation='softmax')(x)

    model = Model(inputs = inp, outputs = [out, aux1, aux2])

    model.compile(optimizer='adam',
                  loss=[losses.sparse_categorical_crossentropy,
                        losses.sparse_categorical_crossentropy,
                        losses.sparse_categorical_crossentropy],
                  loss_weights=[1, 0.3, 0.3],
                  metrics=['accuracy'])

    return model

def trainModel(train_x, train_y, x_val, y_val, model):
    history = model.fit(train_x, [train_y, train_y, train_y], validation_data=(x_val, [y_val, y_val, y_val]), batch_size=64, epochs=1)
    #print(history['accuracy'])

def run_GNet(train_d, test_d):
    Xtr, ytr, Xte, yte = load_dataset(train_d, test_d)
    model = initializeModel(Xtr)
    trainModel(Xtr, ytr, Xte, yte, model)
