import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

def prepareData(data, labels):
    #labels = tf.convert_to_tensor(labels)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size = 0.7, random_state = random.seed(100))
    
    return X_train, X_test, y_train, y_test


from keras.models import Sequential
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.layers.core import *
from tensorflow.keras import regularizers

def defineModel():
    # model = Sequential([
    #     Dense(16, input_shape = (200, 200, 3), activation = 'relu', kernel_regularizer = regularizers.l1(0.01)),
    #     Dropout(0.02),
    #     Conv2D(32, kernel_size = (3, 3), activation = 'tanh', padding = 'valid'),
    #     Dropout(0.02),
    #     Conv2D(32, kernel_size = (3, 3), activation = 'tanh', padding = 'valid'),
    #     Flatten(),
    #     Dense(11, activation = 'relu')
    # ])
    # model.compile(
    #     loss = 'categorical_crossentropy', 
    #     optimizer = 'adam', 
    #     metrics = ['accuracy']
    # )
    model = Sequential()
    model.add(Dense(5, input_shape = (200, 200, 3), activation = 'relu'))
    model.add(Conv2D(3, kernel_size = (3, 3), activation = 'tanh', padding = 'valid'))# 1.1 NO 2D ONLY DENSE!
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(11, activation='relu'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    model.summary()

    return model
    # input1 = Dense(5, input_shape = (None, None, 3), activation = 'relu')
    # conv1 = Conv2D(32, (3, 3), input_shape=(3, None, None), activation="relu")(inpu1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv2 = Conv2D(32, (3, 3), activation="relu")(pool1)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # flat1 = Flatten()(pool2)
    # dense1 = Dense(16, activation="relu")(flat1)
    # dense2 = Dense(1, activation="sigmoid")(dense1)
    # model = Model(inputs=input1, outputs=dense2)
    # model.compile(loss='mse', optimizer='adadelta', metrics=['mse', 'mae'])
    # return model

def plot(results):
    plt.clf()
    history_dict = results.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, (len(history_dict['loss']) + 1))
    plt.plot(epochs, loss_values, 'bo', label = 'Training loss', c = 'lightgreen')
    plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    
# numpy array combining
def numpyfy(df):
    arr = []
    df_numpy = df.to_numpy()
    # print(df_numpy[:2])
    for i in df_numpy:
        arr.append(i)
    arr = np.asarray(arr)
    #print(arr.shape), returns 4 dimensional array
    return arr