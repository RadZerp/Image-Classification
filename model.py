from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from tensorflow import keras
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.layers.core import *

def prepareData(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size = 0.80, random_state = 0)
    y_train = np_utils.to_categorical(y_train, num_classes = 10)
    y_test = np_utils.to_categorical(y_test, num_classes = 10)
    return X_train, X_test, y_train, y_test

def defineModel():
    model = Sequential([
        #https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        
        Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'),
        MaxPooling2D(pool_size = (2, 2)),
        
        Conv2D(32, (3, 3), activation = 'relu'),
        MaxPooling2D(pool_size = (2, 2)),
        
        Conv2D(64, (3, 3), activation = 'relu'),
        MaxPooling2D(pool_size = (2, 2)),
        
        Flatten(),

        Dense(64, activation = 'relu'),
        Dropout(0.5),

        Dense(10, activation = 'sigmoid')
    ])
    opt = keras.optimizers.Adam(learning_rate = 0.01)
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = opt, 
        metrics = ['accuracy']
    )
    return model

# Checked matplotlib for how to create two-axis graphs:
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html
def plot(results):
    plt.clf()
    history_dict = results.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    val_accuracy = history_dict['val_accuracy']
    epochs = range(1, (len(history_dict['loss']) + 1))
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, loss_values, 'b', label = 'Training loss', c = 'lightgreen')
    ax1.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.plot(epochs, val_accuracy, 'b', label = 'Validation accuracy', c = 'red')
    plt.title('Training/validation loss and accuracy')
    
    fig.tight_layout()
    plt.show()
