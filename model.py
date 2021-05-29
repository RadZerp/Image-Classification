from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils

def prepareData(data, labels, trainSize):
    xTrain, xTest, yTrain, yTest = train_test_split(data, labels, train_size = trainSize, random_state = 0)
    yTrain = np_utils.to_categorical(yTrain, num_classes = 10)
    yTest = np_utils.to_categorical(yTest, num_classes = 10)
    return xTrain, xTest, yTrain, yTest

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
