import numpy as np
import matplotlib.pyplot as plt

from utils import loadMNISTLabels, loadMNISTImages
from ff import FF
## Loading the dataset
y_test = loadMNISTLabels('../MNIST_data/t10k-labels-idx1-ubyte')
y_train = loadMNISTLabels('../MNIST_data/train-labels-idx1-ubyte')

X_test = loadMNISTImages('../MNIST_data/t10k-images-idx3-ubyte')
X_train = loadMNISTImages('../MNIST_data/train-images-idx3-ubyte')

## random permutation of the input
# uncomment this to use a fixed random permutation of the images

perm = np.random.permutation(784)
X_test = X_test[perm,:]
X_train = X_train[perm,:]

## Parameters
layers_sizes = [784, 30, 10]  # flexible, but should be [784,...,10]
epochs = 10
eta = 0.1
batch_size = 20

## Training
net = FF(layers_sizes)
steps, test_acc = net.sgd(X_train, y_train, epochs, eta, batch_size, X_test, y_test)

## plotting learning curve and visualizing some examples from test set

plt.xlabel('number of epochs')
plt.ylabel('test accuracy')
plt.plot(steps, test_acc)
plt.show()


ran = np.random.permutation(9999)
X_perm = X_test[:, ran]
y_perm = y_test[:, ran]
indices = []
digit = 0 # which digit we are looking for
num_of_cur_digit = 0 # how many we have from "digit"
i = 0
while digit < 10:
    while num_of_cur_digit < 10:
        if np.argmax(np.transpose(y_perm)[i]) == digit:
            indices.append(i)
            num_of_cur_digit = num_of_cur_digit + 1
        i = i + 1
    digit = digit + 1
    num_of_cur_digit = 0



fig = plt.figure(figsize=(10,10))
for i in range(100):
    y_prediction = net.predict(X_perm[:, indices[i]])
    predicted_val = np.argmax(np.transpose(y_prediction))
    real_val = np.argmax(np.transpose(y_perm[:, indices[i]]))
    img = np.transpose(X_perm[:, indices[i]]).reshape(28,28)
    ax = fig.add_subplot(10,10,i + 1)
    plt.imshow(img)
    if predicted_val != real_val:
        plt.title(str(predicted_val), fontdict={'color':'red', 'fontsize':12})
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
            ax.spines[axis].set_color('red')
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
    else:
        plt.axis('off')

    fig.tight_layout(pad=0.3, w_pad=1, h_pad=0.8)
plt.show()