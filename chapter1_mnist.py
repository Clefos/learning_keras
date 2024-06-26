from keras import models, layers, losses, optimizers, activations
from keras.datasets import mnist
from keras.utils import to_categorical
from matplotlib import pyplot as plt


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

my_slice = train_images[10:100]
print(my_slice.shape)

#digit = train_images[4]
#plt.imshow(digit, cmap=plt.cm.binary)
#plt.show()

network = models.Sequential()
network.add(layers.Dense(512, activation=activations.relu, input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation=activations.softmax))
network.compile(optimizer=optimizers.RMSprop(),
                loss=losses.categorical_crossentropy,
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)
