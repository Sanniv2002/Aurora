import aurora as ar
from keras.utils import np_utils
from keras.datasets import mnist
import dill as pickle
# Prepare the dataset


def preprocess_data(x, y, limit):
    # make input data ready for the network
    x = x.reshape(x.shape[0], 28 * 28, 1)
    # all the values will be between 0-1 which makes the network easy to learn
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    #     the respective number:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)

# neural network
l1 = ar.Layer(28 * 28, 400)
a1 = ar.Sigmoid()
l2 = ar.Layer(400, 10)
a2 = ar.Sigmoid()
network = [l1, a1, l2, a2]

# train
ar.train(network, ar.mse, ar.mse_prime, x_train,
         y_train, epochs=100, learning_rate=0.01)

print()
ar.save_model(network, "MNIST_Model_1")
print("Model Saved Successfully")
