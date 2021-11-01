# There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

# Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs --
# i.e. you should stop training once you reach that level of accuracy.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# add mnist folder
path = "mnist.npz"
def train_mnist():
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>0.99 and logs.get('accuracy')is not None):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training=True
    callbacks=myCallback()
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    x_train =x_train/255.0
    x_test =x_test/255.0
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                        tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print("TRAINING DATA")
    history = model.fit(x_train,y_train,epochs=10,callbacks=[callbacks])
    print("TESTING DATA")
    model.evaluate(x_test, y_test)
    return history.epoch, history.history['accuracy'][-1]

train_mnist()