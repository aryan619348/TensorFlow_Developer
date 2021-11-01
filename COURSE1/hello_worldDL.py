import tensorflow as tf
import numpy as np
from tensorflow import keras
# FIRST PROGRAM
#simplest possible neural network. It has 1 layer, and that layer has 1 neuron, and the input shape to it is just 1 value
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

#providing data
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#training the data, 500 runs
model.fit(xs, ys, epochs=500)

#check prediction for x=10
print(model.predict([10.0]))
