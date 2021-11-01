#import tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
# load fashion MNIST data from the keras library
mnist = tf.keras.datasets.fashion_mnist

#creating training and testing data
(training_images,training_labels),(test_images,test_labels) = mnist.load_data()

#to see what the above values look like
np.set_printoptions(linewidth=200)
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

#the values in the image are from 0 to 255 so we normalize the data to use only 0 and 1 by dividing by 255
training_images =training_images/255
test_images =test_images/255

# design the model:
# first layer flattens the input of 28X28 into a simple linear array
# 2nd layer = hidden layer that handles all the variable in the function
# 3rd layer specifies that we have 10 classes of clothing in the data set
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# building the model
# compile the model and train it by using model.fit

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10)
# for epoch =10 the model is 90.91% accurate

# test the models
model.evaluate(test_images, test_labels)
print("")
# we get the test is 88.05% accurate


classifications = model.predict(test_images)
print(classifications[0])